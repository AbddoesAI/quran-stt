#!/usr/bin/env python3
"""
eval_run.py
-----------
Baseline evaluation script for Islamic STT pipeline.
Measures WER, CER (via jiwer), Quran recall, Flag rate, Arabic Drift, and more.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from datetime import datetime

try:
    import jiwer
except ImportError:
    jiwer = None
    print(
        "Warning: 'jiwer' not installed. WER/CER will not be calculated. Run 'pip install jiwer'.",
        file=sys.stderr,
    )


def load_metadata(dataset_dir: str) -> list[dict]:
    meta_dir = os.path.join(dataset_dir, "metadata")
    files = glob.glob(os.path.join(meta_dir, "*.json"))
    metadata_list = []
    for f in files:
        with open(f, encoding="utf-8") as f_in:
            metadata_list.append(json.load(f_in))
    return metadata_list


def compute_arabic_drift(diagnostics: list[dict]) -> int:
    """Track unexpected language transitions (e.g., ur -> ar -> nl -> ar)."""
    drift_count = 0
    prev_lang = None
    for seg in diagnostics:
        lang = seg.get("detected_lang", "und")
        if prev_lang and prev_lang in ("ur", "ar", "en") and lang not in ("ur", "ar", "en", "und"):
            drift_count += 1
        prev_lang = lang
    return drift_count


def compute_mixed_script_ratio(diagnostics: list[dict]) -> float:
    """Proxy for mixed-script corruption (e.g., Arabic chars in mostly Urdu/English word)."""
    mixed_words = 0
    total_words = 0
    mixed_pattern = re.compile(r"(?:[A-Za-z][\u0600-\u06FF])|(?:[\u0600-\u06FF][A-Za-z])")

    for seg in diagnostics:
        text = seg.get("text_preview", "")
        words = text.split()
        total_words += len(words)
        for w in words:
            if mixed_pattern.search(w):
                mixed_words += 1

    return mixed_words / max(total_words, 1)


def summarize_diagnostics(diagnostics_path: str) -> dict:
    if not os.path.isfile(diagnostics_path):
        return {}

    rows = []
    with open(diagnostics_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    total = len(rows)
    if total == 0:
        return {"total_segments": 0}

    flagged = sum(1 for r in rows if r.get("is_flagged"))
    quran = sum(1 for r in rows if r.get("quran_match"))
    hadith = sum(1 for r in rows if r.get("hadith_match"))

    arabic_drift = compute_arabic_drift(rows)
    mixed_script_ratio = compute_mixed_script_ratio(rows)

    retry_triggered_count = sum(1 for r in rows if r.get("retry_triggered"))
    retry_trigger_rate = retry_triggered_count / total if total > 0 else 0.0

    # Simple proxy for arabic retry success for now (will be updated in Step 3)
    arabic_retry_success = 0.0

    worst_segments = []
    for r in rows:
        msr = r.get("mixed_script_ratio", 0.0)
        logp = r.get("avg_logprob", 0.0)
        flagged = r.get("is_flagged", False)
        retried = r.get("retry_triggered", False)
        rep_ngram = r.get("repeated_ngram_score", 0.0)

        if msr > 0.10 or logp < -1.0 or flagged or retried or rep_ngram > 0.4:
            worst_segments.append(
                {
                    "segment_id": r.get("segment_id"),
                    "original": r.get("original_text_preview", ""),
                    "retry": r.get("text_preview", "") if retried else "",
                    "selected": r.get("text_preview", ""),
                    "confidence_delta": r.get("confidence_delta", 0.0),
                    "mixed_script_ratio": msr,
                    "language_drift": r.get("detected_lang", "und")
                    not in ("ur", "ar", "en", "und"),
                    "retry_triggered": retried,
                    "retry_reason": r.get("retry_reason", []),
                }
            )

    return {
        "stats": {
            "total_segments": total,
            "flagged_rate": flagged / total if total > 0 else 0.0,
            "quran_matches": quran,
            "hadith_matches": hadith,
            "arabic_drift_events": arabic_drift,
            "mixed_script_ratio": mixed_script_ratio,
            "retry_trigger_rate": retry_trigger_rate,
            "arabic_retry_success": arabic_retry_success,
        },
        "worst_segments": worst_segments,
    }


def compute_wer_cer(ref_text: str, hyp_text: str) -> tuple[float, float]:
    if jiwer is None:
        return 0.0, 0.0
    try:
        wer = jiwer.wer(ref_text, hyp_text)
        cer = jiwer.cer(ref_text, hyp_text)
        return wer, cer
    except Exception as e:
        print(f"Error computing WER/CER: {e}", file=sys.stderr)
        return 0.0, 0.0


def evaluate_file(meta: dict, dataset_dir: str, output_dir: str) -> dict:
    """Evaluate a single file based on its generated output vs reference."""
    audio_file = meta.get("file", "")
    ref_file_name = meta.get("reference", "")
    base_name = os.path.splitext(audio_file)[0]

    # Paths
    ref_path = os.path.join(dataset_dir, meta.get("type", ""), ref_file_name)
    # Assume the pipeline generated output in output_dir/base_name/transcript.txt
    hyp_path = os.path.join(output_dir, base_name, "transcript.txt")
    diag_path = os.path.join(output_dir, base_name, "diagnostics.jsonl")

    res = {
        "file": audio_file,
        "type": meta.get("type", "unknown"),
        "wer": 0.0,
        "cer": 0.0,
        "quran_recall": 0.0,  # Placeholder until we have ground truth ayah alignment
    }

    # If reference and hypothesis exist, compute WER/CER
    if os.path.isfile(ref_path) and os.path.isfile(hyp_path):
        with open(ref_path, encoding="utf-8") as f:
            ref_text = f.read()
        with open(hyp_path, encoding="utf-8") as f:
            hyp_text = f.read()

        wer, cer = compute_wer_cer(ref_text, hyp_text)
        res["wer"] = wer
        res["cer"] = cer
    else:
        print(f"Warning: Missing reference or hypothesis for {audio_file}", file=sys.stderr)

    diag_stats = summarize_diagnostics(diag_path)
    res["worst_segments"] = diag_stats.get("worst_segments", [])
    res.update(diag_stats.get("stats", {}))

    return res


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline Evaluation Script")
    parser.add_argument("--dataset", default="eval_dataset", help="Path to eval dataset")
    parser.add_argument("--output", default="eval_results", help="Path to store/read results")
    parser.add_argument("--version-tag", default="", help="Snapshot version tag (e.g., v1)")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    version = args.version_tag or datetime.now().strftime("v%Y%m%d_%H%M%S")
    version_dir = os.path.join(args.output, version)
    os.makedirs(version_dir, exist_ok=True)

    metadata_list = load_metadata(args.dataset)
    if not metadata_list:
        print(f"No metadata JSON files found in {args.dataset}/metadata. Exiting.", file=sys.stderr)
        sys.exit(0)

    print(f"Found {len(metadata_list)} evaluation records.")

    results = []
    for meta in metadata_list:
        res = evaluate_file(
            meta, args.dataset, args.output
        )  # assuming hyp generated in eval_results/
        results.append(res)

    if not results:
        print("No results to aggregate.")
        return

    # Aggregate
    total_files = len(results)
    avg_wer = sum(r.get("wer", 0) for r in results) / total_files
    avg_cer = sum(r.get("cer", 0) for r in results) / total_files
    avg_flag = sum(r.get("flagged_rate", 0) for r in results) / total_files
    total_drift = sum(r.get("arabic_drift_events", 0) for r in results)
    avg_mixed = sum(r.get("mixed_script_ratio", 0) for r in results) / total_files
    total_quran = sum(r.get("quran_matches", 0) for r in results)

    # Extract worst segments from all results
    all_worst_segments = []
    for r in results:
        file_worst = r.pop("worst_segments", [])
        for ws in file_worst:
            ws["file"] = r["file"]
            all_worst_segments.append(ws)

    summary = {
        "total_files": total_files,
        "avg_wer": avg_wer,
        "avg_cer": avg_cer,
        "avg_flag_rate": avg_flag,
        "total_arabic_drift_events": total_drift,
        "avg_mixed_script_ratio": avg_mixed,
        "total_quran_matches_detected": total_quran,
        "avg_retry_trigger_rate": sum(r.get("retry_trigger_rate", 0) for r in results)
        / total_files,
    }

    print("\n=== Evaluation Summary ===")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    report = {"summary": summary, "details": results}

    report_path = os.path.join(version_dir, "eval_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved report snapshot to {report_path}")

    worst_path = os.path.join(version_dir, "worst_segments.json")
    with open(worst_path, "w", encoding="utf-8") as f:
        json.dump(all_worst_segments, f, indent=2)
    print(f"Saved worst segments to {worst_path}")


if __name__ == "__main__":
    main()
