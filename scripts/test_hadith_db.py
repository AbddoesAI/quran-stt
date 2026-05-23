"""Quick smoke test for the local Hadith DB matcher."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from islamic_stt.matchers.hadith_db import LocalHadithMatcher

matcher = LocalHadithMatcher(db_path="data/hadith.db")

# Test 1: Famous hadith - "Actions are by intentions"
result = matcher.match("إنما الأعمال بالنيات وإنما لكل امرئ ما نوى")
if result:
    print(
        f"TEST 1 PASS: {result.collection} #{result.hadith_number} conf={result.confidence:.2f} paraphrase={result.is_paraphrase}"
    )
else:
    print("TEST 1 FAIL: No match for 'Innamal a'mal bin-niyyat'")

# Test 2: "The religion is sincerity"
result2 = matcher.match("الدين النصيحة")
if result2:
    print(
        f"TEST 2 PASS: {result2.collection} #{result2.hadith_number} conf={result2.confidence:.2f}"
    )
else:
    print("TEST 2 FAIL: No match for 'Ad-deen an-naseeha'")

# Test 3: Short text (should return None - below threshold)
result3 = matcher.match("بسم")
print(
    f"TEST 3 {'PASS' if result3 is None else 'FAIL'}: Short text correctly rejected"
    if result3 is None
    else "TEST 3 FAIL: Short text matched"
)

# Test 4: Partial hadith
result4 = matcher.match("من حسن إسلام المرء تركه ما لا يعنيه")
if result4:
    print(
        f"TEST 4 PASS: {result4.collection} #{result4.hadith_number} conf={result4.confidence:.2f} paraphrase={result4.is_paraphrase}"
    )
else:
    print("TEST 4 FAIL: No match for 'Min husni islam'")

matcher.close()
print("\nAll smoke tests complete.")
