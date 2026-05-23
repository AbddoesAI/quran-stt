from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Matcher(Protocol):
    """Protocol all matchers must satisfy.

    Implementations must accept a raw text string and return either
    a match dataclass or None.  The pipeline relies on this contract
    to iterate over matchers generically.
    """

    def match(self, text: str) -> Any | None: ...
