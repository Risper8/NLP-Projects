from __future__ import annotations
import re


CLAIM_REFERENCE_PATTERN = re.compile(r"\b[A-Za-z]{2,6}-?\d{4,}\b")


def extract_claim_reference(text: str) -> str | None:
    match = CLAIM_REFERENCE_PATTERN.search(text)
    return match.group(0).upper() if match else None


def extract_claim_references(text: str) -> list[str]:
    """All distinct claim references in text, in first-seen order."""
    seen: list[str] = []
    for match in CLAIM_REFERENCE_PATTERN.finditer(text):
        ref = match.group(0).upper()
        if ref not in seen:
            seen.append(ref)
    return seen
