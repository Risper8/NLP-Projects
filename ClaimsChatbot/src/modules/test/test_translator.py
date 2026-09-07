from __future__ import annotations
import pytest
from src.modules.orchestrator.translator import extract_claim_reference


@pytest.mark.parametrize(
    "text,expected",
    [
        ("MNCL0502200007", "MNCL0502200007"),
        ("what about claim PRCL17012020000004", "PRCL17012020000004"),
        ("mncl0502200007", "MNCL0502200007"),  # lowercase input, uppercased output
        ("CLM-12345", "CLM-12345"),  # hyphenated form
        (
            "hey so my colleague mentioned PRCL17012020000004 yesterday",
            "PRCL17012020000004",
        ),
    ],
)
def test_extract_claim_reference_finds_the_reference(text, expected):
    assert extract_claim_reference(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "hi there",
        "what is the status?",
        "12345", 
        "call me at 0722123456",  
        "",
    ],
)
def test_extract_claim_reference_returns_none_when_absent(text):
    assert extract_claim_reference(text) is None
