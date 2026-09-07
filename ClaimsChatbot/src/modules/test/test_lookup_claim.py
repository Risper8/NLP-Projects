
from __future__ import annotations
from typing import Any
import pytest
from src.modules.tools.lookup_claim import (
    CLAIMS_EXAMPLES,
    EXAMPLE_QUESTIONS,
    TRACKED_FIELD_HINTS_PATTERN,
    detect_not_tracked_concept,
    get_lookup_claim_tool,
)
from src.modules.test.conftest import (
    KNOWN_DOMAIN,
    KNOWN_CLAIM_REF,
    OTHER_TENANT_CLAIM_REF,
    SECOND_KNOWN_CLAIM_REF,
)


def test_example_bank_stays_within_the_dynamic_selection_budget():
    assert len(CLAIMS_EXAMPLES) <= 13


def test_every_example_question_is_extracted_cleanly_not_truncated():
    # a regex miss falls back to the raw "USER INPUT: ... QUERY: ..." text
    assert len(EXAMPLE_QUESTIONS) == len(CLAIMS_EXAMPLES)
    for question in EXAMPLE_QUESTIONS:
        assert "USER INPUT" not in question
        assert "QUERY:" not in question


@pytest.mark.parametrize(
    "query",
    [
        f"Who is the adjuster assigned to claim {KNOWN_CLAIM_REF}?",
        f"What is the settlement date for claim {KNOWN_CLAIM_REF}?",
        f"When was claim {KNOWN_CLAIM_REF} settled?",
        f"What type of business is claim {KNOWN_CLAIM_REF}?",
        f"Was there an offset reason on claim {KNOWN_CLAIM_REF}?",
        f"What is the broker's own reference number for claim {KNOWN_CLAIM_REF}?",
        f"What country is the cedant in for claim {KNOWN_CLAIM_REF}?",
        f"What is the cedant's contact email for claim {KNOWN_CLAIM_REF}?",
    ],
)
def test_not_tracked_concepts_are_detected(query):
    assert detect_not_tracked_concept(query) is not None


@pytest.mark.parametrize(
    "query",
    [
        f"What is the date of loss for claim {KNOWN_CLAIM_REF}?",  # tracked -- must not collide with settlement date
        f"What is the class of business for claim {KNOWN_CLAIM_REF}?",  # tracked -- must not collide with type of business
        f"What is the reference number for claim {KNOWN_CLAIM_REF}?",  # our own ref_num, tracked -- not a broker/cedant's own reference
        f"What is the settlement amount for claim {KNOWN_CLAIM_REF}?",  # tracked (Payment.amount) -- must not collide on bare "settle*"
        f"What is the status of claim {KNOWN_CLAIM_REF}?",
    ],
)
def test_tracked_lookalike_concepts_are_not_flagged(query):
    assert detect_not_tracked_concept(query) is None


async def test_call_short_circuits_on_a_not_tracked_concept_without_calling_the_model():
    tool = get_lookup_claim_tool()
    result = await tool(
        f"Who is the adjuster for claim {KNOWN_CLAIM_REF}?",
        domain=KNOWN_DOMAIN, memory_client=_FakeMemoryClient(set()),
    )
    assert result["not_tracked"] is True
    assert result["cypher"] == "<deterministic not-tracked short-circuit>"


def test_a_pure_not_tracked_question_has_no_other_answerable_field_hint():
    # this is what makes the full short-circuit above safe: nothing else
    # in the query would be worth running Text2Cypher for
    query = f"Who is the adjuster for claim {KNOWN_CLAIM_REF}?"
    assert detect_not_tracked_concept(query) is not None
    assert TRACKED_FIELD_HINTS_PATTERN.search(query) is None


def test_a_compound_question_mixing_a_real_field_with_a_not_tracked_one_is_recognized_as_such():
    # e.g. "reserve, class and type of business" -- short-circuiting the
    # whole call here would silently drop the real, answerable fields too
    query = f"what are the reserve, class and type of business for claim {KNOWN_CLAIM_REF}?"
    assert detect_not_tracked_concept(query) is not None
    assert TRACKED_FIELD_HINTS_PATTERN.search(query) is not None


class _FakeCypherAccessor:

    def __init__(self, authorized_refs: set[str]):
        self._authorized = authorized_refs

    async def cypher(self, query: str, params: dict[str, Any]):
        requested = set(params["refs"])
        return [{"authorized": ref} for ref in requested if ref in self._authorized]


class _FakeMemoryClient:
    def __init__(self, authorized_refs: set[str]):
        self.query = _FakeCypherAccessor(authorized_refs)


async def test_admin_domain_skips_verification_entirely():
    tool = get_lookup_claim_tool()
    rows = [{"c.ref_num": "ANYTHING0000001"}]
    verified, dropped, confirmed = await tool._verify_domain_scope(
        rows, domain=None, memory_client=_FakeMemoryClient(set())
    )
    assert verified == rows
    assert dropped == 0
    assert confirmed is True  # admin is trusted outright, not "unverified"


async def test_empty_rows_returns_empty():
    tool = get_lookup_claim_tool()
    verified, dropped, confirmed = await tool._verify_domain_scope(
        [], domain=KNOWN_DOMAIN, memory_client=_FakeMemoryClient(set())
    )
    assert verified == []
    assert dropped == 0


async def test_authorized_rows_pass_through():
    tool = get_lookup_claim_tool()
    rows = [{"c.ref_num": KNOWN_CLAIM_REF, "c.status": "Completed"}]
    verified, dropped, confirmed = await tool._verify_domain_scope(
        rows, domain=KNOWN_DOMAIN, memory_client=_FakeMemoryClient({KNOWN_CLAIM_REF})
    )
    assert verified == rows
    assert dropped == 0
    assert confirmed is True


async def test_unauthorized_rows_are_dropped_not_returned():
    tool = get_lookup_claim_tool()
    rows = [
        {"c.ref_num": KNOWN_CLAIM_REF, "c.status": "Completed"},
        {"claim1": OTHER_TENANT_CLAIM_REF, "status1": "Completed"},
    ]
    verified, dropped, confirmed = await tool._verify_domain_scope(
        rows, domain=KNOWN_DOMAIN, memory_client=_FakeMemoryClient({KNOWN_CLAIM_REF})
    )
    assert verified == [rows[0]]
    assert dropped == 1
    assert confirmed is True


async def test_mixed_authorized_and_unauthorized_in_one_comparison_row():
    tool = get_lookup_claim_tool()
    rows = [{"claim1": KNOWN_CLAIM_REF, "claim2": OTHER_TENANT_CLAIM_REF}]
    verified, dropped, confirmed = await tool._verify_domain_scope(
        rows, domain=KNOWN_DOMAIN, memory_client=_FakeMemoryClient({KNOWN_CLAIM_REF})
    )
    assert verified == []
    assert dropped == 1
    assert confirmed is False


async def test_a_ref_less_row_mixed_with_verifiable_ones_is_dropped():
    tool = get_lookup_claim_tool()
    rows = [
        {"c.ref_num": KNOWN_CLAIM_REF, "c.status": "Completed"},
        {"broker_name": "Some Broker"},
    ]
    verified, dropped, confirmed = await tool._verify_domain_scope(
        rows, domain=KNOWN_DOMAIN, memory_client=_FakeMemoryClient({KNOWN_CLAIM_REF})
    )
    assert verified == [rows[0]]
    assert dropped == 1
    assert confirmed is True


async def test_a_result_with_no_identifiable_claims_anywhere_is_returned_unfiltered():
    tool = get_lookup_claim_tool()
    rows = [{"broker_name": "Some Broker"}]
    verified, dropped, confirmed = await tool._verify_domain_scope(
        rows, domain=KNOWN_DOMAIN, memory_client=_FakeMemoryClient(set())
    )
    assert verified == rows
    assert dropped == 0
    assert confirmed is False


class _FakeMultiClaimCypherAccessor:
    def __init__(self, records_by_ref_and_domain: dict[tuple[str, str | None], dict]):
        self._records = records_by_ref_and_domain

    async def cypher(self, query: str, params: dict[str, Any]):
        return [
            self._records[(ref, params["domain"])]
            for ref in params["refs"]
            if (ref, params["domain"]) in self._records
        ]


class _FakeMultiClaimMemoryClient:
    def __init__(self, records_by_ref_and_domain: dict[tuple[str, str | None], dict]):
        self.query = _FakeMultiClaimCypherAccessor(records_by_ref_and_domain)


async def test_multi_ref_lookup_splits_found_and_not_found():
    tool = get_lookup_claim_tool()
    fake_client = _FakeMultiClaimMemoryClient({
        (KNOWN_CLAIM_REF, KNOWN_DOMAIN): {"ref_num": KNOWN_CLAIM_REF, "status": "Completed"},
    })
    result = await tool._lookup_multiple_refs(
        [KNOWN_CLAIM_REF, SECOND_KNOWN_CLAIM_REF],
        domain=KNOWN_DOMAIN, memory_client=fake_client,
    )
    assert result["rows"] == [{"ref_num": KNOWN_CLAIM_REF, "status": "Completed"}]
    assert result["not_found"] == [SECOND_KNOWN_CLAIM_REF]


async def test_multi_ref_lookup_is_domain_scoped():
    tool = get_lookup_claim_tool()
    fake_client = _FakeMultiClaimMemoryClient({
        (OTHER_TENANT_CLAIM_REF, "someone-elses-domain.io"): {"ref_num": OTHER_TENANT_CLAIM_REF},
    })
    result = await tool._lookup_multiple_refs(
        [OTHER_TENANT_CLAIM_REF], domain=KNOWN_DOMAIN, memory_client=fake_client,
    )
    assert result["rows"] == []
    assert result["not_found"] == [OTHER_TENANT_CLAIM_REF]


async def test_call_routes_two_or_more_named_refs_to_the_deterministic_path():
    tool = get_lookup_claim_tool()
    fake_client = _FakeMultiClaimMemoryClient({
        (KNOWN_CLAIM_REF, KNOWN_DOMAIN): {"ref_num": KNOWN_CLAIM_REF, "status": "Completed"},
        (SECOND_KNOWN_CLAIM_REF, KNOWN_DOMAIN): {"ref_num": SECOND_KNOWN_CLAIM_REF, "status": "Completed"},
    })
    result = await tool(
        f"reserve amounts for {KNOWN_CLAIM_REF} and {SECOND_KNOWN_CLAIM_REF}",
        domain=KNOWN_DOMAIN, memory_client=fake_client,
    )
    assert result["cypher"] == "<direct multi-ref lookup>"
    assert len(result["rows"]) == 2
    assert result["not_found"] == []



pytestmark_live = [pytest.mark.integration, pytest.mark.live_llm]


@pytest.mark.integration
@pytest.mark.live_llm
async def test_real_lookup_self_scopes_to_the_requesting_domain(memory):
    tool = get_lookup_claim_tool()
    result = await tool(
        f"what is the status of claim {KNOWN_CLAIM_REF}",
        domain=KNOWN_DOMAIN,
        memory_client=memory.client,
    )
    assert "error" not in result
    assert "client_domains" in result["cypher"] or result["rows"]


@pytest.mark.integration
@pytest.mark.live_llm
async def test_real_lookup_never_returns_another_tenants_claim(memory):
    tool = get_lookup_claim_tool()
    result = await tool(
        f"what is the status of claim {OTHER_TENANT_CLAIM_REF}",
        domain=KNOWN_DOMAIN,
        memory_client=memory.client,
    )
    rows = result.get("rows", [])
    for row in rows:
        assert OTHER_TENANT_CLAIM_REF not in str(row.values())


@pytest.mark.integration
@pytest.mark.live_llm
async def test_real_comparison_query_returns_both_claims(memory):
    tool = get_lookup_claim_tool()
    result = await tool(
        f"compare {KNOWN_CLAIM_REF} and {SECOND_KNOWN_CLAIM_REF}",
        domain=KNOWN_DOMAIN,
        memory_client=memory.client,
    )
    assert "error" not in result
    assert result["rows"]
