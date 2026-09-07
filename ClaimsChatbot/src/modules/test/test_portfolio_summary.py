
from __future__ import annotations

import pytest

from src.modules.tools import portfolio_summary
from src.modules.test.conftest import KNOWN_DOMAIN, KNOWN_TOTAL_CLAIMS_FOR_DOMAIN


@pytest.mark.integration
async def test_list_recent_claims_are_scoped_to_domain(memory):
    claims = await portfolio_summary.list_recent_claims(memory.client, domain=KNOWN_DOMAIN, limit=5)
    assert len(claims) == 5
    assert all("ref_num" in c for c in claims)


@pytest.mark.integration
async def test_portfolio_summary_matches_known_total(memory):
    summary = await portfolio_summary.get_portfolio_summary(memory.client, domain=KNOWN_DOMAIN)
    assert summary["total_claims"] == KNOWN_TOTAL_CLAIMS_FOR_DOMAIN
    assert sum(item["count"] for item in summary["breakdown"]) == KNOWN_TOTAL_CLAIMS_FOR_DOMAIN


@pytest.mark.integration
async def test_portfolio_summary_for_unknown_domain_is_empty_not_an_error(memory):
    summary = await portfolio_summary.get_portfolio_summary(memory.client, domain="no-such-tenant.example")
    assert summary["total_claims"] == 0
    assert summary["breakdown"] == []
