from __future__ import annotations

GLOSSARY_TEXT = """
TERMINOLOGY GLOSSARY -- alphabetical list of words/phrases about claims
that are hard to map onto this knowledge graph, with what they actually
mean here:

ADJUSTER -- NOT TRACKED. No node, relationship, or property in this
graph records who is assigned as adjuster on a claim. Say plainly it
isn't tracked; don't invent a name or borrow an unrelated node/property
(e.g. a Query's id or reason) to answer this.

BASE AMOUNT / CLAIM AMOUNT -- claim.base_amount. Carries no currency of
its own -- see CURRENCY below.

BROKER / INTERMEDIARY -- the Broker node (broker_name / broker_code),
linked via SUBMITTED. The party who placed the business, when there is
one -- some claims are ceded directly, with no broker at all.

BROKER'S OR CEDANT'S OWN REFERENCE NUMBER -- NOT TRACKED. Separate from
this system's own ref_num; not recorded here.

CAUSE OF LOSS -- claim.cause_of_loss.

CEDANT -- see INSURER below; same node, same mapping.

CEDANT'S COUNTRY -- NOT TRACKED.

CLAIMS OFFICER / TEAM LEAD / APPROVER (or similar named role) -- NOT
TRACKED, beyond whatever a Query or ProcessInstance relationship in the
schema actually captures. Most role fields like this are placeholders
with no real data behind them in this system.

CLASS OF BUSINESS -- claim.class_of_business. Do NOT confuse this with
TYPE OF BUSINESS below -- they are different fields, and only one of
them exists in this data.

CONTACT EMAIL (broker's or cedant's) -- NOT TRACKED.

CURRENCY -- not a Claim-level field. The only place a currency is
recorded is a linked Payment's original_currency (via HAS_PAYMENT), and
only when a Payment node exists at all. Never assume a currency; if no
Payment node or original_currency is returned, say the currency isn't
specified.

DATE OF LOSS -- claim.date_of_loss: when the loss occurred. Do NOT
confuse this with SETTLEMENT DATE below -- they are different concepts,
and only date_of_loss is ever recorded here.

INSURED / POLICYHOLDER / THE CLIENT -- the Insured node (insured_name).
Whose risk the policy actually covers -- a different party from the
cedant.

INSURER / INSURANCE COMPANY -- the Cedant node (cedant_name /
cedant_code), linked via HAS_CLAIM. This is the original insurer who
wrote the underlying policy and is ceding part of the risk to the
reinsurer -- it is NOT a separate "Insurer" node; there isn't one.

OFFSET REASON -- NOT TRACKED. Why a payment was offset or reduced;
occasionally present in the source system but never loaded into this
graph.

PORTAL STATUS -- claim.portal_status (e.g. "Terminated") -- "where does
it stand in the portal." A DIFFERENT field from STATUS below, not a
rephrasing of it; report each as its own value, don't treat one as
explaining the other.

QUERY / OUTSTANDING REQUIREMENTS -- the linked Query node (via
HAS_QUERY): a request or question raised on the claim, with its own
status and reason. Every Query in this data has status "Completed" --
there is no "open," "pending," or "outstanding" query state recorded
here. Don't describe a query as outstanding or unresolved; that
distinction doesn't exist in this data.

REASON (claim-level) / "why was this flagged" -- claim.claim_reason,
which is frequently empty or a placeholder value. Don't invent a reason
if this is empty.

REQUEST TYPE -- an internal system code (e.g. "PLA"). Its exact meaning
is not confirmed anywhere in this data -- report the raw code rather
than guessing or inventing an expansion for it.

RESERVE -- claim.reserves. Carries no currency of its own -- see
CURRENCY above.

SETTLEMENT / PAYMENT / "was it paid" -- the linked Payment node (via
HAS_PAYMENT), not a claim-level field. A claim with no payment status,
finance reference, or paid amount recorded has no Payment node at all
-- there's nothing to report, not a null value on the claim itself.

SETTLEMENT DATE -- NOT TRACKED. No field in this graph records when a
claim was settled. Do NOT substitute DATE OF LOSS for it -- that is a
different concept (when the loss occurred, not when it was settled).
Say plainly this isn't tracked.

STATUS -- claim.status (e.g. "Completed"). See PORTAL STATUS above for
the related-but-different field.

TYPE OF BUSINESS -- NOT TRACKED. A real, frequently-populated field in
the source system, but it was never loaded into this graph. Only
CLASS OF BUSINESS above exists here -- these are NOT interchangeable;
if asked for type of business, say it isn't tracked, don't answer with
class of business instead.
"""
