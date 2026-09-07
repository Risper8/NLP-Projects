from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Any
import pandas as pd
from dotenv import load_dotenv
from neo4j import Driver, GraphDatabase
from src.modules.utils.embeddings import SentenceTransformerEmbeddingProvider

load_dotenv()
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_PATH = str(REPO_ROOT / "data" / "CLAIMS_CLAIMS.json")



# Data loading
def load_records(data_path: str) -> pd.DataFrame:
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    df = pd.json_normalize(data)
    return df.astype(object).where(pd.notna(df), None)



# Client scoping
def _client_domains_for_row(row) -> tuple[str | None, str | None, list[str]]:
    domain_id = row.get("BROKERCEDANT_DOMAINID")

    if domain_id:
        domain_id = domain_id.strip().lower()

    broker_present = bool(row.get("BROKER_NAME") or row.get("BROKER_CODE"))

    broker_domain = domain_id if (domain_id and broker_present) else None
    cedant_domain = domain_id if (domain_id and not broker_present) else None

    client_domains = sorted(
        {d for d in [broker_domain, cedant_domain] if d}
    )

    return broker_domain, cedant_domain, client_domains


def create_claim(tx, record):

    query = """
    MERGE (claim:Claim {
        ref_num: $ref_num
    })

    SET claim.claim_id = $claim_id,
        claim.status = $status,
        claim.process_name = $process_name,
        claim.created_date = $created_date,
        claim.modified_date = $modified_date,
        claim.date_of_loss = $date_of_loss,
        claim.request_type = $request_type,
        claim.cause_of_loss = $cause_of_loss,
        claim.class_of_business = $class_of_business,
        claim.base_amount = $base_amount,
        claim.reserves = $reserves,
        claim.paid_claim = $paid_claim,
        claim.payment_status = $payment_status,
        claim.portal_status = $portal_status,
        claim.proceed_decision = $proceed_decision,
        claim.claim_reason = $claim_reason,
        claim.client_domains = $client_domains

    // broker relationship

    // Prefer broker_code when available
    FOREACH (_ IN CASE
        WHEN $broker_code IS NOT NULL
        THEN [1]
        ELSE []
    END |

        MERGE (broker:Broker {
            broker_code: $broker_code
        })

        SET broker.name = $broker_name,
            broker.domain = coalesce($broker_domain, broker.domain)

        MERGE (broker)-[:SUBMITTED]->(claim)
    )

    // If broker_code is missing, use broker_name
    FOREACH (_ IN CASE
        WHEN $broker_code IS NULL
             AND $broker_name IS NOT NULL
        THEN [1]
        ELSE []
    END |

        MERGE (broker:Broker {
            name: $broker_name
        })

        SET broker.domain = coalesce($broker_domain, broker.domain)

        MERGE (broker)-[:SUBMITTED]->(claim)
    )


    // cedant relationship

    // Prefer cedant_code when available
    FOREACH (_ IN CASE
        WHEN $cedant_code IS NOT NULL
        THEN [1]
        ELSE []
    END |

        MERGE (cedant:Cedant {
            cedant_code: $cedant_code
        })

        SET cedant.name = $cedant_name,
            cedant.domain = coalesce($cedant_domain, cedant.domain)

        MERGE (cedant)-[:HAS_CLAIM]->(claim)
    )

    // If cedant_code is missing, use cedant_name
    FOREACH (_ IN CASE
        WHEN $cedant_code IS NULL
             AND $cedant_name IS NOT NULL
        THEN [1]
        ELSE []
    END |

        MERGE (cedant:Cedant {
            name: $cedant_name
        })

        SET cedant.domain = coalesce($cedant_domain, cedant.domain)

        MERGE (cedant)-[:HAS_CLAIM]->(claim)
    )


    // Insured

    FOREACH (_ IN CASE
        WHEN $insured_name IS NOT NULL
        THEN [1]
        ELSE []
    END |

        MERGE (insured:Insured {
            name: $insured_name
        })

        MERGE (insured)-[:HAS_CLAIM]->(claim)
    )

    RETURN claim
    """

    tx.run(query, **record)


def create_process_instance(tx, record):

    query = """
    MATCH (claim:Claim {
        ref_num: $ref_num
    })

    FOREACH (_ IN CASE
        WHEN $process_instance_id IS NOT NULL
        THEN [1]
        ELSE []
    END |

        MERGE (process:ProcessInstance {
            process_instance_id: $process_instance_id
        })

        SET process.process_name = $process_name,
            process.case_id = $case_id,
            process.current_step = $current_step,
            process.previous_step = $previous_step,
            process.portal_status = $portal_status,
            process.proceed_decision = $proceed_decision,
            process.client_domains = $client_domains

        MERGE (claim)-[:HAS_PROCESS]->(process)
    )

    RETURN claim
    """

    tx.run(query, **record)


def create_users(tx, record):

    query = """
    MATCH (claim:Claim {
        ref_num: $ref_num
    })


    // initiator

    FOREACH (_ IN CASE
        WHEN $initiator_userid IS NOT NULL
        THEN [1]
        ELSE []
    END |

        MERGE (u:User {
            user_id: $initiator_userid
        })

        SET u.full_name = $initiator_fullname

        MERGE (u)-[:INITIATED]->(claim)
    )

    //creation


    FOREACH (_ IN CASE
        WHEN $created_by IS NOT NULL
        THEN [1]
        ELSE []
    END |

        MERGE (u:User {
            user_id: $created_by
        })

        MERGE (u)-[:CREATED]->(claim)
    )



    //modification


    FOREACH (_ IN CASE
        WHEN $modified_by IS NOT NULL
        THEN [1]
        ELSE []
    END |

        MERGE (u:User {
            user_id: $modified_by
        })

        MERGE (u)-[:MODIFIED]->(claim)
    )


    // approver


    FOREACH (_ IN CASE
        WHEN $approver_userid IS NOT NULL
        THEN [1]
        ELSE []
    END |

        MERGE (u:User {
            user_id: $approver_userid
        })

        SET u.full_name = $approver_fullname

        MERGE (u)-[:APPROVED]->(claim)
    )


    RETURN claim
    """

    tx.run(query, **record)


def create_payment(tx, record):

    query = """
    MATCH (claim:Claim {
        ref_num: $ref_num
    })

    FOREACH (_ IN CASE
        WHEN $paid_claim IS NOT NULL
             OR $finance_ref_num IS NOT NULL
             OR $payment_status IS NOT NULL
        THEN [1]
        ELSE []
    END |

        MERGE (payment:Payment {
            payment_id: $payment_id
        })

        SET payment.amount = $paid_claim,
            payment.currency = $paid_claim_currency,
            payment.status = $payment_status,
            payment.original_currency = $original_currency,
            payment.type = $type_of_payment,
            payment.finance_ref_num = $finance_ref_num,
            payment.pay_to = $pay_to,
            payment.client_domains = $client_domains

        MERGE (claim)-[:HAS_PAYMENT]->(payment)
    )

    RETURN claim
    """

    tx.run(query, **record)


def create_query(tx, record):

    query = """
    MATCH (claim:Claim {
        ref_num: $ref_num
    })

    FOREACH (_ IN CASE
        WHEN $reason IS NOT NULL
             OR $query_comments IS NOT NULL
        THEN [1]
        ELSE []
    END |

        MERGE (query:Query {
            query_id: $query_id
        })

        SET query.reason = $reason,
            query.status = $query_status,
            query.created_date = $query_created_date,
            query.comments = $query_comments,
            query.client_domains = $client_domains

        MERGE (claim)-[:HAS_QUERY]->(query)
    )

    RETURN claim
    """

    tx.run(query, **record)


def create_location(tx, record):
    query = """
    MATCH (claim:Claim {
        ref_num: $ref_num
    })

    FOREACH (_ IN CASE
        WHEN $location_name IS NOT NULL
        THEN [1]
        ELSE []
    END |

        MERGE (location:Location {
            name: $location_name
        })

        MERGE (claim)-[:OCCURRED_IN]->(location)
    )

    RETURN claim
    """

    tx.run(query, **record)


def create_note(tx, record):
    query = """
    MATCH (claim:Claim {
        ref_num: $ref_num
    })

    MERGE (note:Note {
        note_id: $note_id
    })

    SET note.text = $text,
        note.source_field = $source_field,
        note.client_domains = $client_domains,
        note.embedding = $embedding

    MERGE (claim)-[:HAS_NOTE]->(note)

    RETURN note
    """

    tx.run(query, **record)



# Per-row orchestration
def _claim_record(row, client_domains, broker_domain, cedant_domain) -> dict[str, Any]:
    return {
        "ref_num": row.get("REF_NUM"),
        "claim_id": row.get("CLAIM_ID"),
        "status": row.get("STATUS"),
        "process_name": row.get("PROCESS_NAME"),
        "created_date": row.get("CREATED_DATE"),
        "modified_date": row.get("MODIFIED_DATE"),
        "date_of_loss": row.get("DATE_OF_LOSS"),
        "request_type": row.get("REQUESTTYPE"),
        "cause_of_loss": row.get("CAUSE_OF_LOSS"),
        "class_of_business": row.get("CLASS_OF_BUSINESS"),
        "base_amount": row.get("BASE_AMOUNT"),
        "reserves": row.get("RESERVES"),
        "paid_claim": row.get("PAID_CLAIM"),
        "payment_status": row.get("PAYMENT_STATUS"),
        "portal_status": row.get("PORTAL_STATUS"),
        "proceed_decision": row.get("PROCEED_DECISION"),
        "claim_reason": row.get("CLAIM_REASON"),
        "client_domains": client_domains,

        # Broker info
        "broker_code": row.get("BROKER_CODE"),
        "broker_name": row.get("BROKER_NAME"),
        "broker_domain": broker_domain,

        # Cedant info
        "cedant_code": row.get("CEDANT_CODE"),
        "cedant_name": row.get("CEDANT_NAME"),
        "cedant_domain": cedant_domain,

        # Insured info
        "insured_name": row.get("INSURED_NAME"),
    }


def _process_record(row, client_domains) -> dict[str, Any]:
    return {
        "ref_num": row.get("REF_NUM"),
        "process_instance_id": row.get("PROCESS_INSTANCE_ID"),
        "process_name": row.get("PROCESS_NAME"),
        "case_id": row.get("CASE_ID"),
        "current_step": row.get("CURRENT_STEP"),
        "previous_step": row.get("PREVIOUS_STEP"),
        "portal_status": row.get("PORTAL_STATUS"),
        "proceed_decision": row.get("PROCEED_DECISION"),
        "client_domains": client_domains,
    }


def _user_record(row) -> dict[str, Any]:
    return {
        "ref_num": row.get("REF_NUM"),
        "initiator_userid": row.get("INITIATOR_USERID"),
        "initiator_fullname": row.get("INITIATOR_FULLNAME"),
        "created_by": row.get("CREATED_BY"),
        "modified_by": row.get("MODIFIED_BY"),
        "approver_userid": row.get("APPROVER_USERID"),
        "approver_fullname": row.get("APPROVER_FULLNAME"),
    }


def _payment_record(row, client_domains) -> dict[str, Any]:
    return {
        "ref_num": row.get("REF_NUM"),
        # Using claim reference as temporary payment ID
        "payment_id": f"{row.get('REF_NUM')}-PAYMENT",
        "finance_ref_num": row.get("FINANCE_REF_NUM"),
        "paid_claim": row.get("PAID_CLAIM"),
        "paid_claim_currency": row.get("PAID_CLAIM_CURRENCY"),
        "payment_status": row.get("PAYMENT_STATUS"),
        "original_currency": row.get("ORIGINAL_CURRENCY"),
        "type_of_payment": row.get("TYPE_OF_PAYMENT"),
        "pay_to": row.get("PAY_TO"),
        "client_domains": client_domains,
    }


def _query_record(row, client_domains) -> dict[str, Any]:
    return {
        "ref_num": row.get("REF_NUM"),
        "query_id": f"{row.get('REF_NUM')}-QUERY",
        "reason": row.get("CLAIM_REASON"),
        "query_status": row.get("STATUS"),
        "query_created_date": row.get("CREATED_DATE"),
        "query_comments": row.get("UW_COMMENTS"),
        "client_domains": client_domains,
    }


def _location_record(row) -> dict[str, Any]:
    return {
        "ref_num": row.get("REF_NUM"),
        "location_name": row.get("LOCATION_OF_LOSS"),
    }


def _note_records(row, client_domains, embedding_provider) -> list[dict[str, Any]]:
    """One Note per non-empty free-text field on the row."""

    ref_num = row.get("REF_NUM")
    text_fields = {
        "CLAIM_REASON": row.get("CLAIM_REASON"),
        "UW_COMMENTS": row.get("UW_COMMENTS"),
    }

    records = []

    for source_field, text in text_fields.items():

        if not text or not text.strip():
            continue

        records.append({
            "ref_num": ref_num,
            "note_id": f"{ref_num}-{source_field}",
            "text": text,
            "source_field": source_field,
            "client_domains": client_domains,
            "embedding": embedding_provider.embed_sync(text),
        })

    return records


def ingest_row(session, row, embedding_provider) -> None:

    broker_domain, cedant_domain, client_domains = _client_domains_for_row(row)

    session.execute_write(
        create_claim,
        _claim_record(row, client_domains, broker_domain, cedant_domain),
    )

    session.execute_write(
        create_process_instance,
        _process_record(row, client_domains),
    )

    session.execute_write(create_users, _user_record(row))

    session.execute_write(
        create_payment,
        _payment_record(row, client_domains),
    )

    session.execute_write(
        create_query,
        _query_record(row, client_domains),
    )

    session.execute_write(create_location, _location_record(row))

    for note_record in _note_records(row, client_domains, embedding_provider):
        session.execute_write(create_note, note_record)


def run_ingestion(
    data_path: str = DEFAULT_DATA_PATH,
    driver: Driver | None = None,
    database: str | None = None,
    embedding_provider=None,
) -> None:

    owns_driver = driver is None

    if owns_driver:
        driver = GraphDatabase.driver(
            os.environ["NEO4J_URI"],
            auth=(
                os.environ["NEO4J_USERNAME"],
                os.environ["NEO4J_PASSWORD"],
            ),
        )
        driver.verify_connectivity()

    database = database or os.environ["NEO4J_DATABASE"]
    embedding_provider = embedding_provider or SentenceTransformerEmbeddingProvider()

    data_df = load_records(data_path)

    try:
        with driver.session(database=database) as session:
            for index, row in data_df.iterrows():
                print(f"Processing claim {index + 1}/{len(data_df)}")
                ingest_row(session, row, embedding_provider)
    finally:
        if owns_driver:
            driver.close()


if __name__ == "__main__":
    run_ingestion()
