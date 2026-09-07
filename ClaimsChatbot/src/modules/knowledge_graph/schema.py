from __future__ import annotations
import os
from dotenv import load_dotenv
from neo4j import GraphDatabase
from src.modules.utils.embeddings import SentenceTransformerEmbeddingProvider
from src.modules.knowledge_graph.ontology import NOTE, VECTOR_INDEX_NAME

load_dotenv()

CONSTRAINTS = [
    "CREATE CONSTRAINT claim_ref_num IF NOT EXISTS "
    "FOR (c:Claim) REQUIRE c.ref_num IS UNIQUE",

    "CREATE CONSTRAINT broker_code IF NOT EXISTS "
    "FOR (b:Broker) REQUIRE b.broker_code IS UNIQUE",

    "CREATE CONSTRAINT cedant_code IF NOT EXISTS "
    "FOR (c:Cedant) REQUIRE c.cedant_code IS UNIQUE",

    "CREATE CONSTRAINT user_id IF NOT EXISTS "
    "FOR (u:User) REQUIRE u.user_id IS UNIQUE",

    "CREATE CONSTRAINT process_instance_id IF NOT EXISTS "
    "FOR (p:ProcessInstance) REQUIRE p.process_instance_id IS UNIQUE",

    "CREATE CONSTRAINT payment_id IF NOT EXISTS "
    "FOR (p:Payment) REQUIRE p.payment_id IS UNIQUE",

    "CREATE CONSTRAINT query_id IF NOT EXISTS "
    "FOR (q:Query) REQUIRE q.query_id IS UNIQUE",

    "CREATE CONSTRAINT note_id IF NOT EXISTS "
    "FOR (n:Note) REQUIRE n.note_id IS UNIQUE",

    "CREATE CONSTRAINT location_name IF NOT EXISTS "
    "FOR (l:Location) REQUIRE l.name IS UNIQUE",
]


def create_constraints(driver, database: str) -> None:
    with driver.session(database=database) as session:
        for statement in CONSTRAINTS:
            session.run(statement)


def create_vector_index(driver, database: str, dimensions: int) -> None:
    with driver.session(database=database) as session:
        session.run(
            f"""
            CREATE VECTOR INDEX {VECTOR_INDEX_NAME} IF NOT EXISTS
            FOR (n:{NOTE}) ON (n.embedding)
            OPTIONS {{
                indexConfig: {{
                    `vector.dimensions`: $dimensions,
                    `vector.similarity_function`: 'cosine'
                }}
            }}
            """,
            dimensions=dimensions,
        )


def main() -> None:
    driver = GraphDatabase.driver(
        os.environ["NEO4J_URI"],
        auth=(
            os.environ["NEO4J_USERNAME"],
            os.environ["NEO4J_PASSWORD"],
        ),
    )
    database = os.environ["NEO4J_DATABASE"]

    driver.verify_connectivity()

    create_constraints(driver, database)
    print(f"Created/verified {len(CONSTRAINTS)} constraints.")

    embedding_provider = SentenceTransformerEmbeddingProvider()
    create_vector_index(driver, database, embedding_provider.dimensions)
    print(
        f"Created/verified vector index '{VECTOR_INDEX_NAME}' "
        f"({embedding_provider.dimensions} dimensions, "
        f"model={embedding_provider.model_name})."
    )

    driver.close()


if __name__ == "__main__":
    main()
