from __future__ import annotations
from typing import Any
from pydantic import BaseModel, Field
from src.modules.utils.embeddings import SentenceTransformerEmbeddingProvider
from src.modules.knowledge_graph.ontology import GRAPHRAG_ALLOWED_RELATIONSHIPS, VECTOR_INDEX_NAME


class GraphSeed(BaseModel):
    node_id: str | None = None
    node_type: str | None = None
    name: str | None = None
    score: float | None = None
    properties: dict[str, Any] = Field(default_factory=dict)


class GraphRelationship(BaseModel):
    source: str
    relationship: str
    target: str


class GraphRAGResult(BaseModel):
    query: str
    seeds: list[GraphSeed] = Field(
        default_factory=list
    )
    entities: list[dict[str, Any]] = Field(
        default_factory=list
    )
    relationships: list[GraphRelationship] = Field(
        default_factory=list
    )

    facts: list[dict[str, Any]] = Field(
        default_factory=list
    )

    context: str = ""

class GraphRAG:

    def __init__(
        self,
        memory,
        embedding_provider=None,
    ):
        self.memory = memory
        self.embedding_provider = embedding_provider or SentenceTransformerEmbeddingProvider()

    async def retrieve(
        self,
        query: str,
        domain: str | None,
        limit: int = 10,
        depth: int = 2,
    ) -> GraphRAGResult:
        query_embedding = await self.embed_query(query)
        seeds = await self.vector_search(
            query_embedding=query_embedding,
            domain=domain,
            limit=limit,
        )

        entities = await self.resolve_entities(
            query=query,
            seeds=seeds,
            domain=domain,
        )

        graph = await self.expand_subgraph(
            entities=entities,
            domain=domain,
            depth=depth,
        )

        context = self.build_context(
            query=query,
            seeds=seeds,
            graph=graph,
        )

        return GraphRAGResult(
            query=query,
            seeds=seeds,
            entities=graph.get("entities", []),
            relationships=graph.get(
                "relationships",
                [],
            ),
            facts=graph.get(
                "facts",
                [],
            ),
            context=context,
        )


    async def embed_query(
        self,
        query: str,
    ) -> list[float]:

        return await self.embedding_provider.embed(query)


    async def vector_search(
        self,
        query_embedding: list[float],
        domain: str | None,
        limit: int = 10,
    ) -> list[GraphSeed]:
        records = await self._neo4j_vector_search(
            query_embedding=query_embedding,
            domain=domain,
            limit=limit,
        )

        return [
            GraphSeed(
                node_id=record.get("node_id"),
                node_type=record.get("node_type"),
                name=record.get("name"),
                score=record.get("score"),
                properties=record.get(
                    "properties",
                    {},
                ),
            )
            for record in records
        ]

    async def _neo4j_vector_search(
        self,
        query_embedding: list[float],
        domain: str | None,
        limit: int,
    ):
        query = f"""
        CALL db.index.vector.queryNodes(
            '{VECTOR_INDEX_NAME}',
            $limit,
            $embedding
        )
        YIELD node, score

        WHERE $domain IS NULL OR $domain IN node.client_domains

        RETURN
            elementId(node) AS node_id,
            labels(node)[0] AS node_type,
            coalesce(node.name, node.reference) AS name,
            score,
            properties(node) AS properties

        ORDER BY score DESC
        """

        result = await self.memory.client.query.cypher(
            query,
            {
                "embedding": query_embedding,
                "limit": limit,
                "domain": domain,
            },
        )

        return result


    async def resolve_entities(
        self,
        query: str,
        seeds: list[GraphSeed],
        domain: str | None,
    ) -> list[dict[str, Any]]:
        entities = []

        for seed in seeds:

            if not seed.node_id:
                continue

            entities.append(
                {
                    "node_id": seed.node_id,
                    "node_type": seed.node_type,
                    "name": seed.name,
                }
            )

        return entities

    async def expand_subgraph(
        self,
        entities: list[dict[str, Any]],
        domain: str | None,
        depth: int = 2,
    ) -> dict[str, Any]:
        if not entities:
            return {
                "entities": [],
                "relationships": [],
                "facts": [],
            }

        node_ids = [
            entity["node_id"]
            for entity in entities
            if entity.get("node_id")
        ]

        if not node_ids:
            return {
                "entities": [],
                "relationships": [],
                "facts": [],
            }

  
        depth = int(depth)
        if not 1 <= depth <= 5:
            raise ValueError("depth must be between 1 and 5")
        query = f"""
        MATCH (seed)
        WHERE elementId(seed) IN $node_ids
          AND ($domain IS NULL OR $domain IN seed.client_domains)

        MATCH path =
            (seed)-[*1..{depth}]-(related)

        WHERE ($domain IS NULL OR $domain IN related.client_domains)
          AND ALL(rel IN relationships(path) WHERE type(rel) IN $allowed_relationships)

        RETURN
            [n IN nodes(path) | {{
                id: elementId(n),
                labels: labels(n),
                properties: properties(n)
            }}] AS nodes,
            [r IN relationships(path) | {{
                type: type(r),
                start_id: elementId(startNode(r)),
                end_id: elementId(endNode(r))
            }}] AS relationships
        """

        records = await self.memory.client.query.cypher(
            query,
            {
                "node_ids": node_ids,
                "domain": domain,
                "allowed_relationships": GRAPHRAG_ALLOWED_RELATIONSHIPS,
            },
        )

        return self._normalize_graph_records(
            records
        )


    def _normalize_graph_records(
        self,
        records,
    ) -> dict[str, Any]:
        entities = {}
        relationships = []
        facts = []

        for record in records:

            for node in record.get("nodes", []):

                node_id = node.get("id")

                if node_id is None or node_id in entities:
                    continue

                entities[node_id] = {
                    "id": node_id,
                    "labels": node.get("labels", []),
                    "properties": node.get("properties", {}),
                }

            for relationship in record.get("relationships", []):

                relationships.append(
                    GraphRelationship(
                        source=relationship.get("start_id", ""),
                        relationship=relationship.get(
                            "type",
                            "RELATED_TO",
                        ),
                        target=relationship.get("end_id", ""),
                    )
                )

        return {
            "entities": list(
                entities.values()
            ),
            "relationships": relationships,
            "facts": facts,
        }


    def build_context(
        self,
        query: str,
        seeds: list[GraphSeed],
        graph: dict[str, Any],
    ) -> str:
        lines = []

        lines.append(
            f"User query: {query}"
        )

        lines.append(
            "\nRelevant entities:"
        )

        for entity in graph.get(
            "entities",
            [],
        ):

            properties = entity.get(
                "properties",
                {},
            )

            lines.append(
                f"- {properties}"
            )

        lines.append(
            "\nRelevant relationships:"
        )

        for relationship in graph.get(
            "relationships",
            [],
        ):

            lines.append(
                f"- "
                f"{relationship.source} "
                f"--[{relationship.relationship}]--> "
                f"{relationship.target}"
            )

        if graph.get("facts"):

            lines.append(
                "\nRelevant facts:"
            )

            for fact in graph["facts"]:

                lines.append(
                    f"- {fact}"
                )

        return "\n".join(lines)
