import logging
import os
from dotenv import load_dotenv
from neo4j_agent_memory import MemoryClient
load_dotenv()


class MemoryGraph:
    """
    Provides:
        - Short-term memory
        - Long-term memory
        - Reasoning memory
        - Context retrieval
    """

    def __init__(self, settings, *, embedder=None, extractor=None):
        self.settings = settings
        self.memory = None
        self._embedder = embedder
        self._extractor = extractor

    async def __aenter__(self):
        self.memory = MemoryClient(
            self.settings, embedder=self._embedder, extractor=self._extractor
        )
        await self.memory.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        await self.memory.__aexit__(
            exc_type,
            exc_value,
            traceback,
        )


    # short term memory
    async def store_short_term(
        self,
        session_id: str,
        role: str,
        content: str,
        *,
        extract_entities: bool = True,
    ):

        try:
            return await self.memory.short_term.add_message(
                session_id=session_id,
                role=role,
                content=content,
                extract_entities=extract_entities,
            )
        except Exception as exc:
            if not extract_entities:
                raise
            logging.getLogger(__name__).warning(
                "Long-term extraction failed for a stored message "
                "(message itself should still be persisted): %s",
                exc,
            )
            return None

    async def get_short_term(
        self,
        session_id: str,
        query: str,
        limit: int = 10,
    ):
        """Retrieve short-term conversation memory."""

        conversation = (
            await self.memory.short_term.get_conversation(
                session_id
            )
        )

        summary = (
            await self.memory.short_term.get_conversation_summary(
                session_id
            )
        )

        results = (
            await self.memory.short_term.search_messages(
                query=query,
                session_id=session_id,
                limit=limit,
            )
        )

        return {
            "conversation": conversation,
            "summary": summary,
            "results": results,
        }

    async def clear_short_term(
        self,
        session_id: str,
    ):
        """Clear a short-term session."""

        return await self.memory.short_term.clear_session(
            session_id
        )


    # long-term memory
    async def store_entity(
        self,
        name: str,
        entity_type: str,
        subtype: str,
        description: str,
        properties: dict | None = None,
    ):
        """Store an entity in long-term memory."""

        return await self.memory.long_term.add_entity(
            name=name,
            entity_type=entity_type,
            subtype=subtype,
            description=description,
            properties=properties or {},
        )

    async def store_fact(
        self,
        subject: str,
        predicate: str,
        object: str,
        valid_from: str | None = None,
        valid_until: str | None = None,
    ):
        """Store a temporal fact."""

        return await self.memory.long_term.add_fact(
            subject=subject,
            predicate=predicate,
            object=object,
            valid_from=valid_from,
            valid_until=valid_until,
        )

    async def store_preference(
        self,
        category: str,
        preference: str,
        context: str | None = None,
    ):
        """Store a user or agent preference."""

        return await self.memory.long_term.add_preference(
            category=category,
            preference=preference,
            context=context,
        )

    async def search_long_term(
        self,
        query: str,
        entity_name: str | None = None,
        limit: int = 10,
    ):
        """Search the long-term memory graph."""

        entities = (
            await self.memory.long_term.search_entities(
                query=query,
                limit=limit,
            )
        )

        preferences = (
            await self.memory.long_term.search_preferences(
                query=query,
            )
        )

        related_entities = []

        if entity_name:
            entity = (
                await self.memory.long_term.get_entity_by_name(
                    entity_name
                )
            )

            if entity:
                related_entities = (
                    await self.memory.long_term.get_related_entities(
                        entity,
                        depth=2,
                    )
                )

        return {
            "entities": entities,
            "preferences": preferences,
            "related_entities": related_entities,
        }


    # reasoning
    async def start_reasoning(
        self,
        task: str,
        session_id: str,
    ):

        return await self.memory.reasoning.start_trace(
            task=task,
            session_id=session_id,
        )

    async def add_reasoning_step(
        self,
        trace_id: str,
        thought: str,
        action: str,
    ):

        return await self.memory.reasoning.add_step(
            trace_id=trace_id,
            thought=thought,
            action=action,
        )

    async def record_tool_call(
        self,
        step_id: str,
        tool_name: str,
        arguments: dict,
        result: dict,
        success: bool = True,
    ):
        return await self.memory.reasoning.record_tool_call(
            step_id=step_id,
            tool_name=tool_name,
            arguments=arguments,
            result=result,
            status="success" if success else "failed",
        )

    async def complete_reasoning(
        self,
        trace_id: str,
        outcome: str,
        success: bool = True,
    ):
        return await self.memory.reasoning.complete_trace(
            trace_id,
            outcome=outcome,
            success=success,
        )

    async def get_reasoning_context(
        self,
        task: str,
        limit: int = 3,
    ):

        similar_traces = (
            await self.memory.reasoning.get_similar_traces(
                task=task,
                limit=limit,
            )
        )

        return similar_traces


    # context graph

    async def get_context(
        self,
        query: str,
        session_id: str,
        include_short_term: bool = True,
        include_long_term: bool = True,
        include_reasoning: bool = False,
        max_items: int = 10,
    ):
        return await self.memory.get_context(
            query=query,
            session_id=session_id,
            include_short_term=include_short_term,
            include_long_term=include_long_term,
            include_reasoning=include_reasoning,
            max_items=max_items,
        )

    async def get_session_context(
        self,
        session_id: str,
        max_items: int = 10,
    ) -> str:
        conversation = await self.memory.short_term.get_conversation(
            session_id,
            limit=1000,
        )

        if not conversation.messages:
            return ""

        lines = ["### Recent Conversation"]
        for msg in conversation.messages[-max_items:]:
            lines.append(f"**{msg.role.value}**: {msg.content}")

        return "\n".join(lines)

    @property
    def client(self):
        return self.memory
