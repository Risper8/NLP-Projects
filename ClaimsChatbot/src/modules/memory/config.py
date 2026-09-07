import os

from dotenv import load_dotenv
from neo4j_agent_memory import ExtractionConfig, MemorySettings
from neo4j_agent_memory.config import EmbeddingConfig, EmbeddingProvider, ExtractorType, Neo4jConfig
from neo4j_agent_memory.config.settings import SchemaConfig
from neo4j_agent_memory.llm.adapters.litellm import LiteLLMProvider

from src.modules.llm.client import resolve_api_base, resolve_model_string, resolve_provider

load_dotenv()

# think=False (Ollama only): 
_extraction_llm = LiteLLMProvider(
    resolve_model_string(),
    api_base=resolve_api_base(),
    **({"think": False} if resolve_provider() == "ollama" else {}),
)

_INSURANCE_ENTITY_TYPES = [
    "INSURED",
    "CEDANT",
    "BROKER",
    "CLAIM_REFERENCE",
    "CLASS_OF_BUSINESS",
    "LOCATION",
    "MONETARY_AMOUNT",
    "DATE",
]

settings = MemorySettings(
    neo4j=Neo4jConfig(
        uri=os.environ["NEO4J_URI"],
        username=os.environ["NEO4J_USERNAME"],
        password=os.environ["NEO4J_PASSWORD"],
        database=os.environ["NEO4J_DATABASE"],
    ),
    embedding=EmbeddingConfig(
        provider=EmbeddingProvider.SENTENCE_TRANSFORMERS,
        model="all-MiniLM-L6-v2",
        dimensions=384,
    ),
    llm=_extraction_llm,
    schema_config=SchemaConfig(entity_types=_INSURANCE_ENTITY_TYPES),
    extraction=ExtractionConfig(
        extractor_type=ExtractorType.GLINER,
        gliner_device="cpu",
    ),
)
