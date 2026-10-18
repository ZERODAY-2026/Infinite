"""Local legal knowledge retrieval for ClauseLens."""

from src.legal_knowledge.models import LegalPassage
from src.legal_knowledge.ingestion import LegalKnowledgeStore
from src.legal_knowledge.legal_retrieval import LegalKnowledgeRetriever
from src.legal_knowledge.legal_citation_validator import (
    LegalCitationValidator,
)

__all__ = [
    "LegalPassage",
    "LegalKnowledgeStore",
    "LegalKnowledgeRetriever",
    "LegalCitationValidator",
]