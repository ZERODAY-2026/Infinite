"""Local retrieval of relevant legal passages."""

import re
import sqlite3
from pathlib import Path

from src.legal_knowledge.ingestion import LegalKnowledgeStore
from src.legal_knowledge.models import LegalPassage


class LegalKnowledgeRetriever:
    def __init__(self, database_path: str | Path):
        self.store = LegalKnowledgeStore(database_path)
        self.database_path = Path(database_path)

    @staticmethod
    def _make_fts_query(query: str) -> str:
        """Convert user text into a safe FTS5 OR query."""

        terms = re.findall(r"[A-Za-z0-9_]+", query.lower())

        # Remove duplicates while preserving order.
        terms = list(dict.fromkeys(terms))

        if not terms:
            return ""

        # Quote terms so punctuation cannot change FTS syntax.
        return " OR ".join(f'"{term}"' for term in terms[:30])

    def search(
        self,
        query: str,
        jurisdiction: str | None = None,
        top_k: int = 5,
        verified_only: bool = True,
    ) -> list[LegalPassage]:
        if top_k <= 0:
            return []

        fts_query = self._make_fts_query(query)

        if not fts_query:
            return []

        sql = """
            SELECT
                p.*,
                bm25(legal_passages_fts) AS search_rank
            FROM legal_passages_fts
            JOIN legal_passages AS p
              ON p.passage_id = legal_passages_fts.passage_id
            WHERE legal_passages_fts MATCH ?
        """

        parameters: list[object] = [fts_query]

        if jurisdiction:
            sql += " AND LOWER(p.jurisdiction) = LOWER(?)"
            parameters.append(jurisdiction.strip())

        if verified_only:
            sql += " AND p.verified = 1"

        # FTS5 bm25 scores are generally better when lower.
        sql += " ORDER BY search_rank ASC LIMIT ?"
        parameters.append(top_k)

        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row

        try:
            rows = connection.execute(sql, parameters).fetchall()
        finally:
            connection.close()

        return [
            LegalKnowledgeStore._row_to_passage(row)
            for row in rows
        ]