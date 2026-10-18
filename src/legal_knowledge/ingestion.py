"""Storage and ingestion for the local legal knowledge base."""

import json
import sqlite3
from pathlib import Path

from src.legal_knowledge.models import LegalPassage


class LegalKnowledgeStore:
    def __init__(self, database_path: str | Path):
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        self._initialize_database()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize_database(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS legal_passages (
                    passage_id TEXT PRIMARY KEY,
                    source_title TEXT NOT NULL,
                    source_url TEXT NOT NULL,
                    jurisdiction TEXT NOT NULL,
                    document_type TEXT NOT NULL,
                    provision TEXT NOT NULL,
                    text TEXT NOT NULL,
                    effective_from TEXT NOT NULL DEFAULT '',
                    effective_to TEXT NOT NULL DEFAULT '',
                    source_version TEXT NOT NULL DEFAULT '',
                    verified INTEGER NOT NULL DEFAULT 0,
                    notes TEXT NOT NULL DEFAULT ''
                )
                """
            )

            # FTS5 provides local full-text search.
            connection.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS legal_passages_fts
                USING fts5(
                    passage_id UNINDEXED,
                    source_title,
                    provision,
                    text,
                    jurisdiction,
                    document_type,
                    tokenize='unicode61'
                )
                """
            )

    def upsert(self, passage: LegalPassage) -> None:
        """Insert or update a passage and its FTS entry."""

        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO legal_passages (
                    passage_id,
                    source_title,
                    source_url,
                    jurisdiction,
                    document_type,
                    provision,
                    text,
                    effective_from,
                    effective_to,
                    source_version,
                    verified,
                    notes
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(passage_id) DO UPDATE SET
                    source_title = excluded.source_title,
                    source_url = excluded.source_url,
                    jurisdiction = excluded.jurisdiction,
                    document_type = excluded.document_type,
                    provision = excluded.provision,
                    text = excluded.text,
                    effective_from = excluded.effective_from,
                    effective_to = excluded.effective_to,
                    source_version = excluded.source_version,
                    verified = excluded.verified,
                    notes = excluded.notes
                """,
                (
                    passage.passage_id,
                    passage.source_title,
                    passage.source_url,
                    passage.jurisdiction,
                    passage.document_type,
                    passage.provision,
                    passage.text,
                    passage.effective_from,
                    passage.effective_to,
                    passage.source_version,
                    int(passage.verified),
                    passage.notes,
                ),
            )

            connection.execute(
                """
                DELETE FROM legal_passages_fts
                WHERE passage_id = ?
                """,
                (passage.passage_id,),
            )

            connection.execute(
                """
                INSERT INTO legal_passages_fts (
                    passage_id,
                    source_title,
                    provision,
                    text,
                    jurisdiction,
                    document_type
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    passage.passage_id,
                    passage.source_title,
                    passage.provision,
                    passage.text,
                    passage.jurisdiction,
                    passage.document_type,
                ),
            )

    def import_jsonl(self, jsonl_path: str | Path) -> int:
        """Import legal passages from a JSON Lines file.

        Returns the number of records imported.
        """

        path = Path(jsonl_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Legal records file not found: {path}"
            )

        imported = 0

        with path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue

                try:
                    record = json.loads(line)
                    passage = LegalPassage.from_dict(record)
                except (json.JSONDecodeError, TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Invalid legal record on line {line_number}: {exc}"
                    ) from exc

                self.upsert(passage)
                imported += 1

        return imported

    def get_by_id(self, passage_id: str) -> LegalPassage | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT *
                FROM legal_passages
                WHERE passage_id = ?
                """,
                (passage_id,),
            ).fetchone()

        if row is None:
            return None

        return self._row_to_passage(row)

    def count(self, verified_only: bool = False) -> int:
        query = "SELECT COUNT(*) FROM legal_passages"

        if verified_only:
            query += " WHERE verified = 1"

        with self._connect() as connection:
            result = connection.execute(query).fetchone()

        return int(result[0])

    @staticmethod
    def _row_to_passage(row: sqlite3.Row) -> LegalPassage:
        return LegalPassage(
            passage_id=row["passage_id"],
            source_title=row["source_title"],
            source_url=row["source_url"],
            jurisdiction=row["jurisdiction"],
            document_type=row["document_type"],
            provision=row["provision"],
            text=row["text"],
            effective_from=row["effective_from"],
            effective_to=row["effective_to"],
            source_version=row["source_version"],
            verified=bool(row["verified"]),
            notes=row["notes"],
        )