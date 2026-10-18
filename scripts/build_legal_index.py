"""Build the local ClauseLens legal knowledge index."""

import sys
from pathlib import Path

# Ensure the project root is importable when this script is run directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import LEGAL_KNOWLEDGE_DB, LEGAL_KNOWLEDGE_JSONL
from src.legal_knowledge.ingestion import LegalKnowledgeStore


def main() -> None:
    store = LegalKnowledgeStore(LEGAL_KNOWLEDGE_DB)

    if not LEGAL_KNOWLEDGE_JSONL.exists():
        print(
            "No legal records file found at:\n"
            f"  {LEGAL_KNOWLEDGE_JSONL}\n\n"
            "Create the JSONL file and populate it with verified records "
            "before building the index."
        )
        return

    imported = store.import_jsonl(LEGAL_KNOWLEDGE_JSONL)

    print(f"Records processed: {imported}")
    print(f"Records in index: {store.count()}")
    print(f"Verified records: {store.count(verified_only=True)}")
    print(f"Database: {LEGAL_KNOWLEDGE_DB}")


if __name__ == "__main__":
    main()