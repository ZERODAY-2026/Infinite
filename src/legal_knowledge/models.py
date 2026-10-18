"""Data models for legal knowledge records."""

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class LegalPassage:
    passage_id: str
    source_title: str
    source_url: str
    jurisdiction: str
    document_type: str
    provision: str
    text: str
    effective_from: str = ""
    effective_to: str = ""
    source_version: str = ""
    verified: bool = False
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LegalPassage":
        required = [
            "passage_id",
            "source_title",
            "source_url",
            "jurisdiction",
            "document_type",
            "provision",
            "text",
        ]

        missing = [
            field for field in required
            if not str(data.get(field, "")).strip()
        ]

        if missing:
            raise ValueError(
                f"Legal record is missing required fields: {missing}"
            )

        return cls(
            passage_id=str(data["passage_id"]).strip(),
            source_title=str(data["source_title"]).strip(),
            source_url=str(data["source_url"]).strip(),
            jurisdiction=str(data["jurisdiction"]).strip(),
            document_type=str(data["document_type"]).strip(),
            provision=str(data["provision"]).strip(),
            text=str(data["text"]).strip(),
            effective_from=str(data.get("effective_from", "")).strip(),
            effective_to=str(data.get("effective_to", "")).strip(),
            source_version=str(data.get("source_version", "")).strip(),
            verified=bool(data.get("verified", False)),
            notes=str(data.get("notes", "")).strip(),
        )