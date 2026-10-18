"""
Local, deterministic passage retrieval for ClauseLens.

Design & Trade-offs:
- Uses an in-memory BM25-inspired / IDF-weighted lexical scoring algorithm with
  stopword removal, light English/legal suffix stemming, and adjacent bigram
  phrase boosting.
- Requires zero external ML dependencies or embedding downloads, executes in
  milliseconds on CPU, and works reliably on small single-document chunk sets.
- Limitation: Purely lexical retrieval will not match passages that share zero
  root terms with the user's question. Therefore, retrieval failure is never
  treated as proof that a clause is absent from the contract.
"""

from dataclasses import dataclass
import math
import re
from typing import Optional

import config
from src.schemas import Chunk


# Common conversational and query stopwords that should not drive chunk ranking
STOPWORDS: frozenset[str] = frozenset(
    {
        "a", "about", "above", "after", "again", "against", "all", "am", "an",
        "and", "any", "are", "as", "at", "be", "because", "been", "before",
        "being", "below", "between", "both", "but", "by", "can", "could", "did",
        "do", "does", "doing", "down", "during", "each", "few", "for", "from",
        "further", "had", "has", "have", "having", "he", "her", "here", "hers",
        "herself", "him", "himself", "his", "how", "i", "if", "in", "into", "is",
        "it", "its", "itself", "just", "me", "might", "more", "most", "must",
        "my", "myself", "no", "nor", "not", "now", "of", "off", "on", "once",
        "only", "or", "other", "our", "ours", "ourselves", "out", "over", "own",
        "same", "she", "should", "so", "some", "such", "than", "that", "the",
        "their", "theirs", "them", "themselves", "then", "there", "these", "they",
        "this", "those", "through", "to", "too", "under", "until", "up", "very",
        "was", "we", "were", "what", "when", "where", "which", "while", "who",
        "whom", "why", "will", "with", "would", "you", "your", "yours",
        "yourself", "yourselves",
        # Generic document-query filler words
        "agreement", "contract", "document", "clause", "clauses", "tell",
        "explain", "say", "says", "stated", "state", "states", "specify",
        "specified", "does", "happen", "happens", "there", "main",
    }
)
GENERIC_MATCH_TERMS: frozenset[str] = frozenset(
    {
        "tenant",
        "landlord",
        "party",
        "number",
        "name",
        "person",
        "detail",
        "information",
    }
)
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _stem_token(token: str) -> str:
    """
    Apply lightweight, deterministic suffix normalization so common legal
    inflections (e.g., 'terminate' / 'terminating' / 'termination',
    'obligated' / 'obligations', 'penalties' / 'penalty') map to a shared root.
    """
    if len(token) <= 4:
        return token

    # Normalize common suffixes in order from longest to shortest
    if token.endswith("ations") and len(token) > 8:
        return token[:-6]
    if token.endswith("ation") and len(token) > 7:
        return token[:-5]
    if token.endswith("ments") and len(token) > 7:
        return token[:-5]
    if token.endswith("ment") and len(token) > 6:
        return token[:-4]
    if token.endswith("ities") and len(token) > 7:
        return token[:-5] + "y"
    if token.endswith("ties") and len(token) > 6:
        return token[:-4] + "ty"
    if token.endswith("ies") and len(token) > 5:
        return token[:-3] + "y"
    if token.endswith("ing") and len(token) > 6:
        return token[:-3]
    if token.endswith("ed") and len(token) > 5:
        return token[:-2]
    if token.endswith("es") and len(token) > 5:
        return token[:-2]
    if token.endswith("e") and len(token) > 5:
        return token[:-1]
    if token.endswith("s") and not token.endswith("ss") and len(token) > 4:
        return token[:-1]

    return token


def tokenize(text: str, remove_stopwords: bool = True) -> list[str]:
    """
    Extract lowercase alphanumeric tokens, optionally filtering stopwords,
    and normalize morphological suffixes.
    """
    if not text:
        return []

    raw_tokens = _TOKEN_RE.findall(text.lower())
    processed: list[str] = []
    for tok in raw_tokens:
        if remove_stopwords and tok in STOPWORDS:
            continue
        if len(tok) <= 1 and not tok.isdigit():
            continue
        processed.append(_stem_token(tok))
    return processed


@dataclass(frozen=True)
class ScoredChunk:
    """Pairs a retrieved Chunk with its relevance score."""

    chunk: Chunk
    score: float


class ClauseRetriever:
    """
    Deterministic local retriever that ranks document chunks against a natural
    language question.
    """

    def __init__(self, top_k: int = config.RETRIEVAL_TOP_K) -> None:
        if top_k <= 0:
            raise ValueError("top_k must be a positive integer.")
        self.top_k = top_k

    def retrieve(
        self,
        question: str,
        chunks: list[Chunk],
        top_k: Optional[int] = None,
    ) -> list[Chunk]:
        """
        Return up to top_k relevant Chunk objects ordered from highest to lowest score.
        Chunks with zero relevance score are excluded.
        """
        scored = self.retrieve_with_scores(question=question, chunks=chunks, top_k=top_k)
        return [item.chunk for item in scored]

    def retrieve_with_scores(
        self,
        question: str,
        chunks: list[Chunk],
        top_k: Optional[int] = None,
    ) -> list[ScoredChunk]:
        """
        Rank chunks and return ScoredChunk objects for chunks with score > 0.
        """
        if not question or not question.strip() or not chunks:
            return []

        limit = top_k if top_k is not None else self.top_k
        if limit <= 0:
            return []

        query_tokens = tokenize(question, remove_stopwords=True)
        if not query_tokens:
            return []

        unique_query_terms = list(dict.fromkeys(query_tokens))
        query_bigrams = set(zip(query_tokens, query_tokens[1:]))
        distinctive_query_terms = (
            set(unique_query_terms) - GENERIC_MATCH_TERMS
        )
        # Tokenize every chunk once
        chunk_token_lists: list[list[str]] = [
            tokenize(chunk.text, remove_stopwords=True) for chunk in chunks
        ]

        # Compute document frequency (DF) across chunks for each query term
        num_chunks = len(chunks)
        doc_freq: dict[str, int] = {term: 0 for term in unique_query_terms}
        for c_tokens in chunk_token_lists:
            c_set = set(c_tokens)
            for term in unique_query_terms:
                if term in c_set:
                    doc_freq[term] += 1

        avg_chunk_len = (
            sum(len(t) for t in chunk_token_lists) / max(1, num_chunks)
        ) or 1.0

        # BM25 saturation hyperparameters
        k1 = 1.5
        b = 0.75

        scored_results: list[ScoredChunk] = []

        for idx, chunk in enumerate(chunks):
            c_tokens = chunk_token_lists[idx]
            if not c_tokens:
                continue

            # Reject chunks that match only generic query terms.
            if (
                distinctive_query_terms
                and not distinctive_query_terms.intersection(c_tokens)
            ):
                continue

            # Count term frequencies in this chunk
            tf_map: dict[str, int] = {}
            for tok in c_tokens:
                tf_map[tok] = tf_map.get(tok, 0) + 1

            score = 0.0
            matched_distinct_terms = 0
            chunk_len = len(c_tokens)

            for term in unique_query_terms:
                tf = tf_map.get(term, 0)
                if tf == 0:
                    continue

                matched_distinct_terms += 1
                df = doc_freq.get(term, 0)

                # Smooth positive IDF formula suitable for small chunk collections
                idf = math.log(
                    1.0 + (num_chunks - df + 0.5) / (df + 0.5)
                ) + 1.0

                norm_tf = (tf * (k1 + 1.0)) / (
                    tf + k1 * (1.0 - b + b * (chunk_len / avg_chunk_len))
                )

                score += idf * norm_tf

            if score <= 0.0:
                continue

            # Coverage bonus when multiple distinct query concepts appear together
            if len(unique_query_terms) > 1 and matched_distinct_terms > 1:
                coverage_ratio = matched_distinct_terms / len(unique_query_terms)
                score *= 1.0 + (0.35 * coverage_ratio)

            # Bigram phrase bonus for adjacent query terms appearing in order
            if query_bigrams and len(c_tokens) > 1:
                chunk_bigrams = set(zip(c_tokens, c_tokens[1:]))
                common_bigrams = len(query_bigrams.intersection(chunk_bigrams))

                if common_bigrams > 0:
                    score += 1.5 * common_bigrams

            scored_results.append(
                ScoredChunk(chunk=chunk, score=round(score, 6))
            )

        # Sort deterministically: highest score first, tie-breaking by original chunk_id
        scored_results.sort(key=lambda item: (-item.score, item.chunk.chunk_id))
        return scored_results[:limit]