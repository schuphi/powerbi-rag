"""Lexical retrieval utilities for Power BI artifacts."""

import math
import re
from collections import Counter
from typing import Dict, Iterable, List, Optional

from ..extraction.models import PowerBIArtifact


class LexicalArtifactStore:
    """Simple BM25-style lexical search over artifact content."""

    def __init__(self, records: Optional[List[Dict]] = None):
        """Initialize the lexical store."""
        self.records: List[Dict] = []
        self._tokenized_documents: List[List[str]] = []
        self._document_frequencies: Dict[str, int] = {}
        self._average_document_length = 0.0

        if records:
            self.index_records(records)

    @classmethod
    def from_artifacts(cls, artifacts: List[PowerBIArtifact]) -> "LexicalArtifactStore":
        """Build a lexical store directly from PowerBIArtifact models."""
        records = [
            {
                "id": artifact.id,
                "content": artifact.content,
                "metadata": {
                    "type": artifact.type,
                    "name": artifact.name,
                    "source_file": artifact.source_file or "",
                    **artifact.metadata,
                },
            }
            for artifact in artifacts
        ]
        return cls(records)

    def index_records(self, records: List[Dict]) -> None:
        """Replace the current lexical index with the given records."""
        self.records = records
        self._tokenized_documents = [self._tokenize(record.get("content", "")) for record in records]
        self._document_frequencies = {}

        total_length = 0
        for tokens in self._tokenized_documents:
            total_length += len(tokens)
            for token in set(tokens):
                self._document_frequencies[token] = self._document_frequencies.get(token, 0) + 1

        self._average_document_length = total_length / len(self._tokenized_documents) if self._tokenized_documents else 0.0

    def search(
        self,
        query: str,
        n_results: int = 5,
        artifact_type: Optional[str] = None
    ) -> List[Dict]:
        """Search indexed artifacts using lexical similarity."""
        if not self.records:
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        scored_results = []
        for index, record in enumerate(self.records):
            metadata = record.get("metadata", {})
            if artifact_type and metadata.get("type") != artifact_type:
                continue

            score = self._bm25_score(query_tokens, self._tokenized_documents[index])
            if score <= 0:
                continue

            scored_results.append(
                {
                    "id": record.get("id", ""),
                    "content": record.get("content", ""),
                    "metadata": metadata,
                    "score": score,
                    "retrieval_method": "lexical",
                }
            )

        scored_results.sort(key=lambda item: item["score"], reverse=True)
        if not scored_results:
            return []

        max_score = scored_results[0]["score"] or 1.0
        for result in scored_results:
            result["raw_score"] = result["score"]
            result["score"] = round(result["score"] / max_score, 4)

        return scored_results[:n_results]

    def _bm25_score(self, query_tokens: List[str], document_tokens: List[str], k1: float = 1.5, b: float = 0.75) -> float:
        """Compute a BM25-style score for a tokenized document."""
        if not document_tokens:
            return 0.0

        document_term_counts = Counter(document_tokens)
        score = 0.0
        document_length = len(document_tokens)
        avg_document_length = self._average_document_length or 1.0
        total_documents = len(self._tokenized_documents) or 1

        for token in query_tokens:
            if token not in document_term_counts:
                continue

            document_frequency = self._document_frequencies.get(token, 0)
            inverse_document_frequency = math.log(
                1 + ((total_documents - document_frequency + 0.5) / (document_frequency + 0.5))
            )
            term_frequency = document_term_counts[token]
            denominator = term_frequency + k1 * (1 - b + b * (document_length / avg_document_length))
            score += inverse_document_frequency * ((term_frequency * (k1 + 1)) / denominator)

        return score

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize content into lowercased alphanumeric units."""
        return [
            token
            for token in re.findall(r"[a-z0-9]+", text.lower())
            if len(token) > 1
        ]
