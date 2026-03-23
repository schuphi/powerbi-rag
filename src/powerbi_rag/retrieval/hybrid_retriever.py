"""Hybrid retrieval that combines dense and lexical search."""

from typing import Dict, List, Optional

from ..extraction.models import PowerBIArtifact
from .lexical_store import LexicalArtifactStore


class HybridRetriever:
    """Combine dense vector retrieval with lexical search."""

    VALID_MODES = {"dense", "lexical", "hybrid"}

    def __init__(
        self,
        vector_store=None,
        retrieval_mode: str = "hybrid",
        dense_weight: float = 0.6,
        lexical_weight: float = 0.4,
    ):
        """Initialize the retriever."""
        self.vector_store = vector_store
        self.retrieval_mode = retrieval_mode if retrieval_mode in self.VALID_MODES else "hybrid"
        self.dense_weight = dense_weight
        self.lexical_weight = lexical_weight
        self.lexical_store: Optional[LexicalArtifactStore] = None

    def index_artifacts(self, artifacts: List[PowerBIArtifact]) -> None:
        """Index artifacts for lexical retrieval."""
        if not artifacts:
            return
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
        self.index_records(records)

    def index_records(self, records: List[Dict]) -> None:
        """Index preformatted retrieval records."""
        if not records:
            return
        existing_records = self.lexical_store.records if self.lexical_store is not None else []
        merged_records = {record["id"]: record for record in existing_records}
        merged_records.update({record["id"]: record for record in records})
        self.lexical_store = LexicalArtifactStore(list(merged_records.values()))

    def search(
        self,
        query: str,
        n_results: int = 5,
        artifact_type: Optional[str] = None,
    ) -> List[Dict]:
        """Search using the configured retrieval mode."""
        dense_results = self._dense_search(query, n_results, artifact_type)
        lexical_results = self._lexical_search(query, n_results, artifact_type)

        if self.retrieval_mode == "dense":
            return dense_results or lexical_results
        if self.retrieval_mode == "lexical":
            return lexical_results or dense_results

        if dense_results and lexical_results:
            return self._merge_results(dense_results, lexical_results, n_results)
        return dense_results or lexical_results

    def describe_mode(self) -> Dict[str, bool]:
        """Describe active retriever capabilities."""
        return {
            "dense_available": self.vector_store is not None,
            "lexical_available": self._ensure_lexical_index(),
            "hybrid_enabled": self.retrieval_mode == "hybrid",
        }

    def _dense_search(
        self,
        query: str,
        n_results: int,
        artifact_type: Optional[str],
    ) -> List[Dict]:
        """Run dense retrieval if a vector store is available."""
        if self.vector_store is None:
            return []

        if artifact_type:
            results = self.vector_store.search_by_type(
                query=query,
                artifact_type=artifact_type,
                n_results=n_results,
            )
        else:
            results = self.vector_store.search(
                query=query,
                n_results=n_results,
            )

        for result in results:
            result["retrieval_method"] = "dense"
        return results

    def _lexical_search(
        self,
        query: str,
        n_results: int,
        artifact_type: Optional[str],
    ) -> List[Dict]:
        """Run lexical retrieval if a lexical index is available."""
        if not self._ensure_lexical_index():
            return []
        return self.lexical_store.search(query, n_results=n_results, artifact_type=artifact_type)

    def _ensure_lexical_index(self) -> bool:
        """Build a lexical index from the vector store on demand if needed."""
        if self.lexical_store is not None:
            return True

        if self.vector_store is None or not hasattr(self.vector_store, "list_artifacts"):
            return False

        try:
            records = self.vector_store.list_artifacts()
        except Exception:
            return False

        if not isinstance(records, list) or not records:
            return False

        self.lexical_store = LexicalArtifactStore(records)
        return True

    def _merge_results(
        self,
        dense_results: List[Dict],
        lexical_results: List[Dict],
        n_results: int,
    ) -> List[Dict]:
        """Merge dense and lexical rankings with weighted normalized scores."""
        merged: Dict[str, Dict] = {}
        dense_scale = dense_results[0]["score"] if dense_results else 1.0
        lexical_scale = lexical_results[0]["score"] if lexical_results else 1.0

        for result in dense_results:
            merged[result["id"]] = {
                **result,
                "dense_score": result["score"],
                "lexical_score": 0.0,
                "retrieval_method": "hybrid",
            }

        for result in lexical_results:
            if result["id"] in merged:
                merged[result["id"]]["lexical_score"] = result["score"]
                merged[result["id"]]["content"] = merged[result["id"]].get("content") or result.get("content", "")
                merged[result["id"]]["metadata"] = merged[result["id"]].get("metadata") or result.get("metadata", {})
            else:
                merged[result["id"]] = {
                    **result,
                    "dense_score": 0.0,
                    "lexical_score": result["score"],
                    "retrieval_method": "hybrid",
                }

        merged_results = []
        for result in merged.values():
            dense_component = (result.get("dense_score", 0.0) / dense_scale) if dense_scale else 0.0
            lexical_component = (result.get("lexical_score", 0.0) / lexical_scale) if lexical_scale else 0.0
            result["score"] = round(
                (self.dense_weight * dense_component) + (self.lexical_weight * lexical_component),
                4,
            )
            merged_results.append(result)

        merged_results.sort(key=lambda item: item["score"], reverse=True)
        return merged_results[:n_results]
