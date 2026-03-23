"""Tests for lexical and hybrid retrieval."""

from unittest.mock import Mock

from powerbi_rag.extraction.models import ArtifactType, PowerBIArtifact
from powerbi_rag.retrieval.hybrid_retriever import HybridRetriever
from powerbi_rag.retrieval.lexical_store import LexicalArtifactStore


def _sample_artifacts():
    return [
        PowerBIArtifact(
            id="table_sales",
            type=ArtifactType.TABLE,
            name="Sales",
            content="Sales table contains revenue, order quantity, and customer transactions.",
            metadata={"type": "table", "name": "Sales"},
        ),
        PowerBIArtifact(
            id="measure_total_sales",
            type=ArtifactType.MEASURE,
            name="Total Sales",
            content="Total Sales measure calculates sales amount and revenue using DAX.",
            metadata={"type": "measure", "name": "Total Sales"},
        ),
        PowerBIArtifact(
            id="table_customer",
            type=ArtifactType.TABLE,
            name="Customer",
            content="Customer table contains names, territory, and segmentation details.",
            metadata={"type": "table", "name": "Customer"},
        ),
    ]


def test_lexical_store_ranks_sales_results_first():
    """Lexical search should rank the most relevant sales artifact first."""
    store = LexicalArtifactStore.from_artifacts(_sample_artifacts())

    results = store.search("sales revenue measure", n_results=2)

    assert len(results) == 2
    assert results[0]["id"] in {"table_sales", "measure_total_sales"}
    assert results[0]["score"] >= results[1]["score"]


def test_lexical_store_respects_type_filter():
    """Lexical search should filter artifacts by type."""
    store = LexicalArtifactStore.from_artifacts(_sample_artifacts())

    results = store.search("sales", n_results=5, artifact_type="measure")

    assert len(results) == 1
    assert results[0]["metadata"]["type"] == "measure"


def test_hybrid_retriever_merges_dense_and_lexical_results():
    """Hybrid retrieval should combine dense and lexical evidence."""
    vector_store = Mock()
    vector_store.search.return_value = [
        {
            "id": "table_sales",
            "content": "Sales table contains revenue and transactions.",
            "score": 0.9,
            "metadata": {"type": "table", "name": "Sales"},
        },
        {
            "id": "table_customer",
            "content": "Customer table contains territories.",
            "score": 0.6,
            "metadata": {"type": "table", "name": "Customer"},
        },
    ]
    vector_store.list_artifacts.return_value = [
        {
            "id": "table_sales",
            "content": "Sales table contains revenue and transactions.",
            "metadata": {"type": "table", "name": "Sales"},
        },
        {
            "id": "measure_total_sales",
            "content": "Total Sales measure calculates sales amount and revenue using DAX.",
            "metadata": {"type": "measure", "name": "Total Sales"},
        },
    ]

    retriever = HybridRetriever(vector_store=vector_store, retrieval_mode="hybrid")
    results = retriever.search("sales revenue", n_results=3)

    assert len(results) >= 2
    assert results[0]["retrieval_method"] == "hybrid"
    assert {result["id"] for result in results} >= {"table_sales", "measure_total_sales"}


def test_hybrid_retriever_falls_back_to_lexical_only():
    """Hybrid retriever should still work without a vector store."""
    retriever = HybridRetriever(vector_store=None, retrieval_mode="hybrid")
    retriever.index_artifacts(_sample_artifacts())

    results = retriever.search("customer segmentation", n_results=2)

    assert len(results) >= 1
    assert results[0]["id"] == "table_customer"
