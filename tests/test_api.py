"""Tests for FastAPI backend."""

import os
import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch
import tempfile
from pathlib import Path

from powerbi_rag.api.main import app
from powerbi_rag.utils.caching import cache_manager


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def mock_rag_components():
    """Mock RAG components for testing."""
    cache_manager.clear_all_caches()
    with patch('powerbi_rag.api.main.vector_store') as mock_vector_store, \
         patch('powerbi_rag.api.main.rag_pipeline') as mock_rag_pipeline, \
         patch('powerbi_rag.api.main.conversation_manager') as mock_conversation_manager:
        
        # Configure mock vector store
        mock_vector_store.get_collection_info.return_value = {
            "name": "test_collection",
            "count": 100,
            "artifact_types": ["table", "measure", "column"]
        }
        
        # Configure mock RAG pipeline
        mock_rag_pipeline.answer_question = AsyncMock(return_value={
            "answer": "This is a test answer about Power BI",
            "context": [
                {
                    "content": "Test context content",
                    "type": "table",
                    "name": "Sales",
                    "score": 0.85
                }
            ],
            "sources": ["table: Sales"],
            "confidence": 0.85
        })
        mock_rag_pipeline._retrieve_context = AsyncMock(return_value=[
            {
                "id": "table_sales",
                "content": "Test context content",
                "score": 0.85,
                "metadata": {"type": "table", "name": "Sales", "source_file": "test.pbix"},
            }
        ])
        mock_rag_pipeline.llm_model = "test-llm"
        mock_rag_pipeline.temperature = 0.1
        
        # Configure mock conversation manager
        mock_conversation_manager.ask_question = AsyncMock(return_value={
            "answer": "Test conversational answer",
            "context": [],
            "sources": [],
            "confidence": 0.80
        })
        mock_conversation_manager.get_conversation_history.return_value = [
            {"question": "Test question", "answer": "Test answer", "timestamp": 12345}
        ]
        
        yield {
            "vector_store": mock_vector_store,
            "rag_pipeline": mock_rag_pipeline,
            "conversation_manager": mock_conversation_manager
        }
    cache_manager.clear_all_caches()


class TestHealthEndpoint:
    """Test health check endpoint."""
    
    def test_health_check_success(self, client, mock_rag_components):
        """Test successful health check."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] in ["healthy", "degraded"]
        assert "version" in data
        assert "components" in data
        assert "stats" in data
        assert "vector_store" in data["components"]
    
    def test_health_check_without_components(self, client):
        """Test health check when components not initialized."""
        # This tests the actual startup state before components are initialized
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        
        # Should still return valid structure even if degraded
        assert "status" in data
        assert "components" in data


class TestQuestionEndpoint:
    """Test question answering endpoint."""
    
    def test_ask_question_basic(self, client, mock_rag_components):
        """Test basic question asking."""
        question_data = {
            "question": "What tables are in the report?",
            "session_id": "test-session",
            "use_conversation_history": False
        }
        
        response = client.post("/ask", json=question_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "answer" in data
        assert "context" in data
        assert "sources" in data
        assert "confidence" in data
        assert "cached" in data
        
        assert isinstance(data["answer"], str)
        assert isinstance(data["context"], list)
        assert isinstance(data["sources"], list)
        assert isinstance(data["confidence"], float)
        assert isinstance(data["cached"], bool)
    
    def test_ask_question_with_conversation_history(self, client, mock_rag_components):
        """Test question with conversation history."""
        question_data = {
            "question": "Tell me more about that table",
            "session_id": "test-session",
            "use_conversation_history": True
        }
        
        response = client.post("/ask", json=question_data)
        
        assert response.status_code == 200
        data = response.json()
        
        # Should use conversation manager
        mock_rag_components["conversation_manager"].ask_question.assert_called_once()
        
        assert data["answer"] == "Test conversational answer"

    def test_ask_question_preserves_filter_in_conversation_mode(self, client, mock_rag_components):
        """Test filter passthrough when conversation mode is enabled."""
        question_data = {
            "question": "Tell me about sales measures",
            "session_id": "test-session",
            "filter_by_type": "measure",
            "use_conversation_history": True,
        }

        response = client.post("/ask", json=question_data)

        assert response.status_code == 200
        mock_rag_components["conversation_manager"].ask_question.assert_called_once_with(
            "Tell me about sales measures",
            session_id="test-session",
            use_history=True,
            filter_by_type="measure",
        )
    
    def test_ask_question_with_filter(self, client, mock_rag_components):
        """Test question with artifact type filter."""
        question_data = {
            "question": "What measures are available?",
            "filter_by_type": "measure",
            "use_conversation_history": False
        }
        
        response = client.post("/ask", json=question_data)
        
        assert response.status_code == 200
        
        # Should call RAG pipeline with filter
        mock_rag_components["rag_pipeline"].answer_question.assert_called_once()
        call_args = mock_rag_components["rag_pipeline"].answer_question.call_args
        assert call_args[1]["filter_by_type"] == "measure"
    
    def test_ask_question_missing_question(self, client):
        """Test request with missing question."""
        response = client.post("/ask", json={})
        
        assert response.status_code == 422  # Validation error
    
    def test_ask_question_empty_question(self, client, mock_rag_components):
        """Test request with empty question."""
        question_data = {"question": ""}
        
        response = client.post("/ask", json=question_data)
        
        # Should still process but may return error from pipeline
        assert response.status_code == 200


class TestFileUploadEndpoint:
    """Test file upload endpoint."""
    
    def test_upload_pbix_file(self, client):
        """Test PBIX file upload."""
        # Create temporary PBIX file
        with tempfile.NamedTemporaryFile(suffix='.pbix', delete=False) as temp_file:
            temp_file.write(b"dummy PBIX content")
            temp_file_path = temp_file.name
        
        try:
            with open(temp_file_path, 'rb') as f:
                response = client.post(
                    "/upload",
                    files={"file": ("test.pbix", f, "application/octet-stream")}
                )
            
            assert response.status_code == 200
            data = response.json()
            
            assert "message" in data
            assert "file_name" in data
            assert data["file_name"] == "test.pbix"
            assert "uploaded successfully" in data["message"]
            
        finally:
            Path(temp_file_path).unlink(missing_ok=True)
    
    def test_upload_invalid_file_type(self, client):
        """Test upload of non-PBIX file."""
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as temp_file:
            temp_file.write(b"not a PBIX file")
            temp_file_path = temp_file.name
        
        try:
            with open(temp_file_path, 'rb') as f:
                response = client.post(
                    "/upload",
                    files={"file": ("test.txt", f, "text/plain")}
                )
            
            assert response.status_code == 400
            assert "PBIX file" in response.json()["detail"]
            
        finally:
            Path(temp_file_path).unlink(missing_ok=True)


class TestVectorStoreEndpoints:
    """Test vector store management endpoints."""
    
    def test_vector_store_info(self, client, mock_rag_components):
        """Test vector store info endpoint."""
        response = client.get("/vector-store/info")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["name"] == "test_collection"
        assert data["count"] == 100
        assert "artifact_types" in data
    
    def test_vector_store_search(self, client, mock_rag_components):
        """Test vector store search endpoint."""
        # Configure search results
        mock_rag_components["vector_store"].search.return_value = [
            {
                "id": "table_sales",
                "content": "Sales table content",
                "score": 0.85,
                "metadata": {"type": "table", "name": "Sales"}
            }
        ]
        
        response = client.post("/vector-store/search?query=sales&n_results=5")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "results" in data
        assert "query" in data
        assert "count" in data
        assert data["query"] == "sales"
        assert len(data["results"]) == 1
    
    def test_vector_store_search_with_type_filter(self, client, mock_rag_components):
        """Test vector store search with artifact type filter."""
        mock_rag_components["vector_store"].search_by_type.return_value = []
        
        response = client.post("/vector-store/search?query=measures&artifact_type=measure")
        
        assert response.status_code == 200
        
        # Should call search_by_type
        mock_rag_components["vector_store"].search_by_type.assert_called_once_with(
            "measures", "measure", 5
        )
    
    def test_reset_vector_store(self, client, mock_rag_components):
        """Test vector store reset endpoint."""
        response = client.delete("/vector-store/reset")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "reset successfully" in data["message"]
        mock_rag_components["vector_store"].reset_collection.assert_called_once()


class TestCacheEndpoints:
    """Test cache management endpoints."""
    
    @patch('powerbi_rag.api.main.cache_manager')
    def test_cache_stats(self, mock_cache_manager, client):
        """Test cache statistics endpoint."""
        mock_cache_manager.get_stats.return_value = {
            "sqlite": {"response_entries": 50, "embedding_entries": 100},
            "disk": {"disk_entries": 25, "disk_size_mb": 15.2},
            "caching_enabled": True
        }
        
        response = client.get("/cache/stats")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "sqlite" in data
        assert "disk" in data
        assert "caching_enabled" in data
    
    @patch('powerbi_rag.api.main.cache_manager')
    def test_clear_cache(self, mock_cache_manager, client):
        """Test cache clearing endpoint."""
        response = client.delete("/cache/clear")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "cleared successfully" in data["message"]
        mock_cache_manager.clear_all_caches.assert_called_once()
    
    @patch('powerbi_rag.api.main.cache_manager')
    def test_cleanup_cache(self, mock_cache_manager, client):
        """Test cache cleanup endpoint."""
        response = client.post("/cache/cleanup")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "cleaned up" in data["message"]
        mock_cache_manager.cleanup_expired.assert_called_once()


class TestConversationEndpoints:
    """Test conversation management endpoints."""
    
    def test_get_conversation_history(self, client, mock_rag_components):
        """Test getting conversation history."""
        response = client.get("/conversation/history/test-session")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "session_id" in data
        assert "history" in data
        assert "count" in data
        assert data["session_id"] == "test-session"
        
        mock_rag_components["conversation_manager"].get_conversation_history.assert_called_once_with(
            "test-session"
        )
    
    def test_clear_conversation_history(self, client, mock_rag_components):
        """Test clearing conversation history."""
        response = client.delete("/conversation/history/test-session")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "cleared" in data["message"]
        assert "test-session" in data["message"]
        
        mock_rag_components["conversation_manager"].clear_history.assert_called_once_with(
            "test-session"
        )


class TestSpecializedEndpoints:
    """Test specialized Power BI query endpoints."""
    
    def test_explain_measure(self, client, mock_rag_components):
        """Test measure explanation endpoint."""
        mock_rag_components["rag_pipeline"].explain_measure = AsyncMock(return_value={
            "answer": "Total Sales is calculated using SUM(Sales[Amount])",
            "context": [],
            "sources": [],
            "confidence": 0.9
        })
        
        response = client.post("/explain/measure?measure_name=Total Sales")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "SUM(Sales[Amount])" in data["answer"]
        mock_rag_components["rag_pipeline"].explain_measure.assert_called_once_with("Total Sales")
    
    def test_describe_table(self, client, mock_rag_components):
        """Test table description endpoint."""
        mock_rag_components["rag_pipeline"].describe_table = AsyncMock(return_value={
            "answer": "Customer table contains customer information with 4 columns",
            "context": [],
            "sources": [],
            "confidence": 0.88
        })
        
        response = client.post("/describe/table?table_name=Customer")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "Customer table" in data["answer"]
        mock_rag_components["rag_pipeline"].describe_table.assert_called_once_with("Customer")


@pytest.mark.integration
class TestAPIIntegration:
    """Integration tests for the API with real components."""
    
    @pytest.mark.skipif(
        not all(key in os.environ for key in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY"]),
        reason="API keys not available"
    )
    def test_full_api_workflow(self, client):
        """Test complete API workflow with real components (requires API keys)."""
        # This is test the full workflow:
        # 1.Upload PBIX file
        # 2.Check vector store info
        # 3.Ask questions/queries
        # 4.Check conversation history
        
        # Note: This test requires proper component initialization
        # and must be run in a real environment with API keys
        pass
