"""Tests for RAG pipeline functionality."""

import asyncio
import os
import pytest
from unittest.mock import Mock, AsyncMock, patch

from powerbi_rag.retrieval.rag_pipeline import PowerBIRAGPipeline, ConversationManager
from powerbi_rag.retrieval.vector_store import ChromaVectorStore
from powerbi_rag.extraction.models import PowerBIArtifact, ArtifactType


class TestPowerBIRAGPipeline:
    """Test RAG pipeline functionality."""
    
    @pytest.fixture
    def mock_vector_store(self):
        """Create mock vector store."""
        mock_store = Mock(spec=ChromaVectorStore)
        mock_store.search.return_value = [
            {
                "id": "table_sales",
                "content": "Sales table contains customer transactions and revenue data",
                "score": 0.85,
                "metadata": {
                    "type": "table",
                    "name": "Sales",
                    "source_file": "test.pbix"
                }
            },
            {
                "id": "measure_total_revenue",
                "content": "Total Revenue measure calculates sum of all sales amounts using DAX: SUM(Sales[Amount])",
                "score": 0.78,
                "metadata": {
                    "type": "measure", 
                    "name": "Total Revenue",
                    "source_file": "test.pbix"
                }
            }
        ]
        return mock_store
    
    @pytest.fixture
    def mock_anthropic_response(self):
        """Create mock Anthropic API response."""
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = "Based on the Power BI report, the Sales table contains customer transaction data with measures like Total Revenue that sum sales amounts."
        return mock_response
    
    @pytest.fixture
    def rag_pipeline(self, mock_vector_store):
        """Create RAG pipeline with mocked dependencies."""
        with patch('powerbi_rag.retrieval.rag_pipeline.AsyncAnthropic') as mock_anthropic:
            mock_anthropic.return_value = AsyncMock()
            
            # Mock settings
            with patch('powerbi_rag.retrieval.rag_pipeline.settings') as mock_settings:
                mock_settings.anthropic_api_key = "test-key"
                mock_settings.llm_model = "claude-3-sonnet-20240229"
                mock_settings.processing.max_tokens_per_request = 4000
                
                pipeline = PowerBIRAGPipeline(vector_store=mock_vector_store)
                return pipeline
    
    def test_pipeline_initialization(self, mock_vector_store):
        """Test RAG pipeline initialization."""
        with patch('powerbi_rag.retrieval.rag_pipeline.AsyncAnthropic') as mock_anthropic:
            with patch('powerbi_rag.retrieval.rag_pipeline.settings') as mock_settings:
                mock_settings.anthropic_api_key = "test-key"
                mock_settings.llm_model = "claude-3-sonnet-20240229"
                
                pipeline = PowerBIRAGPipeline(vector_store=mock_vector_store)
                
                assert pipeline.vector_store == mock_vector_store
                assert pipeline.llm_model == "claude-3-sonnet-20240229"
                assert pipeline.max_context_artifacts == 5
                assert pipeline.system_prompt is not None
    
    @pytest.mark.asyncio
    async def test_retrieve_context(self, rag_pipeline, mock_vector_store):
        """Test context retrieval."""
        question = "What is the Sales table?"
        
        context = await rag_pipeline._retrieve_context(question)
        
        mock_vector_store.search.assert_called_once_with(
            query=question,
            n_results=rag_pipeline.max_context_artifacts
        )
        
        assert len(context) == 2  # Both results have score > 0.5
        assert context[0]["score"] == 0.85
        assert context[1]["score"] == 0.78
    
    @pytest.mark.asyncio
    async def test_retrieve_context_with_filter(self, rag_pipeline, mock_vector_store):
        """Test context retrieval with type filter."""
        question = "What measures are available?"
        filter_type = "measure"
        mock_vector_store.search_by_type.return_value = mock_vector_store.search.return_value
        
        context = await rag_pipeline._retrieve_context(question, filter_by_type=filter_type)
        
        mock_vector_store.search_by_type.assert_called_once_with(
            query=question,
            artifact_type=filter_type,
            n_results=rag_pipeline.max_context_artifacts
        )
    
    @pytest.mark.asyncio
    async def test_generate_answer(self, rag_pipeline, mock_anthropic_response):
        """Test answer generation."""
        rag_pipeline.anthropic_client.messages.create = AsyncMock(return_value=mock_anthropic_response)
        
        question = "What is the Sales table?"
        context_results = [
            {
                "content": "Sales table contains transaction data",
                "score": 0.85,
                "metadata": {"type": "table", "name": "Sales"}
            }
        ]
        
        answer = await rag_pipeline._generate_answer(question, context_results)
        
        assert isinstance(answer, str)
        assert len(answer) > 0
        
        # Verify API call
        rag_pipeline.anthropic_client.messages.create.assert_called_once()
        call_args = rag_pipeline.anthropic_client.messages.create.call_args
        assert call_args[1]["model"] == rag_pipeline.llm_model
        assert call_args[1]["system"] == rag_pipeline.system_prompt
    
    @pytest.mark.asyncio
    async def test_answer_question_full_pipeline(self, rag_pipeline, mock_anthropic_response):
        """Test full question answering pipeline."""
        rag_pipeline.anthropic_client.messages.create = AsyncMock(return_value=mock_anthropic_response)
        
        question = "What tables are in the report?"
        
        response = await rag_pipeline.answer_question(question)
        
        # Check response structure
        assert "answer" in response
        assert "context" in response
        assert "sources" in response
        assert "confidence" in response
        
        # Check response content
        assert isinstance(response["answer"], str)
        assert isinstance(response["context"], list)
        assert isinstance(response["sources"], list)
        assert isinstance(response["confidence"], float)
        assert 0 <= response["confidence"] <= 1
    
    @pytest.mark.asyncio
    async def test_answer_question_no_context(self, rag_pipeline):
        """Test handling when no relevant context is found."""
        # Mock vector store to return no results
        rag_pipeline.vector_store.search.return_value = []
        
        question = "What is the XYZ table?"
        response = await rag_pipeline.answer_question(question)
        
        assert "don't have enough information" in response["answer"].lower()
        assert len(response["context"]) == 0
        assert len(response["sources"]) == 0
        assert response["confidence"] == 0.0
    
    def test_extract_sources(self, rag_pipeline):
        """Test source extraction from context."""
        context_results = [
            {
                "metadata": {
                    "source_file": "test.pbix",
                    "name": "Sales",
                    "type": "table"
                }
            },
            {
                "metadata": {
                    "source_file": "test.pbix", 
                    "name": "Total Revenue",
                    "type": "measure"
                }
            }
        ]
        
        sources = rag_pipeline._extract_sources(context_results)
        
        assert len(sources) == 2
        assert "table: Sales" in sources
        assert "measure: Total Revenue" in sources
    
    def test_calculate_confidence(self, rag_pipeline):
        """Test confidence calculation."""
        # High confidence case
        high_context = [
            {"score": 0.9},
            {"score": 0.85},
            {"score": 0.8}
        ]
        confidence = rag_pipeline._calculate_confidence(high_context)
        assert confidence > 0.8
        
        # Low confidence case  
        low_context = [{"score": 0.3}]
        confidence = rag_pipeline._calculate_confidence(low_context)
        assert confidence < 0.5
        
        # Empty context
        confidence = rag_pipeline._calculate_confidence([])
        assert confidence == 0.0


class TestConversationManager:
    """Test conversation management functionality."""
    
    @pytest.fixture
    def mock_rag_pipeline(self):
        """Create mock RAG pipeline."""
        mock_pipeline = AsyncMock(spec=PowerBIRAGPipeline)
        mock_pipeline.answer_question.return_value = {
            "answer": "Test answer",
            "context": [],
            "sources": [],
            "confidence": 0.8
        }
        return mock_pipeline
    
    @pytest.fixture
    def conversation_manager(self, mock_rag_pipeline):
        """Create conversation manager."""
        return ConversationManager(mock_rag_pipeline, max_history=5)
    
    @pytest.mark.asyncio
    async def test_ask_question_basic(self, conversation_manager, mock_rag_pipeline):
        """Test basic question asking."""
        question = "What is the Sales table?"
        
        response = await conversation_manager.ask_question(question, use_history=False)
        
        mock_rag_pipeline.answer_question.assert_called_once_with(question)
        
        assert response["answer"] == "Test answer"
        assert len(conversation_manager.conversation_history) == 1

    @pytest.mark.asyncio
    async def test_ask_question_preserves_filter(self, conversation_manager, mock_rag_pipeline):
        """Test that filter_by_type is passed to the RAG pipeline."""
        await conversation_manager.ask_question(
            "Explain Total Sales",
            session_id="s1",
            use_history=False,
            filter_by_type="measure",
        )

        mock_rag_pipeline.answer_question.assert_called_once_with(
            "Explain Total Sales",
            filter_by_type="measure",
        )
    
    @pytest.mark.asyncio  
    async def test_conversation_history_tracking(self, conversation_manager):
        """Test conversation history is tracked correctly."""
        questions = [
            "What tables are available?",
            "Tell me about the Sales table",
            "What measures does it have?"
        ]
        
        for question in questions:
            await conversation_manager.ask_question(question, use_history=False)
        
        assert len(conversation_manager.conversation_history) == 3
        
        # Check history content
        for i, question in enumerate(questions):
            assert conversation_manager.conversation_history[i]["question"] == question
            assert conversation_manager.conversation_history[i]["answer"] == "Test answer"
    
    def test_conversation_history_by_session(self, conversation_manager):
        """Test session-based history filtering."""
        # Add entries for different sessions
        conversation_manager._add_to_history("Q1", "A1", "session1")
        conversation_manager._add_to_history("Q2", "A2", "session2")
        conversation_manager._add_to_history("Q3", "A3", "session1")
        
        # Get history for session1
        session1_history = conversation_manager.get_conversation_history("session1")
        assert len(session1_history) == 2
        assert session1_history[0]["question"] == "Q1"
        assert session1_history[1]["question"] == "Q3"
        
        # Get history for session2
        session2_history = conversation_manager.get_conversation_history("session2")
        assert len(session2_history) == 1
        assert session2_history[0]["question"] == "Q2"

    def test_enhance_question_uses_session_scoped_history(self, conversation_manager):
        """Test that context enhancement only considers the current session."""
        conversation_manager._add_to_history("Sales question", "Sales answer", "session1")
        conversation_manager._add_to_history("Customer question", "Customer answer", "session2")

        enhanced = conversation_manager._enhance_question_with_context(
            "Tell me more about sales",
            session_id="session2",
        )

        assert "Customer question" not in enhanced
    
    def test_history_size_limit(self, conversation_manager):
        """Test conversation history size limit."""
        # Add more entries than the limit
        for i in range(10):
            conversation_manager._add_to_history(f"Q{i}", f"A{i}", "test")
        
        # Should only keep the last max_history entries
        assert len(conversation_manager.conversation_history) == conversation_manager.max_history
        assert conversation_manager.conversation_history[0]["question"] == "Q5"  # First kept entry
        assert conversation_manager.conversation_history[-1]["question"] == "Q9"  # Last entry
    
    def test_clear_history(self, conversation_manager):
        """Test clearing conversation history."""
        # Add some history
        conversation_manager._add_to_history("Q1", "A1", "session1")
        conversation_manager._add_to_history("Q2", "A2", "session2")
        
        # Clear specific session
        conversation_manager.clear_history("session1")
        
        remaining_history = conversation_manager.get_conversation_history()
        assert len(remaining_history) == 1
        assert remaining_history[0]["question"] == "Q2"
        
        # Clear all history
        conversation_manager.clear_history()
        assert len(conversation_manager.conversation_history) == 0


@pytest.mark.integration
class TestRAGIntegration:
    """Integration tests for the full RAG system."""
    
    @pytest.fixture
    def sample_artifacts(self):
        """Create sample Power BI artifacts for testing."""
        return [
            PowerBIArtifact(
                id="table_customer",
                type=ArtifactType.TABLE,
                name="Customer",
                content="Customer table contains customer information including CustomerID, Name, Email, and Territory. Used for customer analysis and segmentation.",
                metadata={"column_count": 4, "is_hidden": False},
                source_file="test.pbix",
                tags=["table", "customer", "dimension"]
            ),
            PowerBIArtifact(
                id="measure_total_sales",
                type=ArtifactType.MEASURE,
                name="Total Sales",
                content="Total Sales measure calculates the sum of all sales amounts using DAX formula: CALCULATE(SUM(Sales[Amount]), REMOVEFILTERS(Sales[Date]))",
                metadata={"expression": "CALCULATE(SUM(Sales[Amount]), REMOVEFILTERS(Sales[Date]))", "table_name": "Sales"},
                source_file="test.pbix", 
                tags=["measure", "sales", "kpi"]
            ),
            PowerBIArtifact(
                id="relationship_customer_sales",
                type=ArtifactType.RELATIONSHIP,
                name="Customer -> Sales",
                content="Relationship connects Customer table to Sales table via CustomerID column with one-to-many cardinality and single cross-filter direction.",
                metadata={"from_table": "Customer", "to_table": "Sales", "cardinality": "OneToMany"},
                source_file="test.pbix",
                tags=["relationship", "customer", "sales"]
            )
        ]
    
    @pytest.mark.skipif(
        not all(key in os.environ for key in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY"]),
        reason="API keys not available"
    )
    @pytest.mark.asyncio
    async def test_full_rag_pipeline_with_real_apis(self, sample_artifacts):
        """Test full RAG pipeline with real APIs (requires API keys)."""
        from powerbi_rag.retrieval.vector_store import ChromaVectorStore
        
        # Use temporary vector store
        vector_store = ChromaVectorStore(
            persist_directory="./test_chroma_db",
            collection_name="test_collection"
        )
        
        try:
            # Add sample artifacts
            vector_store.add_artifacts(sample_artifacts)
            
            # Initialize RAG pipeline
            rag_pipeline = PowerBIRAGPipeline(vector_store=vector_store)
            
            # Test questions
            test_questions = [
                "What tables are available in the report?",
                "How is Total Sales calculated?", 
                "What relationships exist between Customer and Sales?"
            ]
            
            for question in test_questions:
                response = await rag_pipeline.answer_question(question)
                
                # Validate response
                assert isinstance(response["answer"], str)
                assert len(response["answer"]) > 50  # Substantial answer
                assert response["confidence"] > 0
                assert len(response["context"]) > 0
                
                print(f"Q: {question}")
                print(f"A: {response['answer'][:100]}...")
                print(f"Confidence: {response['confidence']}")
                print("---")
            
        finally:
            # Clean up test vector store
            import shutil
            shutil.rmtree("./test_chroma_db", ignore_errors=True)
