"""RAG pipeline for Power BI question answering."""

import asyncio
import re
import time
from typing import Dict, List, Optional

from ..utils.config import settings
from .hybrid_retriever import HybridRetriever
from .vector_store import ChromaVectorStore


AsyncAnthropic = None


class PowerBIRAGPipeline:
    """RAG pipeline for answering questions about Power BI reports."""
    
    def __init__(
        self,
        vector_store: Optional[ChromaVectorStore] = None,
        llm_model: Optional[str] = None,
        max_context_artifacts: int = 5,
        temperature: float = 0.1
    ):
        """Initialize RAG pipeline."""
        self.vector_store = vector_store
        if self.vector_store is None:
            try:
                self.vector_store = ChromaVectorStore(
                    embedding_function="openai" if settings.openai_api_key else "sentence_transformers"
                )
            except Exception:
                self.vector_store = None

        self.llm_model = llm_model or settings.llm_model
        self.max_context_artifacts = max_context_artifacts
        self.temperature = temperature
        self.retrieval_mode = self._processing_setting("retrieval_mode", "dense")
        self.retriever = HybridRetriever(
            vector_store=self.vector_store,
            retrieval_mode=self.retrieval_mode,
            dense_weight=self._processing_setting("hybrid_dense_weight", 0.6),
            lexical_weight=self._processing_setting("hybrid_lexical_weight", 0.4),
        )
        
        # Initialize Anthropic client
        if not settings.anthropic_api_key:
            raise ValueError("Anthropic API key is required")

        anthropic_client_cls = AsyncAnthropic
        if anthropic_client_cls is None:
            try:
                from anthropic import AsyncAnthropic as anthropic_client_cls
            except ImportError as exc:
                raise ImportError(
                    "anthropic is required to use the RAG pipeline. Install project dependencies first."
                ) from exc

        self.anthropic_client = anthropic_client_cls(api_key=settings.anthropic_api_key)
        
        # Set up prompts
        self.system_prompt = self._create_system_prompt()
        self.user_prompt_template = self._create_user_prompt_template()
    
    def _create_system_prompt(self) -> str:
        """Create system prompt for the assistant."""
        return """You are a Power BI expert assistant. You help users understand Power BI reports, DAX formulas, data models, and visualizations.

Your expertise includes:
- Explaining DAX measures and their calculations
- Describing table relationships and data models
- Analyzing report structure and visualizations
- Providing insights about data sources and transformations
- Helping with Power BI best practices

Guidelines:
1. Always base your answers on the provided context from the Power BI report
2. If information is not available in the context, clearly state this
3. Provide specific examples and references when possible
4. Explain technical concepts in clear, understandable language
5. Include relevant table names, column names, and measure names in your responses
6. When discussing DAX, explain the logic and purpose of formulas

Response format:
- Provide direct, helpful answers
- Include specific references to Power BI artifacts (tables, measures, columns, visuals)
- Use technical terminology appropriately but explain complex concepts
- Structure responses clearly with bullet points or sections when helpful"""
    
    def _create_user_prompt_template(self) -> str:
        """Create user prompt template."""
        return """Based on the Power BI report context provided below, please answer the user's question.

**Context from Power BI Report:**
{context}

**User Question:** {question}

**Answer:**"""
    
    async def answer_question(
        self,
        question: str,
        filter_by_type: Optional[str] = None,
        include_metadata: bool = True
    ) -> Dict:
        """Answer a question about the Power BI report."""
        
        # Retrieve relevant context
        context_results = await self._retrieve_context(
            question,
            filter_by_type=filter_by_type
        )
        
        if not context_results:
            return {
                "answer": "I don't have enough information from the Power BI report to answer this question. Please make sure the report has been processed and added to the knowledge base.",
                "context": [],
                "sources": [],
                "confidence": 0.0
            }
        
        # Generate answer
        answer = await self._generate_answer(question, context_results)
        
        # Prepare response
        response = {
            "answer": answer,
            "context": [
                {
                    "content": result["content"],
                    "type": result["metadata"].get("type", "unknown"),
                    "name": result["metadata"].get("name", ""),
                    "score": result["score"]
                }
                for result in context_results
            ] if include_metadata else [],
            "sources": self._extract_sources(context_results),
            "confidence": self._calculate_confidence(context_results)
        }
        
        return response
    
    async def _retrieve_context(
        self,
        question: str,
        filter_by_type: Optional[str] = None
    ) -> List[Dict]:
        """Retrieve relevant context from vector store."""

        results = self.retriever.search(
            query=question,
            n_results=self.max_context_artifacts,
            artifact_type=filter_by_type,
        )
        
        # Filter results by relevance score
        filtered_results = [
            result for result in results
            if result["score"] >= self._processing_setting("min_relevance_score", 0.3)
        ]
        
        return filtered_results
    
    async def _generate_answer(self, question: str, context_results: List[Dict]) -> str:
        """Generate answer using the language model."""
        
        # Prepare context string
        context_parts = []
        for i, result in enumerate(context_results, 1):
            metadata = result.get("metadata", {})
            context_part = f"""
{i}. **{metadata.get('type', 'Unknown').title()}**: {metadata.get('name', 'Unnamed')}
   Content: {result['content']}
   Relevance: {result['score']:.2f}
"""
            context_parts.append(context_part)
        
        context_string = "\n".join(context_parts)
        
        # Format the prompt
        user_prompt = self.user_prompt_template.format(
            context=context_string,
            question=question
        )
        
        try:
            # Call Anthropic API
            response = await self.anthropic_client.messages.create(
                model=self.llm_model,
                max_tokens=settings.processing.max_tokens_per_request,
                temperature=self.temperature,
                messages=[
                    {"role": "user", "content": user_prompt}
                ],
                system=self.system_prompt
            )
            
            return response.content[0].text if response.content else "I couldn't generate a response."
            
        except Exception as e:
            return f"Error generating answer: {str(e)}"
    
    def _extract_sources(self, context_results: List[Dict]) -> List[str]:
        """Extract source references from context results."""
        sources = []
        
        for result in context_results:
            metadata = result.get("metadata", {})
            source_file = metadata.get("source_file", "")
            artifact_name = metadata.get("name", "")
            artifact_type = metadata.get("type", "")
            
            if source_file and artifact_name:
                source = f"{artifact_type}: {artifact_name}"
                if source not in sources:
                    sources.append(source)
        
        return sources
    
    def _calculate_confidence(self, context_results: List[Dict]) -> float:
        """Calculate confidence score based on context quality."""
        if not context_results:
            return 0.0
        
        # Use average similarity score as confidence
        avg_score = sum(result["score"] for result in context_results) / len(context_results)
        
        # Apply confidence adjustments
        if len(context_results) >= 3:
            confidence_boost = 0.1
        else:
            confidence_boost = 0.0
        
        confidence = min(avg_score + confidence_boost, 1.0)
        return round(confidence, 2)
    
    async def explain_measure(self, measure_name: str) -> Dict:
        """Explain a specific DAX measure."""
        question = f"Explain the DAX measure '{measure_name}' and how it works"
        return await self.answer_question(question, filter_by_type="measure")
    
    async def describe_table(self, table_name: str) -> Dict:
        """Describe a specific table and its structure."""
        question = f"Describe the table '{table_name}', its columns, and relationships"
        return await self.answer_question(question, filter_by_type="table")
    
    async def analyze_visual(self, visual_name: str) -> Dict:
        """Analyze a specific visual."""
        question = f"Analyze the visual '{visual_name}' and explain what it shows"
        return await self.answer_question(question, filter_by_type="visual")
    
    async def find_relationships(self, table_name: str) -> Dict:
        """Find relationships involving a specific table."""
        question = f"What relationships involve the table '{table_name}'?"
        return await self.answer_question(question, filter_by_type="relationship")
    
    def get_pipeline_stats(self) -> Dict:
        """Get statistics about the pipeline and vector store."""
        vector_info = self.vector_store.get_collection_info() if self.vector_store else None
        
        return {
            "vector_store": vector_info,
            "retrieval_mode": self.retrieval_mode,
            "retriever": self.retriever.describe_mode(),
            "llm_model": self.llm_model,
            "max_context_artifacts": self.max_context_artifacts,
            "temperature": self.temperature
        }

    def index_artifacts(self, artifacts) -> None:
        """Index artifacts for retrieval."""
        if self.vector_store:
            self.vector_store.add_artifacts(artifacts)
        self.retriever.index_artifacts(artifacts)

    def _processing_setting(self, name: str, default):
        """Safely read a processing setting while remaining compatible with mocked settings."""
        processing = getattr(settings, "processing", None)
        value = getattr(processing, name, default) if processing is not None else default

        if isinstance(default, str):
            return value if isinstance(value, str) else default
        if isinstance(default, (int, float)) and isinstance(value, (int, float)):
            return value
        return default


class ConversationManager:
    """Manage conversation history and context for multi-turn interactions."""
    
    def __init__(self, rag_pipeline: PowerBIRAGPipeline, max_history: int = 10):
        """Initialize conversation manager."""
        self.rag_pipeline = rag_pipeline
        self.max_history = max_history
        self.conversation_history: List[Dict] = []
    
    async def ask_question(
        self,
        question: str,
        session_id: Optional[str] = None,
        use_history: bool = True,
        filter_by_type: Optional[str] = None,
    ) -> Dict:
        """Ask a question with conversation context."""
        
        # Enhance question with conversation context if available
        enhanced_question = (
            self._enhance_question_with_context(question, session_id) if use_history else question
        )
        
        # Get answer from RAG pipeline
        if filter_by_type is None:
            response = await self.rag_pipeline.answer_question(enhanced_question)
        else:
            response = await self.rag_pipeline.answer_question(
                enhanced_question,
                filter_by_type=filter_by_type,
            )
        
        # Store in conversation history
        self._add_to_history(question, response["answer"], session_id)
        
        return response
    
    def _enhance_question_with_context(
        self,
        question: str,
        session_id: Optional[str] = None
    ) -> str:
        """Enhance question with relevant conversation context."""
        relevant_history = self.get_conversation_history(session_id)
        if not relevant_history:
            return question
        
        # Get recent relevant context
        recent_context = []
        question_tokens = self._tokenize(question)
        for entry in relevant_history[-3:]:  # Last 3 interactions in the current session
            if question_tokens & self._tokenize(entry["question"]):
                recent_context.append(f"Previous: {entry['question']} -> {entry['answer'][:200]}...")
        
        if recent_context:
            context_str = "\n".join(recent_context)
            enhanced_question = f"Context from previous conversation:\n{context_str}\n\nCurrent question: {question}"
            return enhanced_question
        
        return question
    
    def _add_to_history(
        self,
        question: str,
        answer: str,
        session_id: Optional[str] = None
    ):
        """Add interaction to conversation history."""
        entry = {
            "question": question,
            "answer": answer,
            "session_id": session_id,
            "timestamp": self._get_timestamp(),
        }
        
        self.conversation_history.append(entry)
        
        # Trim history if too long
        if len(self.conversation_history) > self.max_history:
            self.conversation_history = self.conversation_history[-self.max_history:]
    
    def get_conversation_history(self, session_id: Optional[str] = None) -> List[Dict]:
        """Get conversation history for a session."""
        if session_id:
            return [
                entry for entry in self.conversation_history
                if entry.get("session_id") == session_id
            ]
        return self.conversation_history.copy()
    
    def clear_history(self, session_id: Optional[str] = None):
        """Clear conversation history."""
        if session_id:
            self.conversation_history = [
                entry for entry in self.conversation_history
                if entry.get("session_id") != session_id
            ]
        else:
            self.conversation_history.clear()

    def _get_timestamp(self) -> float:
        """Get a monotonic timestamp without requiring an active event loop."""
        try:
            return asyncio.get_running_loop().time()
        except RuntimeError:
            return time.monotonic()

    def _tokenize(self, text: str) -> set[str]:
        """Tokenize text for simple context matching."""
        return {
            token
            for token in re.findall(r"[a-z0-9]+", text.lower())
            if len(token) > 2
        }
