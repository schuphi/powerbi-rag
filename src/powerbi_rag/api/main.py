"""FastAPI backend for Power BI RAG Assistant."""

import importlib
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from ..extraction.pbix_extractor import PBIXExtractor
from ..utils.caching import cache_manager
from ..utils.config import settings


# Global instances
vector_store: Optional[Any] = None
rag_pipeline: Optional[Any] = None
conversation_manager: Optional[Any] = None
embedding_processor: Optional[Any] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan."""
    # Startup
    global vector_store, rag_pipeline, conversation_manager, embedding_processor

    vector_store = None
    rag_pipeline = None
    conversation_manager = None
    embedding_processor = None

    try:
        vector_store_cls = _import_symbol("powerbi_rag.retrieval.vector_store", "ChromaVectorStore")
        vector_store = vector_store_cls(
            embedding_function="openai" if settings.openai_api_key else "sentence_transformers"
        )
    except Exception as exc:
        print(f"Vector store unavailable: {exc}")

    try:
        embedding_processor_cls = _import_symbol(
            "powerbi_rag.processing.embeddings",
            "EmbeddingProcessor",
        )
        embedding_processor = embedding_processor_cls(
            provider="openai" if settings.openai_api_key else "sentence_transformers"
        )
    except Exception as exc:
        print(f"Embedding processor unavailable: {exc}")

    try:
        rag_pipeline_cls = _import_symbol("powerbi_rag.retrieval.rag_pipeline", "PowerBIRAGPipeline")
        conversation_manager_cls = _import_symbol(
            "powerbi_rag.retrieval.rag_pipeline",
            "ConversationManager",
        )
        if settings.anthropic_api_key:
            rag_pipeline = rag_pipeline_cls(vector_store=vector_store)
            conversation_manager = conversation_manager_cls(rag_pipeline)
        elif not settings.anthropic_api_key:
            print("RAG pipeline unavailable: ANTHROPIC_API_KEY is not configured")
    except Exception as exc:
        print(f"RAG pipeline unavailable: {exc}")

    if any([vector_store, rag_pipeline, embedding_processor]):
        print("Power BI RAG Assistant API started in degraded mode")
    else:
        print("Power BI RAG Assistant API started with no optional components available")
    
    yield
    
    # Shutdown
    print("Shutting down Power BI RAG Assistant API")


# Create FastAPI app
app = FastAPI(
    title="Power BI RAG Assistant API",
    description="API for answering questions about Power BI reports using RAG",
    version="0.1.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models
class QuestionRequest(BaseModel):
    """Request model for asking questions."""
    question: str = Field(..., description="Question about the Power BI report")
    session_id: Optional[str] = Field(None, description="Session ID for conversation tracking")
    filter_by_type: Optional[str] = Field(None, description="Filter results by artifact type")
    use_conversation_history: bool = Field(True, description="Use conversation history for context")


class QuestionResponse(BaseModel):
    """Response model for question answers."""
    answer: str
    context: List[Dict]
    sources: List[str]
    confidence: float
    cached: bool = False


class UploadResponse(BaseModel):
    """Response model for file uploads."""
    message: str
    artifacts_count: int
    file_name: str
    processing_time: float


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    version: str
    components: Dict[str, str]
    stats: Dict


# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    
    # Check component status
    components = {
        "vector_store": "healthy" if vector_store else "not_initialized",
        "rag_pipeline": "healthy" if rag_pipeline else "not_initialized",
        "embedding_processor": "healthy" if embedding_processor else "not_initialized",
        "cache": "healthy" if cache_manager else "not_initialized"
    }
    
    # Get statistics
    stats = {}
    if vector_store:
        stats["vector_store"] = vector_store.get_collection_info()
    if cache_manager:
        stats["cache"] = cache_manager.get_stats()
    
    return HealthResponse(
        status="healthy" if all(status == "healthy" for status in components.values()) else "degraded",
        version="0.1.0",
        components=components,
        stats=stats
    )


# Question answering endpoint
@app.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """Answer a question about Power BI reports."""
    
    if not rag_pipeline:
        raise HTTPException(status_code=500, detail="RAG pipeline not initialized")
    
    try:
        # Check cache first if enabled
        cached_response = None
        if settings.enable_caching and cache_manager.response_cache:
            # For conversation context, we'll skip cache to maintain conversation flow
            if not request.use_conversation_history:
                context_results = await rag_pipeline._retrieve_context(
                    request.question,
                    filter_by_type=request.filter_by_type
                )
                cached_response = cache_manager.response_cache.get_response(
                    request.question,
                    context_results,
                    rag_pipeline.llm_model,
                    rag_pipeline.temperature
                )
        
        if cached_response:
            return QuestionResponse(**cached_response, cached=True)
        
        # Generate new response
        if request.use_conversation_history and conversation_manager:
            response = await conversation_manager.ask_question(
                request.question,
                session_id=request.session_id,
                use_history=True,
                filter_by_type=request.filter_by_type,
            )
        else:
            response = await rag_pipeline.answer_question(
                request.question,
                filter_by_type=request.filter_by_type
            )
        
        # Cache the response
        if settings.enable_caching and cache_manager.response_cache and not request.use_conversation_history:
            context_results = await rag_pipeline._retrieve_context(
                request.question,
                filter_by_type=request.filter_by_type
            )
            cache_manager.response_cache.set_response(
                request.question,
                context_results,
                rag_pipeline.llm_model,
                rag_pipeline.temperature,
                response
            )
        
        return QuestionResponse(**response, cached=False)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")


# File upload endpoint
@app.post("/upload", response_model=UploadResponse)
async def upload_pbix_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """Upload and process a PBIX file."""
    
    if not file.filename.lower().endswith('.pbix'):
        raise HTTPException(status_code=400, detail="File must be a PBIX file")
    
    # Save uploaded file
    upload_dir = Path("data/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    file_path = upload_dir / file.filename
    
    try:
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Process file in background
        background_tasks.add_task(
            process_pbix_file,
            file_path,
            file.filename
        )
        
        return UploadResponse(
            message=f"File {file.filename} uploaded successfully and is being processed",
            artifacts_count=0,  # Will be updated after processing
            file_name=file.filename,
            processing_time=0.0
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading file: {str(e)}")


async def process_pbix_file(file_path: Path, file_name: str):
    """Process uploaded PBIX file (background task)."""
    
    start_time = time.time()
    
    try:
        # Extract PBIX content
        extractor = PBIXExtractor()
        report = extractor.extract_report(file_path)
        artifacts = extractor.extract_artifacts(report)
        
        # Index artifacts for retrieval.
        if rag_pipeline:
            rag_pipeline.index_artifacts(artifacts)
        elif vector_store:
            batch_size = 50
            for i in range(0, len(artifacts), batch_size):
                batch = artifacts[i:i + batch_size]
                vector_store.add_artifacts(batch)
        
        processing_time = time.time() - start_time
        
        print(f"Processed {file_name}: {len(artifacts)} artifacts in {processing_time:.2f}s")
        
        # Clean up uploaded file
        file_path.unlink(missing_ok=True)
        
    except Exception as e:
        print(f"Error processing {file_name}: {e}")
        # Clean up on error
        file_path.unlink(missing_ok=True)


# Vector store management endpoints
@app.get("/vector-store/info")
async def get_vector_store_info():
    """Get vector store information."""
    
    if not vector_store:
        raise HTTPException(status_code=500, detail="Vector store not initialized")
    
    return vector_store.get_collection_info()


@app.post("/vector-store/search")
async def search_vector_store(
    query: str,
    n_results: int = 5,
    artifact_type: Optional[str] = None
):
    """Search the vector store directly."""
    
    if not vector_store:
        raise HTTPException(status_code=500, detail="Vector store not initialized")
    
    try:
        if artifact_type:
            results = vector_store.search_by_type(query, artifact_type, n_results)
        else:
            results = vector_store.search(query, n_results)
        
        return {"results": results, "query": query, "count": len(results)}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching vector store: {str(e)}")


@app.delete("/vector-store/reset")
async def reset_vector_store():
    """Reset the vector store (clear all data)."""
    
    if not vector_store:
        raise HTTPException(status_code=500, detail="Vector store not initialized")
    
    try:
        vector_store.reset_collection()
        return {"message": "Vector store reset successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error resetting vector store: {str(e)}")


# Cache management endpoints
@app.get("/cache/stats")
async def get_cache_stats():
    """Get cache statistics."""
    return cache_manager.get_stats()


@app.delete("/cache/clear")
async def clear_cache():
    """Clear all caches."""
    
    try:
        cache_manager.clear_all_caches()
        return {"message": "All caches cleared successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing cache: {str(e)}")


@app.post("/cache/cleanup")
async def cleanup_cache():
    """Clean up expired cache entries."""
    
    try:
        cache_manager.cleanup_expired()
        return {"message": "Expired cache entries cleaned up"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error cleaning up cache: {str(e)}")


# Conversation management endpoints
@app.get("/conversation/history/{session_id}")
async def get_conversation_history(session_id: str):
    """Get conversation history for a session."""
    
    if not conversation_manager:
        raise HTTPException(status_code=500, detail="Conversation manager not initialized")
    
    history = conversation_manager.get_conversation_history(session_id)
    return {"session_id": session_id, "history": history, "count": len(history)}


@app.delete("/conversation/history/{session_id}")
async def clear_conversation_history(session_id: str):
    """Clear conversation history for a session."""
    
    if not conversation_manager:
        raise HTTPException(status_code=500, detail="Conversation manager not initialized")
    
    conversation_manager.clear_history(session_id)
    return {"message": f"Conversation history cleared for session {session_id}"}


# Specialized endpoints for specific Power BI queries
@app.post("/explain/measure")
async def explain_measure(measure_name: str):
    """Explain a specific DAX measure."""
    
    if not rag_pipeline:
        raise HTTPException(status_code=500, detail="RAG pipeline not initialized")
    
    try:
        response = await rag_pipeline.explain_measure(measure_name)
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error explaining measure: {str(e)}")


@app.post("/describe/table")
async def describe_table(table_name: str):
    """Describe a specific table."""
    
    if not rag_pipeline:
        raise HTTPException(status_code=500, detail="RAG pipeline not initialized")
    
    try:
        response = await rag_pipeline.describe_table(table_name)
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error describing table: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(
        "powerbi_rag.api.main:app",
        host=settings.api.host,
        port=settings.api.port,
        reload=settings.is_development,
        log_level=settings.log_level.lower()
    )


def _import_symbol(module_name: str, symbol_name: str) -> Any:
    """Import a symbol lazily so the API can start without optional dependencies."""
    module = importlib.import_module(module_name)
    return getattr(module, symbol_name)
