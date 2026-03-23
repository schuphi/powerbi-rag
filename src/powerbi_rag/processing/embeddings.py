"""Embedding generation and processing pipeline."""

import asyncio
import time
from typing import List, Optional, Tuple

from ..extraction.models import PowerBIArtifact
from ..utils.config import settings


class EmbeddingProcessor:
    """Process and generate embeddings for Power BI artifacts."""
    
    def __init__(
        self,
        provider: str = "openai",
        model_name: Optional[str] = None,
        batch_size: int = 100,
        rate_limit_delay: float = 0.1
    ):
        """Initialize embedding processor."""
        self.provider = provider.lower()
        self.model_name = model_name or settings.embedding_model
        self.batch_size = batch_size
        self.rate_limit_delay = rate_limit_delay
        
        if self.provider == "openai":
            if not settings.openai_api_key:
                raise ValueError("OpenAI API key is required for OpenAI embeddings")
            try:
                import openai
                from openai import AsyncOpenAI
            except ImportError as exc:
                raise ImportError(
                    "openai is required to use OpenAI embeddings. Install project dependencies first."
                ) from exc
            self.client = AsyncOpenAI(api_key=settings.openai_api_key)
            self.sync_client = openai.OpenAI(api_key=settings.openai_api_key)
        elif self.provider == "sentence_transformers":
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ImportError(
                    "sentence-transformers is required to use local embeddings. Install project dependencies first."
                ) from exc
            self.model = SentenceTransformer(self.model_name)
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    async def generate_embeddings_async(
        self,
        texts: List[str],
        show_progress: bool = True
    ) -> List[List[float]]:
        """Generate embeddings asynchronously."""
        if self.provider == "openai":
            return await self._generate_openai_embeddings_async(texts, show_progress)
        else:
            # For sentence transformers, use sync method in thread
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,
                self._generate_sentence_transformer_embeddings,
                texts
            )
    
    def generate_embeddings_sync(
        self,
        texts: List[str],
        show_progress: bool = True
    ) -> List[List[float]]:
        """Generate embeddings synchronously."""
        if self.provider == "openai":
            return self._generate_openai_embeddings_sync(texts, show_progress)
        else:
            return self._generate_sentence_transformer_embeddings(texts)
    
    async def _generate_openai_embeddings_async(
        self,
        texts: List[str],
        show_progress: bool = True
    ) -> List[List[float]]:
        """Generate OpenAI embeddings asynchronously."""
        all_embeddings = []
        
        # Process in batches to respect rate limits
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            
            try:
                response = await self.client.embeddings.create(
                    model=self.model_name,
                    input=batch,
                    encoding_format="float"
                )
                
                batch_embeddings = [data.embedding for data in response.data]
                all_embeddings.extend(batch_embeddings)
                
                if show_progress and len(texts) > self.batch_size:
                    print(f"Processed {min(i + self.batch_size, len(texts))}/{len(texts)} embeddings")
                
                # Rate limiting
                if i + self.batch_size < len(texts):
                    await asyncio.sleep(self.rate_limit_delay)
                    
            except Exception as e:
                print(f"Error generating embeddings for batch {i}: {e}")
                # Add empty embeddings as placeholders
                batch_embeddings = [[0.0] * 1536] * len(batch)  # Default OpenAI embedding size
                all_embeddings.extend(batch_embeddings)
        
        return all_embeddings
    
    def _generate_openai_embeddings_sync(
        self,
        texts: List[str],
        show_progress: bool = True
    ) -> List[List[float]]:
        """Generate OpenAI embeddings synchronously."""
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            
            try:
                response = self.sync_client.embeddings.create(
                    model=self.model_name,
                    input=batch,
                    encoding_format="float"
                )
                
                batch_embeddings = [data.embedding for data in response.data]
                all_embeddings.extend(batch_embeddings)
                
                if show_progress and len(texts) > self.batch_size:
                    print(f"Processed {min(i + self.batch_size, len(texts))}/{len(texts)} embeddings")
                
                # Rate limiting
                if i + self.batch_size < len(texts):
                    time.sleep(self.rate_limit_delay)
                    
            except Exception as e:
                print(f"Error generating embeddings for batch {i}: {e}")
                # Add empty embeddings as placeholders
                batch_embeddings = [[0.0] * 1536] * len(batch)
                all_embeddings.extend(batch_embeddings)
        
        return all_embeddings
    
    def _generate_sentence_transformer_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate sentence transformer embeddings."""
        try:
            embeddings = self.model.encode(texts, convert_to_tensor=False)
            return embeddings.tolist()
        except Exception as e:
            print(f"Error generating sentence transformer embeddings: {e}")
            # Return zero embeddings as fallback
            embedding_size = self.model.get_sentence_embedding_dimension()
            return [[0.0] * embedding_size] * len(texts)
    
    async def embed_artifacts_async(
        self,
        artifacts: List[PowerBIArtifact],
        content_field: str = "content"
    ) -> List[Tuple[PowerBIArtifact, List[float]]]:
        """Generate embeddings for artifacts asynchronously."""
        texts = [getattr(artifact, content_field) for artifact in artifacts]
        embeddings = await self.generate_embeddings_async(texts)
        
        return list(zip(artifacts, embeddings))
    
    def embed_artifacts_sync(
        self,
        artifacts: List[PowerBIArtifact],
        content_field: str = "content"
    ) -> List[Tuple[PowerBIArtifact, List[float]]]:
        """Generate embeddings for artifacts synchronously."""
        texts = [getattr(artifact, content_field) for artifact in artifacts]
        embeddings = self.generate_embeddings_sync(texts)
        
        return list(zip(artifacts, embeddings))
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this processor."""
        if self.provider == "openai":
            # Common OpenAI embedding dimensions
            if "text-embedding-3-small" in self.model_name:
                return 1536
            elif "text-embedding-3-large" in self.model_name:
                return 3072
            elif "text-embedding-ada-002" in self.model_name:
                return 1536
            else:
                return 1536  # Default
        else:
            return self.model.get_sentence_embedding_dimension()
    
    def estimate_cost(self, num_tokens: int) -> float:
        """Estimate the cost of embedding generation."""
        if self.provider == "openai":
            # OpenAI pricing (as of 2024)
            if "text-embedding-3-small" in self.model_name:
                cost_per_1k_tokens = 0.00002
            elif "text-embedding-3-large" in self.model_name:
                cost_per_1k_tokens = 0.00013
            elif "text-embedding-ada-002" in self.model_name:
                cost_per_1k_tokens = 0.0001
            else:
                cost_per_1k_tokens = 0.0001  # Default
            
            return (num_tokens / 1000) * cost_per_1k_tokens
        else:
            return 0.0  # Local models are free after initial setup
    
    def estimate_tokens(self, text: str) -> int:
        """Rough estimation of token count for text."""
        # Simple approximation: ~4 characters per token
        return len(text) // 4


class TextChunker:
    """Chunk long text content for embedding processing."""
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separator: str = "\n"
    ):
        """Initialize text chunker."""
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separator = separator
    
    def chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks."""
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            if end >= len(text):
                # Last chunk
                chunks.append(text[start:])
                break
            
            # Try to break at separator near the end
            separator_pos = text.rfind(self.separator, start, end)
            if separator_pos > start:
                end = separator_pos
            
            chunks.append(text[start:end])
            start = end - self.chunk_overlap
        
        return chunks
    
    def chunk_artifacts(
        self,
        artifacts: List[PowerBIArtifact]
    ) -> List[Tuple[PowerBIArtifact, str, int]]:
        """Chunk artifacts and return (artifact, chunk, chunk_index) tuples."""
        chunked_data = []
        
        for artifact in artifacts:
            chunks = self.chunk_text(artifact.content)
            
            if len(chunks) == 1:
                # No chunking needed
                chunked_data.append((artifact, chunks[0], 0))
            else:
                # Multiple chunks
                for i, chunk in enumerate(chunks):
                    chunked_data.append((artifact, chunk, i))
        
        return chunked_data
