"""Vector store implementation using ChromaDB."""

import json
from pathlib import Path
from typing import Dict, List, Optional

from ..extraction.models import PowerBIArtifact
from ..utils.config import settings


class ChromaVectorStore:
    """ChromaDB-based vector store for Power BI artifacts."""
    
    def __init__(
        self,
        persist_directory: Optional[str] = None,
        collection_name: str = "powerbi_artifacts",
        embedding_function: Optional[str] = None
    ):
        """Initialize ChromaDB vector store."""
        try:
            import chromadb
            from chromadb.config import Settings
            from chromadb.utils import embedding_functions
        except ImportError as exc:
            raise ImportError(
                "chromadb is required to use the vector store. Install project dependencies first."
            ) from exc

        self.persist_directory = persist_directory or settings.database.vector_db_path
        self.collection_name = collection_name
        
        # Ensure directory exists
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Set up embedding function
        if embedding_function == "openai" and settings.openai_api_key:
            self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
                api_key=settings.openai_api_key,
                model_name=settings.embedding_model
            )
        else:
            # Fallback to sentence transformers
            self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function
            )
        except ValueError:
            self.collection = self.client.create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function,
                metadata={"hnsw:space": "cosine"}
            )
    
    def add_artifacts(self, artifacts: List[PowerBIArtifact]) -> None:
        """Add Power BI artifacts to the vector store."""
        if not artifacts:
            return
        
        documents = []
        metadatas = []
        ids = []
        
        for artifact in artifacts:
            # Use artifact content as document text
            documents.append(artifact.content)
            
            # Prepare metadata (ChromaDB requires string values)
            metadata = {
                "type": artifact.type,
                "name": artifact.name,
                "source_file": artifact.source_file or "",
                "parent_id": artifact.parent_id or "",
                "tags": json.dumps(artifact.tags)
            }
            
            # Add additional metadata as strings
            for key, value in artifact.metadata.items():
                if value is not None:
                    metadata[f"meta_{key}"] = str(value)
            
            metadatas.append(metadata)
            ids.append(artifact.id)
        
        # Add to collection in batch
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
    
    def search(
        self,
        query: str,
        n_results: int = 5,
        filter_dict: Optional[Dict] = None,
        include_metadata: bool = True
    ) -> List[Dict]:
        """Search for similar artifacts."""
        
        # Perform similarity search
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=filter_dict,
            include=["documents", "metadatas", "distances"] if include_metadata else ["documents", "distances"]
        )
        
        # Format results
        formatted_results = []
        if results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                result = {
                    "id": doc_id,
                    "content": results["documents"][0][i],
                    "distance": results["distances"][0][i],
                    "score": 1 - results["distances"][0][i]  # Convert distance to similarity score
                }
                
                if include_metadata and results["metadatas"][0][i]:
                    metadata = results["metadatas"][0][i]
                    
                    # Parse tags back from JSON
                    if "tags" in metadata:
                        try:
                            metadata["tags"] = json.loads(metadata["tags"])
                        except (json.JSONDecodeError, TypeError):
                            metadata["tags"] = []
                    
                    result["metadata"] = metadata
                
                formatted_results.append(result)
        
        return formatted_results
    
    def search_by_type(
        self,
        query: str,
        artifact_type: str,
        n_results: int = 5
    ) -> List[Dict]:
        """Search for artifacts of a specific type."""
        filter_dict = {"type": {"$eq": artifact_type}}
        return self.search(query, n_results, filter_dict)
    
    def search_by_tags(
        self,
        query: str,
        tags: List[str],
        n_results: int = 5
    ) -> List[Dict]:
        """Search for artifacts with specific tags."""
        # Note: ChromaDB doesn't support array filtering directly,
        # so we'll do post-filtering on the results
        results = self.search(query, n_results * 2)  # Get more results to filter
        
        filtered_results = []
        for result in results:
            if "metadata" in result and "tags" in result["metadata"]:
                result_tags = result["metadata"]["tags"]
                if any(tag in result_tags for tag in tags):
                    filtered_results.append(result)
                    if len(filtered_results) >= n_results:
                        break
        
        return filtered_results[:n_results]
    
    def get_artifact_by_id(self, artifact_id: str) -> Optional[Dict]:
        """Retrieve artifact by ID."""
        try:
            result = self.collection.get(
                ids=[artifact_id],
                include=["documents", "metadatas"]
            )
            
            if result["ids"] and result["ids"][0]:
                metadata = result["metadatas"][0] if result["metadatas"] else {}
                
                # Parse tags back from JSON
                if "tags" in metadata:
                    try:
                        metadata["tags"] = json.loads(metadata["tags"])
                    except (json.JSONDecodeError, TypeError):
                        metadata["tags"] = []
                
                return {
                    "id": result["ids"][0],
                    "content": result["documents"][0],
                    "metadata": metadata
                }
        except Exception:
            pass
        
        return None

    def list_artifacts(self, filter_dict: Optional[Dict] = None) -> List[Dict]:
        """List stored artifacts in retrieval-record format."""
        try:
            result = self.collection.get(
                where=filter_dict,
                include=["documents", "metadatas"],
            )
        except Exception:
            return []

        records = []
        ids = result.get("ids") or []
        documents = result.get("documents") or []
        metadatas = result.get("metadatas") or []

        for index, artifact_id in enumerate(ids):
            metadata = metadatas[index] if index < len(metadatas) and metadatas[index] else {}
            if "tags" in metadata:
                try:
                    metadata["tags"] = json.loads(metadata["tags"])
                except (json.JSONDecodeError, TypeError):
                    metadata["tags"] = []

            records.append(
                {
                    "id": artifact_id,
                    "content": documents[index] if index < len(documents) else "",
                    "metadata": metadata,
                }
            )

        return records
    
    def update_artifact(self, artifact: PowerBIArtifact) -> None:
        """Update an existing artifact."""
        # ChromaDB doesn't have direct update, so we delete and re-add
        self.delete_artifact(artifact.id)
        self.add_artifacts([artifact])
    
    def delete_artifact(self, artifact_id: str) -> None:
        """Delete an artifact by ID."""
        try:
            self.collection.delete(ids=[artifact_id])
        except Exception:
            pass
    
    def delete_by_source_file(self, source_file: str) -> None:
        """Delete all artifacts from a specific source file."""
        try:
            self.collection.delete(where={"source_file": {"$eq": source_file}})
        except Exception:
            pass
    
    def count_artifacts(self) -> int:
        """Get total number of artifacts in the store."""
        return self.collection.count()
    
    def list_artifact_types(self) -> List[str]:
        """List all unique artifact types in the store."""
        # Get all metadata
        all_data = self.collection.get(include=["metadatas"])
        types = set()
        
        if all_data["metadatas"]:
            for metadata in all_data["metadatas"]:
                if "type" in metadata:
                    types.add(metadata["type"])
        
        return sorted(list(types))
    
    def reset_collection(self) -> None:
        """Clear all data from the collection."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"}
        )
    
    def get_collection_info(self) -> Dict:
        """Get information about the collection."""
        return {
            "name": self.collection_name,
            "count": self.count_artifacts(),
            "persist_directory": self.persist_directory,
            "embedding_function": type(self.embedding_function).__name__,
            "artifact_types": self.list_artifact_types()
        }
