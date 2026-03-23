"""Caching utilities for cost optimization."""

import hashlib
import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

from .config import settings


class _FallbackDiskCache(dict):
    """Minimal diskcache-compatible fallback used when diskcache is unavailable."""

    def volume(self) -> int:
        return 0

    def clear(self):
        super().clear()


class SQLiteCache:
    """SQLite-based cache for storing responses and embeddings."""
    
    def __init__(self, db_path: Optional[str] = None):
        """Initialize SQLite cache."""
        self.db_path = db_path or settings.database.cache_db_path
        
        # Ensure directory exists
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_database()
    
    def _init_database(self):
        """Initialize database tables."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Response cache table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS response_cache (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    expires_at REAL,
                    hit_count INTEGER DEFAULT 0
                )
            """)
            
            # Embedding cache table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS embedding_cache (
                    text_hash TEXT PRIMARY KEY,
                    text_content TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    model_name TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
            """)
            
            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_response_expires ON response_cache(expires_at)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_embedding_model ON embedding_cache(model_name)")
            
            conn.commit()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT value, expires_at FROM response_cache 
                WHERE key = ? AND (expires_at IS NULL OR expires_at > ?)
            """, (key, time.time()))
            
            result = cursor.fetchone()
            if result:
                # Update hit count
                cursor.execute("""
                    UPDATE response_cache 
                    SET hit_count = hit_count + 1 
                    WHERE key = ?
                """, (key,))
                conn.commit()
                
                return json.loads(result[0])
        
        return None
    
    def set(
        self,
        key: str,
        value: Any,
        ttl_seconds: Optional[int] = None
    ):
        """Set value in cache."""
        expires_at = None
        if ttl_seconds:
            expires_at = time.time() + ttl_seconds
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO response_cache 
                (key, value, created_at, expires_at) 
                VALUES (?, ?, ?, ?)
            """, (key, json.dumps(value), time.time(), expires_at))
            
            conn.commit()
    
    def delete(self, key: str):
        """Delete value from cache."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM response_cache WHERE key = ?", (key,))
            conn.commit()
    
    def clear_expired(self):
        """Clear expired entries."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                DELETE FROM response_cache 
                WHERE expires_at IS NOT NULL AND expires_at <= ?
            """, (time.time(),))
            conn.commit()
    
    def get_stats(self) -> Dict:
        """Get cache statistics."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Response cache stats
            cursor.execute("SELECT COUNT(*), SUM(hit_count) FROM response_cache")
            response_count, total_hits = cursor.fetchone()
            
            # Embedding cache stats
            cursor.execute("SELECT COUNT(*) FROM embedding_cache")
            embedding_count = cursor.fetchone()[0]
            
            return {
                "response_entries": response_count or 0,
                "total_hits": total_hits or 0,
                "embedding_entries": embedding_count or 0,
                "db_path": self.db_path
            }


class EmbeddingCache:
    """Specialized cache for embeddings."""
    
    def __init__(self, cache_db: Optional[SQLiteCache] = None):
        """Initialize embedding cache."""
        self.cache_db = cache_db or SQLiteCache()
    
    def _hash_text(self, text: str, model_name: str) -> str:
        """Create hash for text and model combination."""
        content = f"{model_name}:{text}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    def get_embedding(self, text: str, model_name: str) -> Optional[list]:
        """Get cached embedding."""
        text_hash = self._hash_text(text, model_name)
        
        with sqlite3.connect(self.cache_db.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT embedding FROM embedding_cache 
                WHERE text_hash = ? AND model_name = ?
            """, (text_hash, model_name))
            
            result = cursor.fetchone()
            if result:
                return json.loads(result[0])
        
        return None
    
    def set_embedding(
        self,
        text: str,
        embedding: list,
        model_name: str
    ):
        """Cache embedding."""
        text_hash = self._hash_text(text, model_name)
        
        with sqlite3.connect(self.cache_db.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO embedding_cache 
                (text_hash, text_content, embedding, model_name, created_at) 
                VALUES (?, ?, ?, ?, ?)
            """, (
                text_hash,
                text[:1000],  # Store first 1000 chars for reference
                json.dumps(embedding),
                model_name,
                time.time()
            ))
            conn.commit()
    
    def get_cached_embeddings(
        self,
        texts: list,
        model_name: str
    ) -> tuple[list, list]:
        """Get cached embeddings and return (cached_embeddings, missing_texts)."""
        cached_embeddings = {}
        missing_texts = []
        
        for text in texts:
            embedding = self.get_embedding(text, model_name)
            if embedding:
                cached_embeddings[text] = embedding
            else:
                missing_texts.append(text)
        
        return cached_embeddings, missing_texts


class ResponseCache:
    """Cache for LLM responses."""
    
    def __init__(self, cache_db: Optional[SQLiteCache] = None, default_ttl: int = 86400):
        """Initialize response cache."""
        self.cache_db = cache_db or SQLiteCache()
        self.default_ttl = default_ttl  # 24 hours default
    
    def _create_cache_key(
        self,
        question: str,
        context_hash: str,
        model_name: str,
        temperature: float
    ) -> str:
        """Create cache key for response."""
        key_content = f"{model_name}:{temperature}:{question}:{context_hash}"
        return hashlib.sha256(key_content.encode()).hexdigest()
    
    def _hash_context(self, context: list) -> str:
        """Create hash for context results."""
        # Sort context by ID to ensure consistent hashing
        sorted_context = sorted(context, key=lambda x: x.get("id", ""))
        context_str = json.dumps(sorted_context, sort_keys=True)
        return hashlib.sha256(context_str.encode()).hexdigest()[:16]
    
    def get_response(
        self,
        question: str,
        context: list,
        model_name: str,
        temperature: float
    ) -> Optional[Dict]:
        """Get cached response."""
        if not settings.enable_caching:
            return None
        
        context_hash = self._hash_context(context)
        cache_key = self._create_cache_key(question, context_hash, model_name, temperature)
        
        return self.cache_db.get(cache_key)
    
    def set_response(
        self,
        question: str,
        context: list,
        model_name: str,
        temperature: float,
        response: Dict,
        ttl_seconds: Optional[int] = None
    ):
        """Cache response."""
        if not settings.enable_caching:
            return
        
        context_hash = self._hash_context(context)
        cache_key = self._create_cache_key(question, context_hash, model_name, temperature)
        
        ttl = ttl_seconds or (settings.cache_ttl_hours * 3600)
        self.cache_db.set(cache_key, response, ttl)


class CacheManager:
    """Unified cache manager."""
    
    def __init__(self):
        """Initialize cache manager."""
        self.sqlite_cache = SQLiteCache()
        self.embedding_cache = EmbeddingCache(self.sqlite_cache)
        self.response_cache = ResponseCache(self.sqlite_cache)
        
        # Also initialize disk cache for large objects
        cache_dir = Path(settings.database.vector_db_path).parent / "disk_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            import diskcache as dc
            self.disk_cache = dc.Cache(str(cache_dir))
        except ImportError:
            self.disk_cache = _FallbackDiskCache()
    
    def cleanup_expired(self):
        """Clean up expired cache entries."""
        self.sqlite_cache.clear_expired()
    
    def get_stats(self) -> Dict:
        """Get comprehensive cache statistics."""
        sqlite_stats = self.sqlite_cache.get_stats()
        disk_stats = {
            "disk_entries": len(self.disk_cache),
            "disk_size_mb": self.disk_cache.volume() / (1024 * 1024)
        }
        
        return {
            "sqlite": sqlite_stats,
            "disk": disk_stats,
            "caching_enabled": settings.enable_caching,
            "cache_ttl_hours": settings.cache_ttl_hours
        }
    
    def clear_all_caches(self):
        """Clear all caches."""
        # Clear SQLite tables
        with sqlite3.connect(self.sqlite_cache.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM response_cache")
            cursor.execute("DELETE FROM embedding_cache")
            conn.commit()
        
        # Clear disk cache
        self.disk_cache.clear()


# Global cache manager instance
cache_manager = CacheManager()
