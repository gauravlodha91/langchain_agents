# updated code from app.py where we can search based on semantic search.  RedisVectorManager Redis-Stack


from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
import redis
import json
import hashlib
import os
import sys
from pathlib import Path
import asyncio
from datetime import datetime, timedelta
import logging
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import tiktoken
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Document Chunking API with Vector Search", version="2.0.0")

# Pydantic models
class ChunkRequest(BaseModel):
    file_path: str = Field(..., description="Local path to the document file")
    chunk_size: int = Field(default=1000, description="Maximum tokens per chunk")
    overlap: int = Field(default=200, description="Overlap between chunks in tokens")
    force_refresh: bool = Field(default=False, description="Force refresh cached chunks")

class QueryRequest(BaseModel):
    query: str = Field(..., description="Query to search for relevant chunks")
    file_path: str = Field(..., description="File path to search in")
    top_k: int = Field(default=5, description="Number of top relevant chunks to return")
    similarity_threshold: float = Field(default=0.3, description="Minimum similarity threshold")
    use_vector_search: bool = Field(default=True, description="Use Redis vector search instead of local similarity")

class ChunkResponse(BaseModel):
    chunk_id: str
    content: str
    metadata: Dict[str, Any]
    similarity_score: Optional[float] = None

class ChunkingStats(BaseModel):
    total_chunks: int
    total_tokens: int
    average_chunk_size: int
    file_size_bytes: int
    processing_time_seconds: float

# Redis Vector Search Manager
class RedisVectorManager:
    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0):
        self.redis_client = redis.Redis(
            host=host, 
            port=port, 
            db=db, 
            decode_responses=False,  # Keep as False for vector operations
            socket_connect_timeout=5,
            socket_timeout=5
        )
        self.default_ttl = 86400  # 24 hours
        self.vector_dim = 384  # Dimension for all-MiniLM-L6-v2 model
        
    def get_connection_info(self):
        try:
            info = self.redis_client.info()
            return {
                "connected": True,
                "redis_version": info.get('redis_version'),
                "used_memory_human": info.get('used_memory_human'),
                "connected_clients": info.get('connected_clients')
            }
        except Exception as e:
            return {"connected": False, "error": str(e)}
    
    def create_vector_index(self, index_name: str, vector_dim: int = 384):
        """Create a vector search index in Redis"""
        try:
            # Check if index already exists
            try:
                self.redis_client.execute_command("FT.INFO", index_name)
                logger.info(f"Vector index '{index_name}' already exists")
                return True
            except:
                pass  # Index doesn't exist, create it
            
            # Create vector index
            schema = [
                "content", "TEXT", "WEIGHT", "1.0",
                "file_path", "TEXT",
                "chunk_id", "TEXT",
                "tokens", "NUMERIC",
                "chunk_index", "NUMERIC",
                "created_at", "TEXT",
                "embedding", "VECTOR", "FLAT", "6", 
                "TYPE", "FLOAT32", 
                "DIM", str(vector_dim), 
                "DISTANCE_METRIC", "COSINE"
            ]
            
            result = self.redis_client.execute_command(
                "FT.CREATE", index_name,
                "ON", "HASH",
                "PREFIX", "1", "doc:",
                "SCHEMA", *schema
            )
            
            logger.info(f"Created vector index '{index_name}': {result}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating vector index: {str(e)}")
            return False
    
    def store_chunk_with_vector(self, key: str, chunk_data: Dict, embedding: np.ndarray):
        """Store chunk data with vector embedding in Redis"""
        try:
            # Convert embedding to bytes
            embedding_bytes = embedding.astype(np.float32).tobytes()
            
            
            
            """
            {
                "content": "...chunk text...",
                "file_path": "...",
                "chunk_id": "...",
                "tokens": "...",
                "chunk_index": "...",
                "created_at": "...",
                "file_hash": "...",
                "embedding": <vector bytes>  # This is the vector representation of content column
            }
            
            """
            # Prepare hash data
            hash_data = {
                "content": chunk_data["content"],
                "file_path": chunk_data["file_path"],
                "chunk_id": chunk_data["chunk_id"],
                "tokens": str(chunk_data["tokens"]),
                "chunk_index": str(chunk_data.get("chunk_index", 0)),
                "created_at": chunk_data.get("created_at", ""),
                "file_hash": chunk_data.get("file_hash", ""),
                "embedding": embedding_bytes
            }
            
            # Store as hash
            self.redis_client.hset(key, mapping=hash_data)
            
            # Set TTL
            self.redis_client.expire(key, self.default_ttl)
            
            return True
            
        except Exception as e:
            logger.error(f"Error storing chunk with vector: {str(e)}")
            return False
    
    def vector_search(self, query_embedding: np.ndarray, file_hash: str = None, 
                     top_k: int = 5, index_name: str = "chunk_index") -> List[Dict]:
        """Perform vector similarity search"""
        try:
            # Convert query embedding to bytes
            query_bytes = query_embedding.astype(np.float32).tobytes()
            
            # Build search query
            if file_hash:
                # Search within specific file
                query_filter = f"@file_hash:{file_hash}"
                search_query = f"({query_filter})=>[KNN {top_k} @embedding $BLOB AS score]"
            else:
                # Search across all documents
                search_query = f"*=>[KNN {top_k} @embedding $BLOB AS score]"
            
            # Execute search
            result = self.redis_client.execute_command(
                "FT.SEARCH", index_name, search_query,
                "PARAMS", "2", "BLOB", query_bytes,
                "SORTBY", "score", "ASC",
                "LIMIT", "0", str(top_k),
                "RETURN", "7", "content", "chunk_id", "file_path", "tokens", "chunk_index", "created_at", "score"
            )
            
            # Parse results
            chunks = []
            if len(result) > 1:
                num_results = result[0]
                for i in range(1, len(result), 2):
                    doc_id = result[i].decode('utf-8')
                    fields = result[i + 1]
                    
                    # Parse field-value pairs
                    chunk_data = {}
                    for j in range(0, len(fields), 2):
                        field_name = fields[j].decode('utf-8')
                        field_value = fields[j + 1]
                        
                        if isinstance(field_value, bytes):
                            try:
                                chunk_data[field_name] = field_value.decode('utf-8')
                            except:
                                chunk_data[field_name] = str(field_value)
                        else:
                            chunk_data[field_name] = field_value
                    
                    # Calculate similarity score (Redis returns distance, convert to similarity)
                    distance = float(chunk_data.get('score', 1.0))
                    similarity = 1.0 - distance  # Convert cosine distance to similarity
                    
                    chunks.append({
                        "chunk_id": chunk_data.get("chunk_id", ""),
                        "content": chunk_data.get("content", ""),
                        "file_path": chunk_data.get("file_path", ""),
                        "tokens": int(chunk_data.get("tokens", 0)),
                        "chunk_index": int(chunk_data.get("chunk_index", 0)),
                        "created_at": chunk_data.get("created_at", ""),
                        "similarity_score": similarity
                    })
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error in vector search: {str(e)}")
            return []
    
    def get_chunks_by_pattern(self, pattern: str) -> List[Dict]:
        """Get chunks by key pattern (fallback method)"""
        try:
            keys = self.redis_client.keys(pattern)
            chunks = []
            
            for key in keys:
                hash_data = self.redis_client.hgetall(key)
                if hash_data:
                    # Convert bytes to strings and parse
                    chunk_data = {}
                    for field, value in hash_data.items():
                        field_name = field.decode('utf-8') if isinstance(field, bytes) else field
                        
                        if field_name == 'embedding':
                            continue  # Skip embedding data in response
                        
                        field_value = value.decode('utf-8') if isinstance(value, bytes) else value
                        
                        # Convert numeric fields
                        if field_name in ['tokens', 'chunk_index']:
                            try:
                                chunk_data[field_name] = int(field_value)
                            except:
                                chunk_data[field_name] = field_value
                        else:
                            chunk_data[field_name] = field_value
                    
                    chunks.append(chunk_data)
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error getting chunks by pattern: {str(e)}")
            return []
    
    def delete_chunks_by_pattern(self, pattern: str) -> int:
        """Delete chunks by pattern"""
        try:
            keys = self.redis_client.keys(pattern)
            if keys:
                return self.redis_client.delete(*keys)
            return 0
        except Exception as e:
            logger.error(f"Error deleting chunks: {str(e)}")
            return 0
    
    def delete_index(self, index_name: str):
        """Delete a search index"""
        try:
            result = self.redis_client.execute_command("FT.DROPINDEX", index_name)
            logger.info(f"Deleted index '{index_name}': {result}")
            return True
        except Exception as e:
            logger.error(f"Error deleting index: {str(e)}")
            return False

# Document chunking service (same as before)
class DocumentChunker:
    def __init__(self):
        self.encoding = tiktoken.get_encoding("cl100k_base")  # GPT-4 encoding
        
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken"""
        return len(self.encoding.encode(text))
    
    def read_file(self, file_path: str) -> str:
        """Read file content with proper encoding handling"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        encodings = ['utf-8', 'utf-16', 'latin-1', 'cp1252']
        content = None
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as file:
                    content = file.read()
                break
            except UnicodeDecodeError:
                continue
        
        if content is None:
            raise ValueError(f"Could not decode file: {file_path}")
        
        return content
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', ' ', text)
        return text.strip()
    
    def create_chunks(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[Dict]:
        """Create overlapping chunks from text"""
        text = self.clean_text(text)
        sentences = self._split_into_sentences(text)
        
        chunks = []
        current_chunk = ""
        current_tokens = 0
        chunk_index = 0
        
        for sentence in sentences:
            sentence_tokens = self.count_tokens(sentence)
            
            # If adding this sentence exceeds chunk size, finalize current chunk
            if current_tokens + sentence_tokens > chunk_size and current_chunk:
                chunk_data = {
                    "chunk_id": f"chunk_{chunk_index}",
                    "content": current_chunk.strip(),
                    "tokens": current_tokens,
                    "start_sentence": chunk_index * (chunk_size - overlap) // 50,  # Approximate
                    "created_at": datetime.utcnow().isoformat()
                }
                chunks.append(chunk_data)
                
                # Start new chunk with overlap
                overlap_text = self._get_overlap_text(current_chunk, overlap)
                current_chunk = overlap_text + " " + sentence
                current_tokens = self.count_tokens(current_chunk)
                chunk_index += 1
            else:
                current_chunk += " " + sentence if current_chunk else sentence
                current_tokens += sentence_tokens
        
        # Add final chunk
        if current_chunk.strip():
            chunk_data = {
                "chunk_id": f"chunk_{chunk_index}",
                "content": current_chunk.strip(),
                "tokens": current_tokens,
                "start_sentence": chunk_index * (chunk_size - overlap) // 50,
                "created_at": datetime.utcnow().isoformat()
            }
            chunks.append(chunk_data)
        
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitting - can be enhanced with NLTK or spaCy
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _get_overlap_text(self, text: str, overlap_tokens: int) -> str:
        """Get the last overlap_tokens worth of text"""
        words = text.split()
        if len(words) <= overlap_tokens // 4:  # Rough estimation
            return text
        
        overlap_words = words[-(overlap_tokens // 4):]
        return " ".join(overlap_words)

# Enhanced Similarity searcher
class SimilaritySearcher:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        logger.info(f"Loaded embedding model: {model_name}")
    
    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text"""
        return self.model.encode([text])[0]
    
    def find_similar_chunks(self, query: str, chunks: List[Dict], 
                          top_k: int = 5, threshold: float = 0.3) -> List[Dict]:
        """Find most similar chunks to query (fallback method)"""
        if not chunks:
            return []
        
        query_embedding = self.get_embedding(query)
        chunk_embeddings = []
        
        # Get embeddings for all chunks
        for chunk in chunks:
            chunk_embedding = self.get_embedding(chunk['content'])
            chunk_embeddings.append(chunk_embedding)
        
        # Calculate similarities
        similarities = cosine_similarity([query_embedding], chunk_embeddings)[0]
        
        # Create results with similarity scores
        results = []
        for i, (chunk, similarity) in enumerate(zip(chunks, similarities)):
            if similarity >= threshold:
                chunk_with_score = chunk.copy()
                chunk_with_score['similarity_score'] = float(similarity)
                results.append(chunk_with_score)
        
        # Sort by similarity and return top_k
        results.sort(key=lambda x: x['similarity_score'], reverse=True)
        return results[:top_k]

# Initialize services
redis_vector_manager = RedisVectorManager()
document_chunker = DocumentChunker()
similarity_searcher = SimilaritySearcher()

# Create vector index on startup
INDEX_NAME = "chunk_index"
redis_vector_manager.create_vector_index(INDEX_NAME, vector_dim=384)

def generate_file_hash(file_path: str) -> str:
    """Generate hash for file path and modification time"""
    stat = os.stat(file_path)
    content = f"{file_path}:{stat.st_mtime}:{stat.st_size}"
    return hashlib.md5(content.encode()).hexdigest()

@app.get("/")
async def root():
    return {"message": "Document Chunking API with Vector Search", "version": "2.0.0"}

@app.get("/health")
async def health_check():
    redis_info = redis_vector_manager.get_connection_info()
    return {
        "status": "healthy",
        "redis": redis_info,
        "vector_search_enabled": True,
        "timestamp": datetime.utcnow().isoformat()
    }

@app.post("/chunk-document", response_model=ChunkingStats)
async def chunk_document(request: ChunkRequest, background_tasks: BackgroundTasks):
    """Chunk a document and store chunks with vectors in Redis"""
    try:
        start_time = datetime.utcnow()
        
        # Validate file exists
        if not os.path.exists(request.file_path):
            raise HTTPException(status_code=404, detail=f"File not found: {request.file_path}")
        
        file_hash = generate_file_hash(request.file_path)
        cache_pattern = f"doc:{file_hash}:*"
        
        # Check if chunks already exist and not forcing refresh
        if not request.force_refresh:
            existing_chunks = redis_vector_manager.get_chunks_by_pattern(cache_pattern)
            if existing_chunks:
                logger.info(f"Found {len(existing_chunks)} existing chunks for {request.file_path}")
                return ChunkingStats(
                    total_chunks=len(existing_chunks),
                    total_tokens=sum(chunk.get('tokens', 0) for chunk in existing_chunks),
                    average_chunk_size=sum(chunk.get('tokens', 0) for chunk in existing_chunks) // len(existing_chunks),
                    file_size_bytes=os.path.getsize(request.file_path),
                    processing_time_seconds=0.0
                )
        
        # Delete existing chunks if force refresh
        if request.force_refresh:
            deleted_count = redis_vector_manager.delete_chunks_by_pattern(cache_pattern)
            logger.info(f"Deleted {deleted_count} existing chunks")
        
        # Read and chunk document
        logger.info(f"Processing file: {request.file_path}")
        content = document_chunker.read_file(request.file_path)
        chunks = document_chunker.create_chunks(
            content, 
            chunk_size=request.chunk_size, 
            overlap=request.overlap
        )
        
        # Store chunks with vectors in Redis
        for i, chunk in enumerate(chunks):
            chunk_key = f"doc:{file_hash}:{i}"
            chunk_data = {
                **chunk,
                "file_path": request.file_path,
                "file_hash": file_hash,
                "chunk_index": i
            }
            
            # Generate embedding
            embedding = similarity_searcher.get_embedding(chunk['content'])
            
            # Store in Redis with vector
            redis_vector_manager.store_chunk_with_vector(chunk_key, chunk_data, embedding)
        
        # Calculate statistics
        total_tokens = sum(chunk['tokens'] for chunk in chunks)
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        logger.info(f"Processed {len(chunks)} chunks with {total_tokens} total tokens and stored with vectors")
        
        return ChunkingStats(
            total_chunks=len(chunks),
            total_tokens=total_tokens,
            average_chunk_size=total_tokens // len(chunks) if chunks else 0,
            file_size_bytes=os.path.getsize(request.file_path),
            processing_time_seconds=processing_time
        )
        
    except Exception as e:
        logger.error(f"Error chunking document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search-chunks", response_model=List[ChunkResponse])
async def search_chunks(request: QueryRequest):
    """Search for relevant chunks using vector search or fallback similarity"""
    try:
        # Validate file exists
        if not os.path.exists(request.file_path):
            raise HTTPException(status_code=404, detail=f"File not found: {request.file_path}")
        
        file_hash = generate_file_hash(request.file_path)
        
        # Generate query embedding
        query_embedding = similarity_searcher.get_embedding(request.query)
        
        similar_chunks = []
        
        if request.use_vector_search:
            # Use Redis vector search
            logger.info(f"Using Redis vector search for query: {request.query}")
            similar_chunks = redis_vector_manager.vector_search(
                query_embedding,
                file_hash=file_hash,
                top_k=request.top_k,
                index_name=INDEX_NAME
            )
            
            # Filter by similarity threshold
            similar_chunks = [
                chunk for chunk in similar_chunks 
                if chunk.get('similarity_score', 0) >= request.similarity_threshold
            ]
        
        # Fallback to local similarity search if vector search fails or returns no results
        if not similar_chunks:
            logger.info("Falling back to local similarity search")
            cache_pattern = f"doc:{file_hash}:*"
            chunks = redis_vector_manager.get_chunks_by_pattern(cache_pattern)
            
            if not chunks:
                raise HTTPException(
                    status_code=404, 
                    detail=f"No chunks found for file: {request.file_path}. Please chunk the document first."
                )
            
            similar_chunks = similarity_searcher.find_similar_chunks(
                request.query,
                chunks,
                top_k=request.top_k,
                threshold=request.similarity_threshold
            )
        
        # Convert to response format
        response_chunks = []
        for chunk in similar_chunks:
            response_chunks.append(ChunkResponse(
                chunk_id=chunk['chunk_id'],
                content=chunk['content'],
                metadata={
                    "tokens": chunk.get('tokens', 0),
                    "chunk_index": chunk.get('chunk_index', 0),
                    "created_at": chunk.get('created_at'),
                    "file_path": chunk.get('file_path', request.file_path)
                },
                similarity_score=chunk.get('similarity_score')
            ))
        
        logger.info(f"Found {len(response_chunks)} relevant chunks")
        return response_chunks
        
    except Exception as e:
        logger.error(f"Error searching chunks: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/chunks/{file_path:path}")
async def get_all_chunks(file_path: str):
    """Get all chunks for a specific file"""
    try:
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
        
        file_hash = generate_file_hash(file_path)
        cache_pattern = f"doc:{file_hash}:*"
        chunks = redis_vector_manager.get_chunks_by_pattern(cache_pattern)
        
        return {
            "file_path": file_path,
            "total_chunks": len(chunks),
            "chunks": chunks
        }
        
    except Exception as e:
        logger.error(f"Error getting chunks: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/chunks/{file_path:path}")
async def delete_chunks(file_path: str):
    """Delete all chunks for a specific file"""
    try:
        file_hash = generate_file_hash(file_path)
        cache_pattern = f"doc:{file_hash}:*"
        deleted_count = redis_vector_manager.delete_chunks_by_pattern(cache_pattern)
        
        return {
            "file_path": file_path,
            "deleted_chunks": deleted_count,
            "message": f"Deleted {deleted_count} chunks"
        }
        
    except Exception as e:
        logger.error(f"Error deleting chunks: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/redis-stats")
async def get_redis_stats():
    """Get Redis statistics and stored chunks info"""
    try:
        info = redis_vector_manager.get_connection_info()
        all_keys = redis_vector_manager.redis_client.keys("doc:*")
        
        # Group by file hash
        file_stats = {}
        for key in all_keys:
            key_str = key.decode('utf-8') if isinstance(key, bytes) else key
            parts = key_str.split(":")
            if len(parts) >= 2:
                file_hash = parts[1]
                if file_hash not in file_stats:
                    file_stats[file_hash] = 0
                file_stats[file_hash] += 1
        
        return {
            "redis_info": info,
            "total_chunk_keys": len(all_keys),
            "files_cached": len(file_stats),
            "chunks_per_file": file_stats,
            "vector_search_enabled": True,
            "vector_index": INDEX_NAME
        }
        
    except Exception as e:
        logger.error(f"Error getting Redis stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/redis-flush")
async def flush_redis():
    """Delete all keys in the Redis database (dangerous operation)."""
    try:
        deleted = redis_vector_manager.redis_client.flushdb()
        # Recreate the vector index after flush
        redis_vector_manager.create_vector_index(INDEX_NAME, vector_dim=384)
        
        return {
            "message": "All Redis data has been deleted and vector index recreated.",
            "result": deleted
        }
    except Exception as e:
        logger.error(f"Error flushing Redis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/recreate-index")
async def recreate_vector_index():
    """Recreate the vector search index"""
    try:
        # Delete existing index
        redis_vector_manager.delete_index(INDEX_NAME)
        
        # Create new index
        success = redis_vector_manager.create_vector_index(INDEX_NAME, vector_dim=384)
        
        return {
            "message": f"Vector index '{INDEX_NAME}' recreated successfully" if success else "Failed to recreate index",
            "success": success
        }
    except Exception as e:
        logger.error(f"Error recreating index: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)