import os
import chromadb
from chromadb.utils import embedding_functions
from chromadb.config import Settings
import logging
import cohere

# Add this import:
from chromadb import PersistentClient

logger = logging.getLogger(__name__)

# Define the persist directory for ChromaDB
PERSIST_DIRECTORY = "./chroma_persistent_data"

def initialize_database(cohere_api_key):
    """Initialize ChromaDB with Cohere embedding function and persistence."""
    logger.info("Initializing ChromaDB with Cohere embedding function and persistence")
    
    try:
        # Use the new PersistentClient
        chroma_client = PersistentClient(path=PERSIST_DIRECTORY)
        
        # Set up Cohere embedding function
        cohere_ef = embedding_functions.CohereEmbeddingFunction(
            api_key=cohere_api_key,
            model_name="embed-english-v3.0"
        )
        
        # Get or create collection
        try:
            collection = chroma_client.get_or_create_collection(
                name="youtube_transcripts",
                embedding_function=cohere_ef
            )
            logger.info("Collection initialized successfully")
            return collection, chroma_client
        except Exception as e:
            logger.error("Error initializing collection: %s", e)
            raise e
            
    except Exception as e:
        logger.error("Error initializing ChromaDB: %s", e)
        raise e

def index_chunks(collection, chunks):
    """Index chunks into the ChromaDB collection."""
    try:
        logger.info("Indexing %d chunks into ChromaDB", len(chunks))
        
        # Extract the required components from chunks
        documents = [chunk["text"] for chunk in chunks]
        metadatas = [chunk["metadata"] for chunk in chunks]
        ids = [chunk["id"] for chunk in chunks]
        
        # Use the collection.add method with the correct parameters
        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        
        logger.info("Successfully indexed chunks into ChromaDB")
    except Exception as e:
        logger.error("Error indexing chunks: %s", e)
        raise e

def get_all_chunks(collection):
    """Retrieve all chunks from the vector database."""
    logger.info("Retrieving all chunks from the vector database")
    try:
        all_chunks = collection.get(include=["metadatas", "documents"])
        chunks_json = [
            {
                "id": chunk_id,
                "text": document,
                "metadata": metadata
            }
            for chunk_id, document, metadata in zip(all_chunks["ids"], all_chunks["documents"], all_chunks["metadatas"])
        ]
        logger.info("Retrieved %d chunks from the vector database", len(chunks_json))
        return chunks_json
    except Exception as e:
        logger.error("Error retrieving chunks: %s", e)
        return []

def get_collection_stats(collection):
    """Get statistics about the collection."""
    try:
        if not collection:
            return {"error": "No collection provided"}
        
        stats = {
            "total_chunks": collection.count(),  # This is correct if collection is a Collection object
            "estimated_videos": 0
        }
        
        if stats["total_chunks"] > 0:
            try:
                sample = collection.peek(10)
                # ChromaDB's peek returns a tuple: (ids, documents, metadatas)
                # Unpack it properly
                if isinstance(sample, tuple) and len(sample) == 3:
                    ids, documents, metadatas = sample
                    video_ids = {
                        metadata.get("video_id")
                        for metadata in metadatas
                        if metadata and "video_id" in metadata
                    }
                    stats["estimated_videos"] = len(video_ids)
            except Exception as e:
                logger.warning("Error getting sample data: %s", e)
        
        return stats
    except Exception as e:
        logger.error("Error getting collection stats: %s", e)
        return {"error": str(e)}

def persist_database(client):
    """Persist the ChromaDB data to disk."""
    try:
        logger.info("Persisting ChromaDB data to disk")
        client.persist()
        logger.info("Data successfully persisted")
    except Exception as e:
        logger.error("Error persisting database: %s", e)