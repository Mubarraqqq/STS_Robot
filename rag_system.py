"""
Modular RAG (Retrieval-Augmented Generation) System
Handles document indexing, retrieval, and context management
"""

import os
import pickle
import logging
import numpy as np
import faiss
import openai
from typing import List, Tuple, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class KnowledgeBase:
    """Manages document chunks and embeddings."""
    
    def __init__(self, knowledge_base_path: str = "info.txt"):
        """
        Initialize knowledge base.
        
        Args:
            knowledge_base_path: Path to the knowledge base text file
        """
        self.knowledge_base_path = knowledge_base_path
        self.chunks: List[str] = []
        
    def load_and_chunk(self, chunk_size: int = 500, chunk_overlap: int = 50) -> List[str]:
        """
        Load and chunk the knowledge base document.
        
        Args:
            chunk_size: Size of each chunk
            chunk_overlap: Overlap between consecutive chunks
            
        Returns:
            List of text chunks
        """
        try:
            if not os.path.exists(self.knowledge_base_path):
                logger.warning(f"Knowledge base not found at {self.knowledge_base_path}")
                return []
            
            with open(self.knowledge_base_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            logger.info(f"Loaded {len(text)} characters from knowledge base")
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            self.chunks = text_splitter.split_text(text)
            logger.info(f"Created {len(self.chunks)} chunks")
            
            return self.chunks
            
        except Exception as e:
            logger.error(f"Error loading knowledge base: {e}")
            return []
    
    def add_chunk(self, text: str):
        """Add a new chunk to the knowledge base."""
        self.chunks.append(text)
    
    def update_chunks(self, new_chunks: List[str]):
        """Update all chunks."""
        self.chunks = new_chunks
        logger.info(f"Updated knowledge base with {len(new_chunks)} chunks")


class EmbeddingManager:
    """Manages document embeddings using OpenAI API."""
    
    def __init__(self, model: str = "text-embedding-ada-002"):
        """
        Initialize embedding manager.
        
        Args:
            model: OpenAI embedding model to use
        """
        self.model = model
        self.embeddings: Optional[np.ndarray] = None
    
    def embed_chunks(self, chunks: List[str]) -> np.ndarray:
        """
        Generate embeddings for text chunks.
        
        Args:
            chunks: List of text chunks to embed
            
        Returns:
            NumPy array of embeddings
        """
        try:
            if not chunks:
                logger.warning("No chunks to embed")
                return np.array([])
            
            logger.info(f"Embedding {len(chunks)} chunks...")
            
            response = openai.Embedding.create(
                input=chunks,
                model=self.model
            )
            
            embeddings = [item["embedding"] for item in response["data"]]
            self.embeddings = np.array(embeddings, dtype="float32")
            
            logger.info(f"Created embeddings with shape {self.embeddings.shape}")
            return self.embeddings
            
        except Exception as e:
            logger.error(f"Error creating embeddings: {e}")
            return np.array([])
    
    def embed_query(self, query: str) -> Optional[np.ndarray]:
        """
        Generate embedding for a query string.
        
        Args:
            query: Query text to embed
            
        Returns:
            NumPy array containing the query embedding
        """
        try:
            response = openai.Embedding.create(
                input=[query],
                model=self.model
            )
            embedding = np.array([response["data"][0]["embedding"]], dtype="float32")
            return embedding
            
        except Exception as e:
            logger.error(f"Error embedding query: {e}")
            return None
    
    def save(self, filepath: str):
        """Save embeddings to file."""
        if self.embeddings is not None:
            np.save(filepath, self.embeddings)
            logger.info(f"Saved embeddings to {filepath}")
    
    def load(self, filepath: str):
        """Load embeddings from file."""
        if os.path.exists(filepath):
            self.embeddings = np.load(filepath)
            logger.info(f"Loaded embeddings from {filepath}")
            return self.embeddings
        return None


class FAISSIndex:
    """Manages FAISS index for similarity search."""
    
    def __init__(self, embeddings: np.ndarray):
        """
        Initialize FAISS index.
        
        Args:
            embeddings: NumPy array of document embeddings
        """
        self.index = self._create_index(embeddings)
    
    def _create_index(self, embeddings: np.ndarray) -> faiss.IndexFlatL2:
        """Create FAISS index from embeddings."""
        if embeddings.size == 0:
            logger.warning("Cannot create index from empty embeddings")
            return None
        
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        logger.info(f"Created FAISS index with {index.ntotal} vectors")
        return index
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for similar documents.
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            
        Returns:
            Tuple of (distances, indices)
        """
        if self.index is None:
            return np.array([]), np.array([])
        
        return self.index.search(query_embedding, k)
    
    def save(self, filepath: str):
        """Save index to file."""
        if self.index:
            faiss.write_index(self.index, filepath)
            logger.info(f"Saved FAISS index to {filepath}")
    
    @staticmethod
    def load(filepath: str) -> Optional['FAISSIndex']:
        """Load index from file."""
        if os.path.exists(filepath):
            index = faiss.read_index(filepath)
            logger.info(f"Loaded FAISS index from {filepath}")
            obj = FAISSIndex.__new__(FAISSIndex)
            obj.index = index
            return obj
        return None


class RAGRetriever:
    """Retrieves relevant context from knowledge base."""
    
    def __init__(self, chunks: List[str], embeddings: np.ndarray):
        """
        Initialize retriever.
        
        Args:
            chunks: List of document chunks
            embeddings: NumPy array of embeddings
        """
        self.chunks = chunks
        self.embeddings = embeddings
        self.faiss_index = FAISSIndex(embeddings) if embeddings.size > 0 else None
    
    def retrieve(self, query_embedding: np.ndarray, k: int = 5) -> List[Tuple[str, float]]:
        """
        Retrieve relevant chunks for a query.
        
        Args:
            query_embedding: Query embedding
            k: Number of chunks to retrieve
            
        Returns:
            List of (chunk, similarity_score) tuples, sorted by relevance
        """
        if self.faiss_index is None:
            logger.warning("FAISS index not available")
            return []
        
        distances, indices = self.faiss_index.search(query_embedding, k)
        
        results = []
        for i in range(len(indices[0])):
            if indices[0][i] != -1:  # Valid index
                chunk = self.chunks[indices[0][i]]
                chunk_embedding = self.embeddings[indices[0][i]].reshape(1, -1)
                similarity = cosine_similarity(query_embedding, chunk_embedding)[0][0]
                results.append((chunk, similarity))
        
        # Sort by similarity (highest first)
        results.sort(key=lambda x: x[1], reverse=True)
        return results


class RAGSystem:
    """Main RAG system orchestrator."""
    
    def __init__(self, 
                 kb_path: str = "info.txt",
                 faiss_path: str = "faiss_index.idx",
                 embeddings_path: str = "embeddings.npy",
                 chunks_path: str = "doc_chunks.pkl"):
        """
        Initialize RAG system.
        
        Args:
            kb_path: Path to knowledge base file
            faiss_path: Path to FAISS index file
            embeddings_path: Path to embeddings file
            chunks_path: Path to chunks pickle file
        """
        self.kb_path = kb_path
        self.faiss_path = faiss_path
        self.embeddings_path = embeddings_path
        self.chunks_path = chunks_path
        
        self.knowledge_base = KnowledgeBase(kb_path)
        self.embedding_manager = EmbeddingManager()
        self.retriever: Optional[RAGRetriever] = None
    
    def initialize(self):
        """Initialize or load RAG system."""
        if self._load_from_disk():
            logger.info("✅ Loaded RAG system from disk")
        else:
            logger.info("⚙️ Building RAG system from scratch")
            self._build_from_scratch()
    
    def _load_from_disk(self) -> bool:
        """Load RAG system from disk."""
        try:
            if not all(os.path.exists(p) for p in [
                self.faiss_path, 
                self.chunks_path, 
                self.embeddings_path
            ]):
                return False
            
            # Load chunks
            with open(self.chunks_path, "rb") as f:
                chunks = pickle.load(f)
            
            # Load embeddings
            embeddings = self.embedding_manager.load(self.embeddings_path)
            
            # Load FAISS index
            faiss_index = FAISSIndex.load(self.faiss_path)
            
            if chunks and embeddings is not None and faiss_index:
                self.knowledge_base.chunks = chunks
                self.embedding_manager.embeddings = embeddings
                self.retriever = RAGRetriever(chunks, embeddings)
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error loading from disk: {e}")
            return False
    
    def _build_from_scratch(self):
        """Build RAG system from scratch."""
        try:
            # Load and chunk knowledge base
            chunks = self.knowledge_base.load_and_chunk()
            if not chunks:
                logger.warning("No chunks loaded from knowledge base")
                return
            
            # Generate embeddings
            embeddings = self.embedding_manager.embed_chunks(chunks)
            if embeddings.size == 0:
                logger.warning("Failed to generate embeddings")
                return
            
            # Create retriever
            self.retriever = RAGRetriever(chunks, embeddings)
            
            # Save to disk
            self._save_to_disk()
            logger.info("✅ RAG system built and saved")
            
        except Exception as e:
            logger.error(f"Error building RAG system: {e}")
    
    def _save_to_disk(self):
        """Save RAG system to disk."""
        try:
            # Save embeddings
            self.embedding_manager.save(self.embeddings_path)
            
            # Save chunks
            with open(self.chunks_path, "wb") as f:
                pickle.dump(self.knowledge_base.chunks, f)
            
            # Save FAISS index
            faiss_index = FAISSIndex(self.embedding_manager.embeddings)
            faiss_index.save(self.faiss_path)
            
            logger.info("Saved RAG system to disk")
            
        except Exception as e:
            logger.error(f"Error saving RAG system: {e}")
    
    def retrieve_context(self, query: str, k: int = 3) -> List[Tuple[str, float]]:
        """
        Retrieve context for a query.
        
        Args:
            query: User query
            k: Number of results to retrieve
            
        Returns:
            List of (chunk, similarity_score) tuples
        """
        if self.retriever is None:
            logger.warning("Retriever not initialized")
            return []
        
        query_embedding = self.embedding_manager.embed_query(query)
        if query_embedding is None:
            return []
        
        return self.retriever.retrieve(query_embedding, k)
    
    def get_confidence_level(self, similarity_score: float) -> str:
        """
        Determine confidence level based on similarity score.
        
        Args:
            similarity_score: Similarity score (0-1)
            
        Returns:
            Confidence level string
        """
        if similarity_score >= 0.85:
            return "HIGH"
        elif similarity_score >= 0.75:
            return "MEDIUM"
        else:
            return "LOW"
