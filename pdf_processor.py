import os
import PyPDF2
import chromadb
from sentence_transformers import SentenceTransformer
from typing import List, Optional
import uuid
import logging
import time
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFProcessor:
    def __init__(self, vector_store_path="./vector_store"):
        self.vector_store_path = vector_store_path
        os.makedirs(vector_store_path, exist_ok=True)
        
        logger.info(f"Initializing PDFProcessor with vector store at: {vector_store_path}")
        
        try:
            # Initialize embedding model - using a faster, smaller model
            logger.info("Loading embedding model...")
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # Fast and efficient
            logger.info("Embedding model loaded successfully")
            
            # Initialize ChromaDB
            logger.info("Initializing ChromaDB...")
            self.client = chromadb.PersistentClient(path=vector_store_path)
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name="pdf_documents",
                metadata={"hnsw:space": "cosine"}
            )
            
            # Count existing documents
            doc_count = self.collection.count()
            logger.info(f"PDFProcessor initialized. Existing documents: {doc_count}")
            
            # Pre-load some embeddings for faster search
            self._preload_embeddings()
            
        except Exception as e:
            logger.error(f"Failed to initialize PDFProcessor: {str(e)}")
            raise
    
    def _preload_embeddings(self):
        """Pre-load a subset of embeddings for common queries"""
        self.common_query_embeddings = {}
        # Pre-compute embeddings for common queries
        common_queries = [
            "contact", "email", "phone", "address", "location",
            "product", "service", "price", "cost", "specification",
            "support", "help", "technical", "manual", "guide"
        ]
        
        for query in common_queries:
            self.common_query_embeddings[query] = self.embedding_model.encode(query)
    
    def get_collection(self):
        """Get the ChromaDB collection"""
        return self.collection
    
    def extract_text_from_pdf(self, pdf_path: str) -> List[str]:
        """Extract text from PDF and split into chunks - optimized"""
        text_chunks = []
        
        try:
            logger.info(f"Extracting text from PDF: {pdf_path}")
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                total_pages = len(pdf_reader.pages)
                
                logger.info(f"PDF has {total_pages} pages")
                
                # Extract text from all pages at once
                all_text = []
                for page_num in range(total_pages):
                    page = pdf_reader.pages[page_num]
                    text = page.extract_text()
                    if text and text.strip():
                        all_text.append(text)
                
                # Join and split into larger chunks for better context
                full_text = " ".join(all_text)
                chunks = self._split_text_into_chunks(full_text, chunk_size=1000)  # Larger chunks
                text_chunks.extend(chunks)
                
                logger.info(f"Extracted {len(text_chunks)} text chunks from {total_pages} pages")
                
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            raise
        
        return text_chunks
    
    def _split_text_into_chunks(self, text: str, chunk_size: int = 1000) -> List[str]:
        """Split text into manageable chunks - optimized"""
        # Simple split by sentences first, then combine
        sentences = text.replace('\n', ' ').split('. ')
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence.split())
            if current_length + sentence_length <= chunk_size:
                current_chunk.append(sentence)
                current_length += sentence_length
            else:
                if current_chunk:
                    chunks.append('. '.join(current_chunk) + '.')
                current_chunk = [sentence]
                current_length = sentence_length
        
        if current_chunk:
            chunks.append('. '.join(current_chunk) + '.')
        
        return chunks
    
    def process_pdf(self, pdf_path: str):
        """Process a PDF file and store embeddings - optimized"""
        logger.info(f"Processing PDF: {pdf_path}")
        
        try:
            # Extract text chunks
            text_chunks = self.extract_text_from_pdf(pdf_path)
            
            if not text_chunks:
                logger.warning("No text extracted from PDF")
                return
            
            # Limit number of chunks for very large documents
            if len(text_chunks) > 50:
                logger.info(f"Reducing {len(text_chunks)} chunks to 50 for efficiency")
                text_chunks = text_chunks[:50]  # Keep first 50 chunks
            
            logger.info(f"Generating embeddings for {len(text_chunks)} chunks...")
            
            # Generate embeddings in batches for efficiency
            batch_size = 10
            all_embeddings = []
            all_ids = []
            all_metadatas = []
            
            for i in range(0, len(text_chunks), batch_size):
                batch = text_chunks[i:i+batch_size]
                batch_embeddings = self.embedding_model.encode(batch).tolist()
                all_embeddings.extend(batch_embeddings)
                
                # Create IDs and metadata for this batch
                for j in range(len(batch)):
                    all_ids.append(str(uuid.uuid4()))
                    all_metadatas.append({
                        "source": os.path.basename(pdf_path),
                        "chunk_index": i + j
                    })
            
            # Add to collection in one go
            self.collection.add(
                documents=text_chunks,
                embeddings=all_embeddings,
                metadatas=all_metadatas,
                ids=all_ids
            )
            
            logger.info(f"Successfully stored {len(text_chunks)} chunks in vector database")
            logger.info(f"Total documents in collection: {self.collection.count()}")
            
        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}")
            raise
    
    def fast_search(self, query: str, top_k: int = 2) -> str:
        """Faster search using keyword matching first"""
        try:
            if self.collection.count() == 0:
                return ""
            
            # First try keyword search for speed
            keywords = query.lower().split()
            results = self.collection.get(
                where={"source": {"$ne": ""}},  # Get all with source
                limit=top_k * 3  # Get more for filtering
            )
            
            if not results or not results.get('documents'):
                return self.search_context(query, top_k)
            
            # Simple keyword matching
            scored_chunks = []
            for i, doc in enumerate(results['documents']):
                doc_lower = doc.lower()
                score = sum(1 for keyword in keywords if keyword in doc_lower)
                if score > 0:
                    source = results['metadatas'][i].get('source', 'Unknown') if results['metadatas'] else 'Unknown'
                    scored_chunks.append((score, f"[From: {source}]\n{doc}"))
            
            # Sort by score and take top_k
            scored_chunks.sort(reverse=True, key=lambda x: x[0])
            context_chunks = [chunk for _, chunk in scored_chunks[:top_k]]
            
            if context_chunks:
                return "\n\n---\n\n".join(context_chunks)
            else:
                # Fall back to embedding search
                return self.search_context(query, top_k)
            
        except Exception as e:
            logger.error(f"Error in fast search: {str(e)}")
            return self.search_context(query, top_k)
    
    def search_context(self, query: str, top_k: int = 2) -> str:
        """Search for relevant context based on query - optimized"""
        try:
            if self.collection.count() == 0:
                logger.info("No documents in collection")
                return ""
            
            start_time = time.time()
            
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query]).tolist()[0]
            
            # Search in vector database with reduced results for speed
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, self.collection.count()),
                include=["documents", "metadatas"]
            )
            
            search_time = time.time() - start_time
            logger.info(f"Vector search completed in {search_time:.2f}s")
            
            # Combine results
            if results['documents']:
                context_chunks = []
                for i, doc in enumerate(results['documents'][0]):
                    source = results['metadatas'][0][i].get('source', 'Unknown') if results['metadatas'] else 'Unknown'
                    # Truncate long documents
                    if len(doc) > 500:
                        doc = doc[:497] + "..."
                    context_chunks.append(f"[From: {source}]\n{doc}")
                
                context = "\n\n---\n\n".join(context_chunks)
                logger.info(f"Found {len(context_chunks)} relevant chunks for query")
                return context
            
            return ""
            
        except Exception as e:
            logger.error(f"Error searching context: {str(e)}")
            return ""