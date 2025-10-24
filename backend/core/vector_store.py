import chromadb
from chromadb.config import Settings
import os
from typing import List, Dict, Any
import uuid
import PyPDF2
from docx import Document
from io import BytesIO

class VectorStore:
    def __init__(self, persist_directory: str = "./data/chroma_db"):
        self.persist_directory = persist_directory
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collections = {}
        
    def initialize_collections(self):
        """Initialize collections for different types of data"""
        collections_config = {
            "knowledge_base": "Educational content and book knowledge",
            "user_profiles": "User learning profiles and preferences", 
            "job_descriptions": "Job postings and descriptions",
            "user_resumes": "User resume content and skills"
        }
        
        for collection_name, description in collections_config.items():
            try:
                self.collections[collection_name] = self.client.get_or_create_collection(
                    name=collection_name,
                    metadata={"description": description}
                )
                print(f"✅ Collection '{collection_name}' initialized")
            except Exception as e:
                print(f"❌ Failed to initialize collection '{collection_name}': {e}")
    
    def add_document(self, file_path: str, doc_type: str, user_id: str = None):
        """Add document to appropriate collection"""
        chunks = self._process_document(file_path)
        
        collection_name = self._get_collection_name(doc_type)
        if collection_name not in self.collections:
            raise ValueError(f"Unknown document type: {doc_type}")
        
        # Prepare documents for storage
        documents = []
        metadatas = []
        ids = []
        
        for i, chunk in enumerate(chunks):
            documents.append(chunk["content"])
            metadata = chunk["metadata"]
            metadata["user_id"] = user_id
            metadata["doc_type"] = doc_type
            metadatas.append(metadata)
            ids.append(f"{user_id}_{doc_type}_{i}_{uuid.uuid4().hex[:8]}")
        
        # Add to collection
        self.collections[collection_name].add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        
        return len(chunks)
    
    def add_document_content(self, content: str, metadata: Dict, collection_name: str = "knowledge_base"):
        """Add raw content directly to a collection - NEW METHOD for job dataset"""
        if collection_name not in self.collections:
            raise ValueError(f"Collection '{collection_name}' not found")
        
        # Simple chunking for direct content
        chunks = self._chunk_content(content)
        
        documents = []
        metadatas = []
        ids = []
        
        for i, chunk in enumerate(chunks):
            documents.append(chunk)
            chunk_metadata = metadata.copy()
            chunk_metadata["chunk_index"] = i
            chunk_metadata["total_chunks"] = len(chunks)
            metadatas.append(chunk_metadata)
            ids.append(f"direct_{collection_name}_{i}_{uuid.uuid4().hex[:8]}")
        
        # Add to collection
        self.collections[collection_name].add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        
        return len(chunks)
    
    def search(self, collection_name: str, query: str, n_results: int = 10, filters: Dict = None) -> List[Dict]:
        """Search in vector store - WORKAROUND VERSION"""
        try:
            if collection_name not in self.collections:
                print(f"❌ Collection '{collection_name}' not found")
                return []
            
            collection = self.collections[collection_name]
            
            print(f"🔍 Searching '{collection_name}' for: '{query}'")
            
            # WORKAROUND: Get all documents and filter client-side
            try:
                # Try to get all documents from the collection
                all_docs = collection.get()
                
                if not all_docs or 'documents' not in all_docs or not all_docs['documents']:
                    print(f"ℹ️ No documents found in collection '{collection_name}'")
                    return []
                
                documents = all_docs['documents']
                metadatas = all_docs.get('metadatas', [{}] * len(documents))
                ids = all_docs.get('ids', [])
                
                # Simple text-based matching (fallback when vector search fails)
                matching_docs = []
                query_lower = query.lower()
                
                for i, (doc_content, metadata) in enumerate(zip(documents, metadatas)):
                    content_lower = doc_content.lower()
                    metadata_str = str(metadata).lower()
                    
                    # Check if query matches content or metadata
                    if (query_lower in content_lower or 
                        query_lower in metadata_str or 
                        any(query_lower in str(val).lower() for val in metadata.values() if val)):
                        
                        matching_docs.append({
                            "content": doc_content,
                            "metadata": metadata or {},
                            "id": ids[i] if i < len(ids) else f"doc_{i}",
                            "relevance_score": 0.8  # Default score
                        })
                
                # If no matches found with query, return some random documents
                if not matching_docs and not query.strip():
                    print("🔄 No matches found, returning random documents")
                    for i in range(min(n_results, len(documents))):
                        matching_docs.append({
                            "content": documents[i],
                            "metadata": metadatas[i] if i < len(metadatas) else {},
                            "id": ids[i] if i < len(ids) else f"doc_{i}",
                            "relevance_score": 0.5
                        })
                
                print(f"✅ Found {len(matching_docs)} matching documents")
                return matching_docs[:n_results]
                
            except Exception as get_error:
                print(f"❌ Failed to get documents from collection: {get_error}")
                return []
            
        except Exception as e:
            print(f"❌ Search failed in collection '{collection_name}': {e}")
            return []
        
    def _get_collection_name(self, doc_type: str) -> str:
        """Map document type to collection name"""
        mapping = {
            "knowledge": "knowledge_base",
            "resume": "user_resumes", 
            "job": "job_descriptions",
            "profile": "user_profiles"
        }
        return mapping.get(doc_type, "knowledge_base")
    
    def _process_document(self, file_path: str) -> List[Dict]:
        """Process document and split into chunks"""
        try:
            # Extract text based on file type
            if file_path.lower().endswith('.pdf'):
                content = self._extract_pdf_content(file_path)
            elif file_path.lower().endswith(('.docx', '.doc')):
                content = self._extract_docx_content(file_path)
            else:
                # Assume text file
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            
            # Split into chunks
            chunks = self._chunk_content(content)
            
            # Add metadata
            result = []
            for i, chunk in enumerate(chunks):
                result.append({
                    "content": chunk,
                    "metadata": {
                        "source": os.path.basename(file_path),
                        "chunk_index": i,
                        "total_chunks": len(chunks)
                    }
                })
            
            return result
            
        except Exception as e:
            print(f"❌ Document processing failed for {file_path}: {e}")
            return []
    
    def _extract_pdf_content(self, file_path: str) -> str:
        """Extract text from PDF files"""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                
                return text.strip()
                
        except Exception as e:
            print(f"❌ PDF extraction failed: {e}")
            return ""
    
    def _extract_docx_content(self, file_path: str) -> str:
        """Extract text from DOCX files"""
        try:
            doc = Document(file_path)
            text = ""
            
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            
            return text.strip()
            
        except Exception as e:
            print(f"❌ DOCX extraction failed: {e}")
            return ""
    
    def _chunk_content(self, content: str, chunk_size: int = 1000) -> List[str]:
        """Split content into manageable chunks"""
        if not content:
            return []
        
        # Simple chunking by paragraphs
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        chunks = []
        
        current_chunk = ""
        
        for paragraph in paragraphs:
            # If adding this paragraph exceeds chunk size, save current chunk
            if len(current_chunk) + len(paragraph) > chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
            
            current_chunk += paragraph + "\n\n"
        
        # Add the last chunk if any content remains
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def clear_collection(self, collection_name: str) -> bool:
        """Clear all documents from a collection - FIXED"""
        try:
            if collection_name in self.collections:
                collection = self.collections[collection_name]
                # Get all document IDs and delete them
                results = collection.get()
                if results and 'ids' in results and results['ids']:
                    collection.delete(ids=results['ids'])
                    print(f"🧹 Cleared {len(results['ids'])} documents from: {collection_name}")
                else:
                    print(f"ℹ️ Collection '{collection_name}' is already empty")
            return True
        except Exception as e:
            print(f"❌ Failed to clear collection {collection_name}: {e}")
            # Alternative: delete and recreate collection
            try:
                print("🔄 Attempting collection recreation...")
                del self.collections[collection_name]
                self.collections[collection_name] = self.client.get_or_create_collection(collection_name)
                print(f"✅ Recreated collection: {collection_name}")
                return True
            except Exception as recreate_error:
                print(f"❌ Collection recreation failed: {recreate_error}")
                return False