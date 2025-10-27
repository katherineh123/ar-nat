"""
RAG Question Answering Tool for AR Lab Assistant.
This module contains the RAGSystem class and the RAG question answering function.
"""

import logging
import os
import pickle
from typing import List, Dict, Any

import faiss
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig

logger = logging.getLogger(__name__)


class RAGSystem:
    """
    A RAG (Retrieval-Augmented Generation) system using FAISS for vector search
    and sentence transformers for embeddings.
    """
    
    def __init__(self, pdf_path: str, index_path: str = "data/faiss_index", 
                 documents_path: str = "data/documents.pkl"):
        self.pdf_path = pdf_path
        self.index_path = index_path
        self.documents_path = documents_path
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        self.index = None
        self.documents = []
        
    def load_or_create_index(self):
        """Load existing FAISS index or create new one from PDF."""
        if os.path.exists(self.index_path) and os.path.exists(self.documents_path):
            logger.info("Loading existing FAISS index and documents")
            self.index = faiss.read_index(self.index_path)
            with open(self.documents_path, 'rb') as f:
                self.documents = pickle.load(f)
        else:
            logger.info("Creating new FAISS index from PDF")
            self._create_index_from_pdf()
            
    def _create_index_from_pdf(self):
        """Extract text from PDF, create embeddings, and build FAISS index."""
        try:
            # Load PDF
            loader = PyPDFLoader(self.pdf_path)
            pages = loader.load()
            
            # Split text into chunks
            texts = self.text_splitter.split_documents(pages)
            self.documents = [doc.page_content for doc in texts]
            
            logger.info(f"Created {len(self.documents)} text chunks from PDF")
            
            # Create embeddings
            embeddings = self.embedding_model.encode(self.documents)
            
            # Create FAISS index
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings)
            self.index.add(embeddings.astype('float32'))
            
            # Save index and documents
            index_dir = os.path.dirname(self.index_path)
            if index_dir and not os.path.exists(index_dir):
                os.makedirs(index_dir)
            faiss.write_index(self.index, self.index_path)
            with open(self.documents_path, 'wb') as f:
                pickle.dump(self.documents, f)
                
            logger.info(f"FAISS index created and saved with {len(self.documents)} documents")
            
        except Exception as e:
            logger.error(f"Error creating index from PDF: {e}")
            raise
    
    def search(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """Search for relevant documents using semantic similarity."""
        if self.index is None:
            self.load_or_create_index()
            
        # Create query embedding
        query_embedding = self.embedding_model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding.astype('float32'), k)
        
        # Return results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.documents):
                results.append({
                    'content': self.documents[idx],
                    'score': float(score),
                    'index': int(idx)
                })
        
        return results


class RAGQuestionAnsweringConfig(FunctionBaseConfig, name="rag_question_answering"):
    """
    RAG-based question answering tool for lab science experiments.
    Answers questions about lab procedures, safety, and scientific concepts using PDF knowledge base.
    """
    pdf_path: str = Field(
        default="src/ar_lab_assistant/data/Kirby-Bauer-Disk-Diffusion-Susceptibility-Test-Protocol.pdf",
        description="Path to the PDF knowledge base file"
    )
    experiment_type: str = Field(
        default="Kirby-Bauer disk diffusion assay",
        description="Type of experiment being performed"
    )
    max_results: int = Field(
        default=3,
        description="Maximum number of relevant documents to retrieve"
    )


@register_function(config_type=RAGQuestionAnsweringConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def rag_question_answering_function(config: RAGQuestionAnsweringConfig, builder: Builder):
    """
    Registers the RAG question answering function for lab science questions.
    """
    
    # Initialize RAG system
    rag_system = RAGSystem(
        pdf_path=config.pdf_path,
        index_path=f"data/{config.experiment_type.replace(' ', '_').lower()}_faiss_index",
        documents_path=f"data/{config.experiment_type.replace(' ', '_').lower()}_documents.pkl"
    )
    
    async def _rag_question_answering(question: str) -> str:
        """
        Answers questions about lab experiments using RAG (Retrieval-Augmented Generation).
        Uses semantic search on PDF knowledge base to find relevant information.
        
        Args:
            question (str): The question about the lab experiment.
            
        Returns:
            str: The answer to the question based on retrieved documents.
        """
        logger.info("RAG Question Answering: %s", question)
        
        try:
            # Search for relevant documents
            results = rag_system.search(question, k=config.max_results)
            
            if not results:
                return f"I couldn't find relevant information about '{question}' in the {config.experiment_type} protocol. Could you try rephrasing your question or ask about a different aspect of the experiment?"
            
            # Combine relevant documents
            context_parts = []
            for i, result in enumerate(results, 1):
                content = result['content'].strip()
                score = result['score']
                context_parts.append(f"Source {i} (relevance: {score:.2f}):\n{content}")
            
            context = "\n\n".join(context_parts)
            
            # Generate answer using retrieved context
            answer = f"Based on the {config.experiment_type} protocol, here's what I found:\n\n{context}\n\nThis information should help answer your question about '{question}'. If you need more specific details, please let me know!"
            
            return answer
            
        except Exception as e:
            logger.error(f"Error in RAG question answering: {e}")
            return f"I encountered an error while searching for information about '{question}'. Please try again or ask a different question about the {config.experiment_type} experiment."
    
    yield FunctionInfo.from_fn(_rag_question_answering, description=_rag_question_answering.__doc__)
