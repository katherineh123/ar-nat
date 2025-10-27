import asyncio
import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import AsyncGenerator, List, Dict, Any
import pickle
import os

import faiss
import numpy as np
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


class VisualProcessGuidanceConfig(FunctionBaseConfig, name="visual_process_guidance"):
    """
    Visual Process Guidance tool for AR lab experiments.
    Provides step-by-step guidance through lab procedures with visual feedback.
    """
    experiment_type: str = Field(default="Kirby-Bauer disk diffusion assay", description="Type of experiment being performed")


class RAGQuestionAnsweringConfig(FunctionBaseConfig, name="rag_question_answering"):
    """
    RAG-based question answering tool for lab science experiments.
    Answers questions about lab procedures, safety, and scientific concepts using PDF knowledge base.
    """
    pdf_path: str = Field(default="src/ar_lab_assistant/data/Kirby-Bauer-Disk-Diffusion-Susceptibility-Test-Protocol.pdf", description="Path to the PDF knowledge base file")
    experiment_type: str = Field(default="Kirby-Bauer disk diffusion assay", description="Type of experiment being performed")
    max_results: int = Field(default=3, description="Maximum number of relevant documents to retrieve")


class ExperimentLoggingConfig(FunctionBaseConfig, name="experiment_logging"):
    """
    Experiment logging tool for recording lab sessions.
    Logs experiment summaries, timestamps, and student information.
    """
    db_path: str = Field(default="data/experiment_logs.db", description="Path to the SQLite database file")
    experiment_type: str = Field(default="Kirby-Bauer disk diffusion assay", description="Type of experiment being performed")


@register_function(config_type=VisualProcessGuidanceConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def visual_process_guidance_function(config: VisualProcessGuidanceConfig, builder: Builder):
    """
    Registers the Visual Process Guidance function for AR lab experiments.
    """
    
    async def _visual_process_guidance(trigger: str = "start") -> str:
        """
        Initiates visual process guidance for lab experiments.
        Provides step-by-step instructions with audio feedback.
        This is a mock implementation that simulates the AR guidance process.
        
        Args:
            trigger (str): Trigger to start the guidance process
        
        Returns:
            str: Confirmation that the guidance process is complete.
        """
        logger.info("Starting Visual Process Guidance for %s experiment", config.experiment_type)
        
        # Mock the AR guidance process with step-by-step feedback
        steps = [
            "Step 1: Prepare your bacterial suspension by pipetting 4ml of saline.",
            "Step 2: Gently shake the petri dish back and forth.",
            "Congratulations! You have successfully completed the experiment."
        ]
        
        guidance_output = []
        for i, step in enumerate(steps, 1):
            guidance_output.append(f"Step {i}: {step}")
            await asyncio.sleep(0.5)  # Simulate processing time
        
        logger.info("Visual Process Guidance completed")
        return "\n".join(guidance_output)
    
    yield FunctionInfo.from_fn(_visual_process_guidance, description=_visual_process_guidance.__doc__)


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


@register_function(config_type=ExperimentLoggingConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def experiment_logging_function(config: ExperimentLoggingConfig, builder: Builder):
    """
    Registers the experiment logging function for recording lab sessions.
    """
    
    async def _experiment_logging(conversation_summary: str) -> str:
        """
        Logs the experiment session to a database.
        
        Args:
            conversation_summary (str): Summary of the conversation and experiment performed.
            
        Returns:
            str: Confirmation of logging completion.
        """
        logger.info("Logging experiment session")
        
        # Ensure data directory exists
        data_dir = Path(config.db_path).parent
        data_dir.mkdir(exist_ok=True)
        
        # Initialize database if it doesn't exist
        conn = sqlite3.connect(config.db_path)
        cursor = conn.cursor()
        
        # Create table if it doesn't exist
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS experiment_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                scientist_name TEXT NOT NULL,
                experiment_type TEXT NOT NULL,
                conversation_summary TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Get the next student number
        cursor.execute('SELECT COUNT(*) FROM experiment_logs')
        student_count = cursor.fetchone()[0]
        scientist_name = f"Student {student_count + 1}"
        
        # Insert the log entry
        timestamp = datetime.now().isoformat()
        cursor.execute('''
            INSERT INTO experiment_logs (timestamp, scientist_name, experiment_type, conversation_summary)
            VALUES (?, ?, ?, ?)
        ''', (timestamp, scientist_name, config.experiment_type, conversation_summary))
        
        conn.commit()
        conn.close()
        
        logger.info("Experiment logged for %s", scientist_name)
        return f"Experiment session logged successfully for {scientist_name}. Session ended."
    
    yield FunctionInfo.from_fn(_experiment_logging, description=_experiment_logging.__doc__)
