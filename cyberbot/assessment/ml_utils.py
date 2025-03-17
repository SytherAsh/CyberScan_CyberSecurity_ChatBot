from django.conf import settings
import os
import pickle
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from .utils import load_documents, split_documents
from .qa_utils import setup_retrieval_qa
import torch
from typing import List

import logging
from threading import Lock

from pydantic import BaseModel
from transformers import pipeline

logger = logging.getLogger(__name__)
lock = Lock()

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_DIR = os.path.join(settings.BASE_DIR, 'models', 'sentence-transformers')
EMBEDDINGS_FILE = os.path.join(settings.FAISS_INDEX_PATH, 'embeddings.pkl')

llm, embeddings, vectorstore, retrieval_qa = None, None, None, None
INITIALIZED = False

def create_embeddings():
    """Create or load embeddings model."""
    try:
        if not os.path.exists(MODEL_DIR):
            from huggingface_hub import snapshot_download
            os.makedirs(MODEL_DIR, exist_ok=True)
            snapshot_download(repo_id=MODEL_NAME, local_dir=MODEL_DIR)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        embeddings = HuggingFaceEmbeddings(model_name=MODEL_DIR, model_kwargs={'device': device})
        logger.info("Embeddings initialized")
        return embeddings
    except Exception as e:
        logger.error(f"Embeddings creation failed: {e}")
        raise

def create_vectorstore(documents, embeddings):
    """Create or load FAISS vectorstore."""
    try:
        save_path = settings.FAISS_INDEX_PATH
        if os.path.exists(save_path):
            vectorstore = FAISS.load_local(save_path, embeddings, allow_dangerous_deserialization=True)
            logger.info("Loaded existing vectorstore")
        else:
            vectorstore = FAISS.from_documents(documents, embeddings)
            vectorstore.save_local(save_path)
            logger.info("Created and saved new vectorstore")
        return vectorstore
    except Exception as e:
        logger.error(f"Vectorstore creation failed: {e}")
        raise

def setup_llm():
    """Initialize the language model."""
    try:
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.3,
            max_tokens=512,
            api_key=settings.GROQ_API_KEY
        )
        logger.info("LLM initialized")
        return llm
    except Exception as e:
        logger.error(f"LLM setup failed: {e}")
        raise

def initialize_models():
    """Thread-safe model initialization."""
    global llm, embeddings, vectorstore, retrieval_qa, INITIALIZED
    with lock:
        if not INITIALIZED:
            embeddings = create_embeddings()
            documents = load_documents()
            if not documents:
                raise ValueError("No documents found")
            chunks = split_documents(documents)
            vectorstore = create_vectorstore(chunks, embeddings)
            llm = setup_llm()
            retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
            retrieval_qa = setup_retrieval_qa(llm, retriever)
            INITIALIZED = True
            logger.info("All models initialized")


def process_query(query: str, chat_history: List[dict]) -> str:
    # print(f"Processing query: {query}")
    # if chat_history:
    #     print(f"Chat history provided: {chat_history}")
    
    # Convert chat_history to list of (human, ai) tuples
    history = [(entry["question"], entry["answer"]) for entry in chat_history]
    
    input_dict = {"question": query, "chat_history": history}
    # print(f"Invoking retrieval_qa with: {input_dict}")
    result = retrieval_qa.invoke(input_dict)
    answer = result["answer"]
    # print(f"Query processed with answer: {answer[:50]}...")
    return answer

