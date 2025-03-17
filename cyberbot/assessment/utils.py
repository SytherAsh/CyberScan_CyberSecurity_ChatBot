import os
from typing import List
from langchain.schema import Document
from django.conf import settings
from langchain_community.document_loaders import PyPDFLoader
import pytesseract
from pdf2image import convert_from_path
from langchain.text_splitter import RecursiveCharacterTextSplitter
import logging

logger = logging.getLogger(__name__)

def get_pdf_files(directory: str = None) -> List[str]:
    """Retrieve list of PDF files from directory."""
    directory = directory or settings.PDF_DIR
    if not os.path.exists(directory):
        logger.error(f"PDF directory not found: {directory}")
        return []
    pdf_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.pdf')]
    logger.info(f"Found {len(pdf_files)} PDF files")
    return pdf_files

def load_and_process_pdf(file_path: str) -> List[Document]:
    """Load PDF and apply OCR to scanned pages."""
    try:
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        images = convert_from_path(file_path)
        for i, doc in enumerate(documents):
            if len(doc.page_content.strip()) < 50:
                logger.info(f"Applying OCR to page {i+1} in {file_path}")
                doc.page_content = pytesseract.image_to_string(images[i])
        return documents
    except Exception as e:
        logger.error(f"Failed to process {file_path}: {e}")
        return []

def load_documents(file_paths: List[str] = None) -> List[Document]:
    """Load and process multiple PDFs."""
    file_paths = file_paths or get_pdf_files()
    all_documents = []
    for file_path in file_paths:
        docs = load_and_process_pdf(file_path)
        all_documents.extend(docs)
    logger.info(f"Loaded {len(all_documents)} document pages")
    return all_documents

def split_documents(documents: List[Document]) -> List[Document]:
    """Split documents into manageable chunks."""
    try:
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(documents)
        logger.info(f"Split into {len(chunks)} chunks")
        return chunks
    except Exception as e:
        logger.error(f"Document splitting failed: {e}")
        return []