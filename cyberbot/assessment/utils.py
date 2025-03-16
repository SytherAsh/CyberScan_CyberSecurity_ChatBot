from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate  # For dynamic questioning
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.schema import BaseRetriever, Document
from pdf2image import convert_from_path
import pytesseract
from rank_bm25 import BM25Okapi
from transformers import pipeline
from typing import List
import os
import torch

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# New imports for additional modules
from transformers import AutoTokenizer, AutoModelForTokenClassification  # NER for entity extraction
import pandas as pd  # For report structuring
from datetime import datetime  # For report timestamp