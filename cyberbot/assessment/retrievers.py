# myapp/retrievers.py
from pydantic import BaseModel, ConfigDict
from langchain.schema import Document
from langchain_core.retrievers import BaseRetriever
from rank_bm25 import BM25Okapi
from langchain_community.vectorstores import FAISS

from typing import List
from langchain.schema import Document

class HybridRetriever(BaseRetriever):
    """A retriever that combines FAISS semantic search with BM25 keyword search."""
    
    model_config = ConfigDict(extra='allow')
    
    def __init__(self, vectorstore: FAISS, bm25: BM25Okapi, lambda_: float = 0.5, k: int = 5, candidates: int = 100):
        super().__init__()
        print("Initializing HybridRetriever")
        self._vectorstore = vectorstore
        self._bm25 = bm25
        self.lambda_ = lambda_
        self.k = k
        self.candidates = candidates
        print("HybridRetriever initialized")
    
    def _get_relevant_documents(self, query: str) -> List[Document]:
        print(f"Starting document retrieval for query: {query}")
        tokenized_query = query.split()
        docs_and_scores = self._vectorstore.similarity_search_with_score(query, k=self.candidates)
        similarities = [1 - (distance ** 2 / 2) for _, distance in docs_and_scores]
        indices = [doc.metadata['id'] for doc, _ in docs_and_scores]
        docs = [doc for doc, _ in docs_and_scores]
        bm25_scores = self._bm25.get_scores(tokenized_query)
        bm25_selected = [bm25_scores[i] for i in indices]
        combined_scores = [sim + self.lambda_ * bm25 for sim, bm25 in zip(similarities, bm25_selected)]
        sorted_pairs = sorted(zip(combined_scores, docs), reverse=True)
        relevant_docs = [doc for _, doc in sorted_pairs][:self.k]
        print(f"Retrieved {len(relevant_docs)} relevant documents")
        return relevant_docs