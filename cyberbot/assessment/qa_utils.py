from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from transformers import pipeline
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.memory import ConversationBufferMemory
import logging

logger = logging.getLogger(__name__)
summarizer = None

from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate

def setup_retrieval_qa(llm, retriever):
    """Set up the ConversationalRetrievalChain without memory."""
    print("Starting to set up ConversationalRetrievalChain")
    custom_prompt = PromptTemplate.from_template(
        "You are a cybersecurity assistant. Answer the following question using the provided context from retrieved documents and chat history:\n"
        "Context: {context}\n"
        "Chat History: {chat_history}\n"
        "Question: {question}\n"
        "Answer:"
    )
    retrieval_qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        combine_docs_chain_kwargs={"prompt": custom_prompt}
    )
    print("ConversationalRetrievalChain setup completed")
    return retrieval_qa





def get_summarizer():
    """Lazily load the summarization model."""
    global summarizer
    if summarizer is None:
        try:
            summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
            logger.info("Summarizer loaded")
        except Exception as e:
            logger.error(f"Summarizer loading failed: {e}")
            raise
    return summarizer