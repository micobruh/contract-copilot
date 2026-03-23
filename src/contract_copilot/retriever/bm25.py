from langchain_community.retrievers import BM25Retriever
import pickle
import streamlit as st
from ..config import config


@st.cache_data
def load_bm25_docs():
    # Retrieve from BM25
    with open(config.bm25_doc_link, "rb") as f:
        return pickle.load(f)
    

@st.cache_resource
def load_bm25_retriever(k_bm25=config.k_bm25):
    docs = load_bm25_docs()
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = k_bm25
    return bm25_retriever    


def bm25_retrieval(query, k_bm25=config.k_bm25):
    # Retrieve from BM25
    bm25_retriever = load_bm25_retriever(k_bm25)
    bm25_docs = bm25_retriever.invoke(query)
    return bm25_docs
