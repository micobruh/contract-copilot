import streamlit as st
from langchain_chroma import Chroma
from ..config import config
from ..model.embeddings import create_embedding_wrapper


@st.cache_resource
def load_vectorstore(embedding_model_name=config.default_embedding_model_name):
    embedding_wrapper = create_embedding_wrapper(embedding_model_name)
    return Chroma(
        collection_name=config.collection_name,
        persist_directory=config.chroma_db_link,
        embedding_function=embedding_wrapper,
    )


def vector_similarity_search(query, embedding_model_name=config.default_embedding_model_name, k_vector=config.k_vector):
    # Retrieve from vector store
    new_query = query if embedding_model_name == "jinaai/jina-embeddings-v5-text-small" else "query: " + query
    vectorstore = load_vectorstore(embedding_model_name)
    vector_docs = vectorstore.similarity_search(new_query, k=k_vector)
    return vector_docs
