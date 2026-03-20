from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_ollama import OllamaLLM
import pickle
from collections import defaultdict
import streamlit as st
from model.embeddings import create_embedding_wrapper
from model.reranker import rerank


@st.cache_resource
def load_vectorstore(embedding_model_name):
    embedding_wrapper = create_embedding_wrapper(embedding_model_name)
    return Chroma(
        collection_name="Law_RAG",
        persist_directory="./chroma_db",
        embedding_function=embedding_wrapper,
    )


def vector_similarity_search(query, embedding_model_name="BAAI/bge-m3", k_vector=10):
    # Retrieve from vector store
    new_query = query if embedding_model_name == "jinaai/jina-embeddings-v5-text-small" else "query: " + query
    vectorstore = load_vectorstore(embedding_model_name)
    vector_docs = vectorstore.similarity_search(new_query, k=k_vector)
    return vector_docs


@st.cache_data
def load_bm25_docs():
    # Retrieve from BM25
    with open("documents/docs.pkl", "rb") as f:
        return pickle.load(f)
    

@st.cache_resource
def load_bm25_retriever(k_bm25):
    docs = load_bm25_docs()
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = k_bm25
    return bm25_retriever    


def bm25_retrieval(query, k_bm25=10):
    # Retrieve from BM25
    bm25_retriever = load_bm25_retriever(k_bm25)
    bm25_docs = bm25_retriever.invoke(query)
    return bm25_docs


def add_rrf_scores(docs, scores, doc_map, k_rrf):
    for rank, doc in enumerate(docs, start=1):
        doc_id = doc.metadata.get("doc_id") or doc.metadata.get("id") or doc.page_content
        scores[doc_id] += 1.0 / (k_rrf + rank)
        doc_map[doc_id] = doc    


def rrf_fuse(vector_docs, bm25_docs, k_rrf=10):
    """
    Reciprocal Rank Fusion.
    Higher score = better.
    """
    scores = defaultdict(float)
    doc_map = {}

    # Add scores from both retrievals
    add_rrf_scores(vector_docs, scores, doc_map, k_rrf)
    add_rrf_scores(bm25_docs, scores, doc_map, k_rrf)

    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_map[doc_id] for doc_id, _ in fused][: k_rrf]


def retrieve(query, embedding_model_name="BAAI/bge-m3", reranker_model_name="BAAI/bge-reranker-base", k_vector=10, k_bm25=10, k_rrf=10, k_rerank=3, rerank_needed=True):
    vector_docs = vector_similarity_search(query, embedding_model_name=embedding_model_name, k_vector=k_vector)
    bm25_docs = bm25_retrieval(query, k_bm25=k_bm25)
    fused_docs = rrf_fuse(vector_docs, bm25_docs, k_rrf=k_rrf)
    if rerank_needed:
        final_docs = rerank(query, fused_docs, reranker_model_name=reranker_model_name, k_rerank=k_rerank)
    else:
        final_docs = fused_docs[: k_rerank]
    return final_docs


def format_context(docs):
    parts = []
    for i, doc in enumerate(docs, 1):
        title = doc.metadata.get("title", "Unknown")
        source = doc.metadata.get("source", "")
        parts.append(
            f"[{i}] Title: {title}\n"
            f"Source: {source}\n"
            f"Text: {doc.page_content.strip()}"
        )
    return "\n\n---\n\n".join(parts)


@st.cache_resource
def load_llm(llm_model_name):
    return OllamaLLM(model=llm_model_name)


@st.cache_resource
def build_rag_chain(llm_model_name):
    prompt = ChatPromptTemplate.from_template(
        """
        You are a helpful Law AI assistant.

        Given this question: {query}

        Answer using ONLY the provided document sources: {context} 

        Extract the most relevant passage from the retrieved documents that answers the question.

        If the answer isn't there, say "I don't know." Do not use prior knowledge.
        """
    )
    retriever = RunnableLambda(retrieve)
    llm = load_llm(llm_model_name)
    rag_chain = (
        {
            "context": retriever | format_context,
            "query": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain


def answer_question(query, model_name):
    rag_chain = build_rag_chain(model_name)
    return rag_chain.invoke(query)