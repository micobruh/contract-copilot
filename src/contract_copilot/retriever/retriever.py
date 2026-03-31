from .vector_search import vector_similarity_search
from .bm25 import bm25_retrieval
from .hybrid_search import hybrid_retrieval
from ..config import config
from ..model.reranker import rerank


def retrieve(
    query,
    embedding_model_name=config.default_embedding_model_name,
    reranker_model_name=config.default_reranker_model_name,
    k_vector=config.k_vector,
    k_bm25=config.k_bm25,
    k_rrf=config.k_rrf,
    k_rerank=config.k_rerank,
    search_method=config.search_method,
    rerank_needed=config.rerank_needed,
):
    if search_method == "vector":
        docs_before_ranking = vector_similarity_search(query, embedding_model_name=embedding_model_name, k_vector=k_vector)
    elif search_method == "bm25":
        docs_before_ranking = bm25_retrieval(query, k_bm25=k_bm25)
    elif search_method == "hybrid":    
        docs_before_ranking = hybrid_retrieval(query, embedding_model_name=embedding_model_name, k_rrf=k_rrf)
    else:
        raise ValueError(f"Invalid search method: {search_method}")     
    
    if rerank_needed:
        return rerank(query, docs_before_ranking, reranker_model_name=reranker_model_name, k_rerank=k_rerank)
    return docs_before_ranking


def format_context(docs):
    parts = []
    for i, doc in enumerate(docs, 1):
        title = doc.metadata.get("title") or doc.metadata.get("document_title") or "Unknown"
        source = doc.metadata.get("source") or doc.metadata.get("source_path") or ""
        parts.append(
            f"[{i}] Title: {title}\n"
            f"Source: {source}\n"
            f"Text: {doc.page_content.strip()}"
        )
    return "\n\n---\n\n".join(parts)
