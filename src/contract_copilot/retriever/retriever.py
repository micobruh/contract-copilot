from .vector_search import vector_similarity_search
from .bm25 import bm25_retrieval
from .hybrid_search import rrf_fuse
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
    rerank_needed=config.rerank_needed,
):
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
