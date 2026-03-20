from sentence_transformers import CrossEncoder
import streamlit as st
from utils.utils import determine_device, determine_dtype, determine_model_path


@st.cache_resource
def load_reranker(reranker_model_name="BAAI/bge-reranker-base", reranker_models_root="reranker_models"):
    device = determine_device()
    dtype = determine_dtype(device)
    
    local_reranker_model_map = {
        "BAAI/bge-reranker-base": "bge-reranker-base",
    }
    reranker_model_path_str = determine_model_path(
        reranker_model_name,
        local_reranker_model_map,
        reranker_models_root,
    )

    reranker = CrossEncoder(
        reranker_model_path_str,
        device=device
    )

    return reranker


def rerank(query, docs, reranker_model_name="BAAI/bge-reranker-base", k_rerank=5):
    reranker = load_reranker(reranker_model_name)
    pairs = [(query, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs)

    reranked_docs = sorted(
        zip(docs, scores),
        key=lambda x: x[1],
        reverse=True,
    )
    return [doc for doc, _ in reranked_docs[: k_rerank]]