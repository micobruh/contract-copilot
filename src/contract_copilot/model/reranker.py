from sentence_transformers import CrossEncoder
import streamlit as st
from ..config import config
from ..utils.utils import determine_device, determine_dtype, determine_model_path


@st.cache_resource
def load_reranker(
    reranker_model_name=config.default_reranker_model_name,
    reranker_models_root=config.reranker_models_root,
):
    device = determine_device()
    dtype = determine_dtype(device)

    reranker_model_path_str = determine_model_path(
        reranker_model_name,
        config.local_reranker_model_map,
        reranker_models_root,
    )

    reranker = CrossEncoder(
        reranker_model_path_str,
        device=device
    )

    return reranker


def rerank(query, docs, reranker_model_name=config.default_reranker_model_name, k_rerank=config.k_rerank):
    reranker = load_reranker(reranker_model_name)
    pairs = [(query, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs)

    reranked_docs = sorted(
        zip(docs, scores),
        key=lambda x: x[1],
        reverse=True,
    )
    return [doc for doc, _ in reranked_docs[: k_rerank]]
