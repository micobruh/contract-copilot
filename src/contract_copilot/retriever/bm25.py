import streamlit as st
from ..config import config
from ..qdrant_sparse import SPARSE_VECTOR_NAME, encode_sparse_text
from .vector_search import load_vectorstore, qdrant_hit_to_document

try:
    from qdrant_client.http import models
except ImportError:
    models = None


@st.cache_resource
def load_bm25_retriever(k_bm25=config.k_bm25):
    return load_vectorstore()


def bm25_retrieval(query, k_bm25=config.k_bm25):
    if models is None:
        raise ImportError(
            "qdrant-client is required to run sparse retrieval from Qdrant. "
            "Install it with `pip install qdrant-client`."
        )

    bm25_retriever = load_bm25_retriever(k_bm25)
    indices, values = encode_sparse_text(query)
    search_result = bm25_retriever.query_points(
        collection_name=config.collection_name,
        query=models.SparseVector(
            indices=indices,
            values=values,
        ),
        using=SPARSE_VECTOR_NAME,
        limit=k_bm25,
        with_payload=True,
    )

    points = search_result.points if hasattr(search_result, "points") else search_result
    bm25_docs = [qdrant_hit_to_document(hit) for hit in points]
    return bm25_docs
