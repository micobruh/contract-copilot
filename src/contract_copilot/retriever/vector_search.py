import streamlit as st
from langchain_core.documents import Document
from ..config import config
from ..model.embeddings import create_embedding_wrapper
from ..qdrant_sparse import DENSE_VECTOR_NAME

try:
    from qdrant_client import QdrantClient
except ImportError:
    QdrantClient = None


def qdrant_hit_to_document(hit):
    payload = hit.payload or {}
    metadata = {key: value for key, value in payload.items() if key != "page_content"}
    return Document(
        id=str(getattr(hit, "id", metadata.get("chunk_id", ""))),
        page_content=payload.get("page_content", ""),
        metadata=metadata,
    )


@st.cache_resource
def load_vectorstore(embedding_model_name=config.default_embedding_model_name):
    if QdrantClient is None:
        raise ImportError(
            "qdrant-client is required to load vectors from Qdrant. "
            "Install it with `pip install qdrant-client`."
        )

    return QdrantClient(path=config.qdrant_db_link)


def vector_similarity_search(query, embedding_model_name=config.default_embedding_model_name, k_vector=config.k_vector):
    new_query = query if embedding_model_name == "jinaai/jina-embeddings-v5-text-small" else "query: " + query
    embedding_wrapper = create_embedding_wrapper(embedding_model_name)
    query_vector = embedding_wrapper.embed_query(new_query)

    vectorstore = load_vectorstore(embedding_model_name)
    search_result = vectorstore.query_points(
        collection_name=config.collection_name,
        query=query_vector,
        using=DENSE_VECTOR_NAME,
        limit=k_vector,
        with_payload=True,
    )

    points = search_result.points if hasattr(search_result, "points") else search_result
    vector_docs = [qdrant_hit_to_document(hit) for hit in points]
    return vector_docs
