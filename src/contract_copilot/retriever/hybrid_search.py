from collections import defaultdict
from ..config import config
from ..model.embeddings import create_embedding_wrapper
from ..qdrant_sparse import DENSE_VECTOR_NAME, SPARSE_VECTOR_NAME, encode_sparse_text
from .vector_search import load_vectorstore, qdrant_hit_to_document

try:
    from qdrant_client.http import models
except ImportError:
    models = None


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


def hybrid_retrieval(query, embedding_model_name=config.default_embedding_model_name, k_rrf=config.k_rrf):
    if models is None:
        raise ImportError(
            "qdrant-client is required to run hybrid retrieval from Qdrant. "
            "Install it with `pip install qdrant-client`."
        )

    new_query = query if embedding_model_name == "jinaai/jina-embeddings-v5-text-small" else "query: " + query
    embedding_wrapper = create_embedding_wrapper(embedding_model_name)
    dense_query_vector = embedding_wrapper.embed_query(new_query)
    sparse_indices, sparse_values = encode_sparse_text(query)

    vectorstore = load_vectorstore(embedding_model_name)
    search_result = vectorstore.query_points(
        collection_name=config.collection_name,
        prefetch=[
            models.Prefetch(
                query=models.SparseVector(
                    indices=sparse_indices,
                    values=sparse_values,
                ),
                using=SPARSE_VECTOR_NAME,
                limit=k_rrf,
            ),
            models.Prefetch(
                query=dense_query_vector,
                using=DENSE_VECTOR_NAME,
                limit=k_rrf,
            ),
        ],
        query=models.FusionQuery(
            fusion=models.Fusion.RRF,
        ),
        limit=k_rrf,
        with_payload=True,
    )

    points = search_result.points if hasattr(search_result, "points") else search_result
    return [qdrant_hit_to_document(hit) for hit in points]
