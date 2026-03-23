from collections import defaultdict


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