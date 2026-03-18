from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
import pickle
from collections import defaultdict
from indexer.indexer import UniversalEmbeddingModel, LocalEmbeddingWrapper


def vector_similarity_search(query, model_name="BAAI/bge-m3", k_vector=10):
    # Retrieve from vector store
    embedding_model = UniversalEmbeddingModel(model_name)
    embedding_wrapper = LocalEmbeddingWrapper(embedding_model)
    vectorstore = Chroma(
        collection_name="Law_RAG",
        persist_directory="./chroma_db",
        embedding_function=embedding_wrapper,
    )
    vector_docs = vectorstore.similarity_search(query, k=k_vector)
    return vector_docs


def bm25_retrieval(query, k_bm25=10):
    # Retrieve from BM25
    with open("documents/docs.pkl", "rb") as f:
        docs = pickle.load(f)
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = k_bm25
    bm25_docs = bm25_retriever.invoke(query)
    return bm25_docs


def add_rrf_scores(docs, scores, doc_map, k_final=60):
    for rank, doc in enumerate(docs, start=1):
        doc_id = doc.metadata.get("doc_id") or doc.metadata.get("id") or doc.page_content
        scores[doc_id] += 1.0 / (k_final + rank)
        doc_map[doc_id] = doc    


def rrf_fuse(vector_docs, bm25_docs, k_final=60):
    """
    Reciprocal Rank Fusion.
    Higher score = better.
    """
    scores = defaultdict(float)
    doc_map = {}

    # Add scores from both retrievals
    add_rrf_scores(vector_docs, scores, doc_map, k_final)
    add_rrf_scores(bm25_docs, scores, doc_map, k_final)

    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_map[doc_id] for doc_id, _ in fused]


def retrieve(query, model_name="BAAI/bge-m3", k_vector=10, k_bm25=10, k_final=5):
    vector_docs = vector_similarity_search(query, model_name=model_name, k_vector=k_vector)
    bm25_docs = bm25_retrieval(query, k_bm25=k_bm25)
    fused_docs = rrf_fuse(vector_docs, bm25_docs, k_final=k_final)
    print(f"Total docs found: {len(vector_docs)} from vector search, {len(bm25_docs)} from BM25, fused to {len(fused_docs)}")
    return fused_docs[: k_final]