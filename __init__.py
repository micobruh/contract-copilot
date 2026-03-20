__version__ = "0.1.0"

from .indexer import (
    __version__,
    store_documents_in_batches,
    build_chroma_database,
    load_corpus,
    load_qa,
)
from .model import (
    UniversalEmbeddingModel,
    LocalEmbeddingWrapper,
    create_embedding_wrapper,
    load_reranker,
    rerank,
)

from .retriever import (
    vector_similarity_search,
    bm25_retrieval,
    add_rrf_scores,
    rrf_fuse,
    retrieve,
    format_context,
    answer_question,
)

from .utils import (
    determine_device,
    determine_dtype,
    determine_model_path,            
)

__all__ = [
    "__version__",
    "store_documents_in_batches",
    "build_chroma_database",
    "load_corpus",
    "load_qa",
    "UniversalEmbeddingModel",
    "LocalEmbeddingWrapper",
    "create_embedding_wrapper",
    "load_reranker",
    "rerank",
    "vector_similarity_search",
    "bm25_retrieval",
    "add_rrf_scores",
    "rrf_fuse",
    "retrieve",
    "format_context",
    "answer_question",
    "determine_device",
    "determine_dtype",
    "determine_model_path",
]