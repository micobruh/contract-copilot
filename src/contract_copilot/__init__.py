__version__ = "0.1.0"

from .config import (
    AppConfig,
    CONFIG_PATH,
    config,
    load_config,
)
from .indexer import build_qdrant_database
from .qdrant_sparse import (
    DENSE_VECTOR_NAME,
    SPARSE_VECTOR_NAME,
    encode_sparse_text,
)

__all__ = [
    "__version__",
    "AppConfig",
    "CONFIG_PATH",
    "DENSE_VECTOR_NAME",
    "SPARSE_VECTOR_NAME",
    "build_qdrant_database",
    "config",
    "encode_sparse_text",
    "load_config",
]
