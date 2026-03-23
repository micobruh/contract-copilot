from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml


@dataclass(frozen=True)
class AppConfig:
    storing_batch_size: int
    embedding_batch_size: int
    rerank_batch_size: int
    collection_name: str
    bm25_doc_link: str
    chroma_db_link: str
    embedding_models_root: str
    reranker_models_root: str
    default_embedding_model_name: str
    default_reranker_model_name: str
    default_llm_model_name: str
    local_embedding_model_map: dict[str, str]
    local_reranker_model_map: dict[str, str]
    k_vector: int
    k_bm25: int
    k_rrf: int
    k_rerank: int
    rerank_needed: bool


CONFIG_PATH = Path(__file__).with_name("config.yaml")


@lru_cache(maxsize=1)
def load_config(path: Path | None = None) -> AppConfig:
    config_path = path or CONFIG_PATH
    with config_path.open("r", encoding="utf-8") as config_file:
        raw_config = yaml.safe_load(config_file)
    return AppConfig(**raw_config)


config = load_config()
