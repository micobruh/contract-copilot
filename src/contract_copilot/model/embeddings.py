import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
from langchain_core.embeddings import Embeddings
import streamlit as st
from ..config import config
from ..utils.utils import (
    configure_torch_threads,
    determine_device,
    determine_dtype,
    determine_model_path,
)


class UniversalEmbeddingModel:
    def __init__(self, embedding_model_name, embedding_models_root=config.embedding_models_root):
        self.embedding_model_name = embedding_model_name

        self.device = determine_device()
        self.dtype = determine_dtype(self.device)
        configure_torch_threads(
            self.device,
            config.cpu_num_threads,
            config.cpu_num_interop_threads,
        )

        self.embedding_model_path_str = determine_model_path(
            self.embedding_model_name,
            config.local_embedding_model_map,
            embedding_models_root,
        )

        if self.embedding_model_name == "BAAI/bge-m3":
            self.backend = "sentence_transformer"
            self.model = SentenceTransformer(
                self.embedding_model_path_str,
                device=self.device,
                local_files_only=True,
            )

        elif self.embedding_model_name == "Qwen/Qwen3-Embedding-4B":
            self.backend = "sentence_transformer"
            self.model = SentenceTransformer(
                self.embedding_model_path_str,
                device=self.device,
                local_files_only=True,
                model_kwargs={"dtype": self.dtype},
                tokenizer_kwargs={"padding_side": "left"},
            )

        elif self.embedding_model_name == "jinaai/jina-embeddings-v5-text-small":
            self.backend = "transformers"
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.embedding_model_path_str,
                trust_remote_code=True,
                local_files_only=True,
            )
            self.model = AutoModel.from_pretrained(
                self.embedding_model_path_str,
                trust_remote_code=True,
                dtype=self.dtype,
                local_files_only=True,
            ).to(self.device)
            self.model.eval()

    def encode_documents(self, texts):
        if isinstance(texts, str):
            texts = [texts]

        if self.backend == "sentence_transformer":
            if self.embedding_model_name == "Qwen/Qwen3-Embedding-4B":
                return self.model.encode(
                    texts,
                    prompt_name="document",
                    batch_size=config.embedding_batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                )
            return self.model.encode(
                texts,
                batch_size=config.embedding_batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )

        elif self.backend == "transformers":
            with torch.inference_mode():
                return self.model.encode(
                    texts=texts,
                    task="retrieval",
                    prompt_name="document",
                )

    def encode_query(self, text):
        if self.backend == "sentence_transformer":
            encode_kwargs = {
                "batch_size": config.embedding_batch_size,
                "show_progress_bar": False,
                "convert_to_numpy": True,
            }
            if self.embedding_model_name == "Qwen/Qwen3-Embedding-4B":
                encode_kwargs["prompt_name"] = "query"
            return self.model.encode([text], **encode_kwargs)

        with torch.inference_mode():
            return self.model.encode(
                texts=[text],
                task="retrieval",
                prompt_name="query",
            )


class LocalEmbeddingWrapper(Embeddings):
    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts):
        embeddings = self.model.encode_documents(texts)
        return embeddings.tolist()

    def embed_query(self, text):
        embedding = self.model.encode_query(text)[0]
        return embedding.tolist()
    

@st.cache_resource
def create_embedding_wrapper(embedding_model_name=config.default_embedding_model_name):
    embedding_model = UniversalEmbeddingModel(embedding_model_name)
    embedding_wrapper = LocalEmbeddingWrapper(embedding_model)
    return embedding_wrapper
