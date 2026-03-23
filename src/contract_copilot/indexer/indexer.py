from tqdm import tqdm
import pickle
from langchain_core.documents import Document
from langchain_chroma import Chroma
from .ocr_loader import load_corpus
from ..config import config
from ..model.embeddings import create_embedding_wrapper


def store_documents_in_batches(chunks, embedding_wrapper, storing_batch_size=config.storing_batch_size):
    total_chunks = len(chunks)
    successful_writes = 0
    failed_writes = 0

    print(f"Starting to process {total_chunks} chunks")
    print(f"Batch size: {storing_batch_size}")
    print(f"Using custom embedding model: {embedding_wrapper.model.embedding_model_name}")

    documents = [
        Document(
            page_content=chunk["text"],
            metadata={
                "id": chunk["id"],
                "title": chunk["title"],
                "footnotes": chunk["footnotes"] if chunk["footnotes"] else "",
                "source": "Judicial College of Victoria's Criminal Charge Book"
            },
            id=chunk["id"]
        )
        for chunk in chunks
    ]
    with open(config.bm25_doc_link, "wb") as f:
        pickle.dump(documents, f)        

    vectorstore = Chroma(
        collection_name=config.collection_name,
        embedding_function=embedding_wrapper,
        persist_directory=config.chroma_db_link
    )

    for min_document_index in tqdm(range(0, len(documents), storing_batch_size), desc="Processing document batches"):
        try:
            max_document_index = min(min_document_index + storing_batch_size, total_chunks)
            vectorstore.add_documents(documents[min_document_index: max_document_index])
            successful_writes += max_document_index - min_document_index
        except Exception as e:
            failed_writes += max_document_index - min_document_index

    print(f"Finished processing chunks. Successful writes: {successful_writes}, Failed writes: {failed_writes}")        
    return successful_writes, failed_writes


def build_chroma_database(embedding_model_name=config.default_embedding_model_name):
    legal_corpus_chunks = load_corpus()
    print("Loaded legal corpus chunks, total chunks:", len(legal_corpus_chunks))

    embedding_wrapper = create_embedding_wrapper(embedding_model_name)

    successful, failed = store_documents_in_batches(
        chunks=legal_corpus_chunks,
        embedding_wrapper=embedding_wrapper,
    )

    return successful, failed
