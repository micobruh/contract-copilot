# Notice that this database is already preprocessed and contains chunked passages extracted from the original documents using OCR. 
# The passages are stored in the "text" field of the dataset. 
# You can access the passages as follows:

import datasets

def load_corpus():
    # Load passages in Legal RAG Bench.
    corpus = datasets.load_dataset("isaacus/legal-rag-bench", name="corpus", split="test")
    return corpus

def load_qa():
    # Load question-answer-passage triplets from Legal RAG Bench.
    qa = datasets.load_dataset("isaacus/legal-rag-bench", name="qa", split="test")
    return qa