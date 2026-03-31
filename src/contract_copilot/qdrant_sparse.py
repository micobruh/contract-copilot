import hashlib
import math
import re
from collections import Counter, defaultdict


DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"
SPARSE_HASH_BUCKETS = 1 << 20
TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def tokenize_sparse_text(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def hash_token(token: str, bucket_count: int = SPARSE_HASH_BUCKETS) -> int:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big") % bucket_count


def encode_sparse_text(text: str, bucket_count: int = SPARSE_HASH_BUCKETS) -> tuple[list[int], list[float]]:
    token_counts = Counter(tokenize_sparse_text(text))
    sparse_values = defaultdict(float)

    for token, count in token_counts.items():
        sparse_values[hash_token(token, bucket_count=bucket_count)] += 1.0 + math.log(count)

    sorted_items = sorted(sparse_values.items())
    indices = [index for index, _ in sorted_items]
    values = [value for _, value in sorted_items]
    return indices, values
