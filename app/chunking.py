"""
app/chunking.py

Token-aware chunking with overlap. Uses tiktoken for token counting
(cl100k_base — close enough for our purposes even though we're not using
an OpenAI model for generation; it's just a consistent tokenizer for sizing).
"""

import tiktoken

ENCODER = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(ENCODER.encode(text))


def chunk_text(text: str, chunk_size: int = 512, overlap: int = 50):
    """
    Splits text into overlapping chunks measured in tokens.

    Returns a list of chunk strings.
    """
    tokens = ENCODER.encode(text)
    if len(tokens) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunks.append(ENCODER.decode(chunk_tokens))
        if end == len(tokens):
            break
        start += chunk_size - overlap

    return chunks


def chunk_document(doc: dict, chunk_size: int = 512, overlap: int = 50):
    """
    doc: {"title": str, "url": str, "content": str}
    Returns a list of chunk dicts with metadata, ready for embedding/indexing.
    """
    raw_chunks = chunk_text(doc["content"], chunk_size=chunk_size, overlap=overlap)
    chunk_dicts = []
    for i, chunk in enumerate(raw_chunks):
        chunk_dicts.append({
            "id": f"{doc['title']}::chunk_{i}",
            "text": chunk,
            "source_title": doc["title"],
            "source_url": doc["url"],
            "chunk_index": i,
            "n_tokens": count_tokens(chunk),
        })
    return chunk_dicts
