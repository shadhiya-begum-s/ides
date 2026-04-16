#retriever.py 
import os
import numpy as np
from typing import List, Tuple
from embeddings import embed_texts
from vectorstore import FaissStore

DEFAULT_DIM = int(os.getenv("EMBED_DIM", 384))
_store = None

def get_store() -> FaissStore:
    """Get or initialize the FAISS store."""
    global _store
    if _store is None:
        _store = FaissStore(dim=DEFAULT_DIM)
    return _store

def reset_store():
    """Reset the FAISS index and metadata."""
    global _store
    _store = FaissStore(dim=DEFAULT_DIM)
    _store.reset()

def index_document_chunks(
    documents: List[dict],
    chunker,
    chunk_size: int = 800,
    overlap: int = 100
) -> int:
    """
    Index document chunks into FAISS vector store.
    Returns the number of chunks indexed.
    """
    store = get_store()
    all_meta = []

    for doc in documents:
        chunks = chunker(doc["text"], chunk_size=chunk_size, overlap=overlap)
        for idx, ch in enumerate(chunks):
            meta = {
                "doc_id": doc["id"],
                "source": doc.get("name", "unknown"),
                "chunk_index": ch.get("chunk_index", idx),
                "page": ch.get("page"),
                "text": ch.get("text"),
            }
            all_meta.append(meta)

    if not all_meta:
        return 0

    texts = [m["text"] for m in all_meta]
    batch_size = 64
    vecs = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        emb = embed_texts(batch, batch_size=batch_size)
        vecs.append(emb)

    if vecs:
        vecs = np.vstack(vecs).astype("float32")
        store.add_vectors(vecs, all_meta)
        return len(all_meta)

    return 0
def retrieve(query: str, top_k: int = 5) -> List[Tuple[dict, float, int]]:
    """
    Retrieve top_k relevant document chunks for a query.
    Returns a list of tuples: (metadata, similarity_score, chunk_index)
    """
    store = get_store()
    qemb = embed_texts([query])[0].astype("float32")
    results = store.search(qemb, top_k=top_k)
    return results