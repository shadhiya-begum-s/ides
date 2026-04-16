# vectorstore.py
import os
import json
import faiss
import numpy as np

DATA_DIR = os.getenv("VECTOR_STORE_PATH", "./faiss_index")
os.makedirs(DATA_DIR, exist_ok=True)
INDEX_PATH = os.path.join(DATA_DIR, "faiss_index.bin")
META_PATH = os.path.join(DATA_DIR, "metadata.json")

class FaissStore:
    def __init__(self, dim: int = 384):
        self.dim = int(dim)
        self.index = None
        self.metadatas = []
        self._load_or_initialize()

    def _load_or_initialize(self):
        if os.path.exists(INDEX_PATH) and os.path.exists(META_PATH):
            try:
                self.index = faiss.read_index(INDEX_PATH)
                with open(META_PATH, "r", encoding="utf-8") as f:
                    self.metadatas = json.load(f)
                return
            except Exception:
                pass
        self.index = faiss.IndexFlatIP(self.dim)
        self.metadatas = []

    def _save(self):
        os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
        faiss.write_index(self.index, INDEX_PATH)
        with open(META_PATH, "w", encoding="utf-8") as f:
            json.dump(self.metadatas, f, ensure_ascii=False, indent=2)

    def add_vectors(self, vectors: np.ndarray, metadatas: list):
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        faiss.normalize_L2(vectors)
        if vectors.shape[1] != self.dim:
            self.dim = vectors.shape[1]
            self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(vectors)
        self.metadatas.extend(metadatas)
        self._save()

    def search(self, query_vector: np.ndarray, top_k: int = 5):
        q = query_vector.reshape(1, -1).astype("float32")
        faiss.normalize_L2(q)
        D, I = self.index.search(q, top_k)
        results = []
        for score, idx in zip(D[0], I[0]):
            if idx < 0 or idx >= len(self.metadatas):
                continue
            results.append((self.metadatas[idx], float(score), int(idx)))
        return results

    def reset(self):
        self.index = faiss.IndexFlatIP(self.dim)
        self.metadatas = []
        self._save()