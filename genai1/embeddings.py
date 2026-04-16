# embeddings.py
import os
import numpy as np

_MODEL = None
EMBED_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
USE_OPENAI_EMB = os.getenv("USE_OPENAI_EMBEDDINGS", "false").lower() in ("1", "true", "yes")

def load_sentence_transformer():
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer
        _MODEL = SentenceTransformer(EMBED_MODEL_NAME)
    return _MODEL

def embed_texts(texts, batch_size: int = 32):
    if not texts:
        return np.zeros((0, 0), dtype="float32")

    if USE_OPENAI_EMB:
        from openai import OpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        client = OpenAI(api_key=api_key)
        embs = []
        for t in texts:
            resp = client.embeddings.create(model="text-embedding-3-small", input=t)
            embs.append(resp.data[0].embedding)
        return np.array(embs, dtype="float32")

    model = load_sentence_transformer()
    emb = model.encode(texts, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True)
    return emb.astype("float32")