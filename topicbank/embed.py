import numpy as np
from sentence_transformers import SentenceTransformer

_model = None
def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer('all-MiniLM-L6-v2')
    return _model

def encode(texts):
    m = get_model()
    vecs = m.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
    return vecs.astype("float32")
