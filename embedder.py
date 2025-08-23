# In app/svc/embedder.py

import os
import sqlite3
import numpy as np
from pathlib import Path
from typing import List
from sentence_transformers import SentenceTransformer
from optimum.onnxruntime import ORTModelForFeatureExtraction

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = np.expand_dims(attention_mask, axis=-1).repeat(token_embeddings.shape[-1], axis=-1)
    sum_embeddings = np.sum(token_embeddings * input_mask_expanded, axis=1)
    sum_mask = np.clip(input_mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
    return sum_embeddings / sum_mask

class Embedder:
    def __init__(self, model_name: str = 'BAAI/bge-large-en-v1.5', db_path: str = "app/data/embedding_cache.sqlite"):
        self.model_name = model_name
        self.db_path = db_path
        self.model_dir = Path("app/models")
        self.onnx_path = self.model_dir / f"{self.model_name.split('/')[-1]}-onnx"
        
        self.model_dir.mkdir(exist_ok=True, parents=True)
        Path(os.path.dirname(self.db_path)).mkdir(exist_ok=True, parents=True)

        self._init_model()
        self._init_db()

    def _init_model(self):
        if not self.onnx_path.exists() or not list(self.onnx_path.glob("*.onnx")):
            print("ONNX model not found. Starting one-time export process...")
            original_model = SentenceTransformer(self.model_name)
            original_model.save(str(self.onnx_path))

            # Convert to ONNX using optimum
            original_model.save_pretrained(path=str(self.onnx_path))
            self.model = ORTModelForFeatureExtraction.from_pretrained(self.onnx_path, export=True)
            self.model.save_pretrained(save_directory=str(self.onnx_path))
            print("Export complete.")

        print("Loading ONNX model for inference...")
        self.model = ORTModelForFeatureExtraction.from_pretrained(self.onnx_path)
        self.tokenizer = SentenceTransformer(self.model_name).tokenizer

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("CREATE TABLE IF NOT EXISTS embeddings (sentence TEXT PRIMARY KEY, embedding BLOB)")
            conn.commit()

    def encode_cached(self, sentences: List[str]) -> np.ndarray:
        final_embeddings = {}
        new_sentences_to_encode = [s for s in set(sentences) if s] # Ensure unique and not empty
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            # Use a placeholder for a batch query
            placeholders = ','.join('?' for _ in new_sentences_to_encode)
            cursor.execute(f"SELECT sentence, embedding FROM embeddings WHERE sentence IN ({placeholders})", new_sentences_to_encode)
            for row in cursor.fetchall():
                sentence, embedding_blob = row
                final_embeddings[sentence] = np.frombuffer(embedding_blob, dtype=np.float32).reshape(1, -1)
                new_sentences_to_encode.remove(sentence)

        if new_sentences_to_encode:
            inputs = self.tokenizer(new_sentences_to_encode, padding=True, truncation=True, return_tensors='np')
            model_output = self.model(**inputs)
            pooled_output = mean_pooling(model_output, inputs['attention_mask'])
            new_embeddings = pooled_output / np.linalg.norm(pooled_output, axis=1, keepdims=True)
            new_embeddings = new_embeddings.astype(np.float32)

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                for sentence, embedding_array in zip(new_sentences_to_encode, new_embeddings):
                    embedding_blob = embedding_array.tobytes()
                    cursor.execute("INSERT OR IGNORE INTO embeddings (sentence, embedding) VALUES (?, ?)", (sentence, embedding_blob))
                    final_embeddings[sentence] = embedding_array.reshape(1, -1)
                conn.commit()

        ordered_embeddings = [final_embeddings[s] for s in sentences]
        return np.vstack(ordered_embeddings) if ordered_embeddings else np.array([])

embedder_service = Embedder()