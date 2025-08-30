"""
Handles text embedding using a cached, ONNX-optimized sentence transformer.
Assumes the ONNX model has been built by `build_models.py`.
"""
import os
import logging
import sqlite3
import numpy as np
from pathlib import Path
from typing import List
from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def mean_pooling(model_output, attention_mask):
    """Performs mean pooling on token embeddings."""
    token_embeddings = model_output[0]
    input_mask_expanded = np.expand_dims(attention_mask, axis=-1).repeat(token_embeddings.shape[-1], axis=-1)
    sum_embeddings = np.sum(token_embeddings * input_mask_expanded, axis=1)
    sum_mask = np.clip(input_mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
    return sum_embeddings / sum_mask

class Embedder:
    """
    A service class for generating text embeddings, with ONNX optimization and SQLite caching.
    """
    def __init__(self, model_name: str = 'BAAI/bge-large-en-v1.5', db_path: str = "data/embedding_cache.sqlite"):
        self.model_name = model_name
        self.db_path = db_path
        self.model_dir = Path("models")
        self.onnx_path = self.model_dir / f"{self.model_name.split('/')[-1]}-onnx"
        
        self._init_model()
        self._init_db()

    def _init_model(self):
        """Initializes the model from the pre-built ONNX path."""
        if not self.onnx_path.exists() or not list(self.onnx_path.glob("*.onnx")):
            raise RuntimeError(
                f"ONNX model not found at {self.onnx_path}. "
                "Please run the `build_models.py` script first."
            )

        logging.info(f"Loading ONNX model for inference from {self.onnx_path}...")
        self.model = ORTModelForFeatureExtraction.from_pretrained(self.onnx_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.onnx_path)
        logging.info("ONNX model loaded successfully.")

    def _init_db(self):
        """Initializes the SQLite database for caching embeddings."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("CREATE TABLE IF NOT EXISTS embeddings (sentence TEXT PRIMARY KEY, embedding BLOB)")
            conn.commit()

    def encode_cached(self, sentences: List[str]) -> np.ndarray:
        """
        Encodes a list of sentences, using a cache to avoid re-computing existing embeddings.
        """
        if not sentences:
            return np.array([])

        unique_sentences = list(set(s for s in sentences if s))
        final_embeddings = {}
        
        # 1. Check cache for existing sentences
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                placeholders = ','.join('?' for _ in unique_sentences)
                if placeholders:
                    cursor.execute(f"SELECT sentence, embedding FROM embeddings WHERE sentence IN ({placeholders})", unique_sentences)
                    for sentence, blob in cursor.fetchall():
                        final_embeddings[sentence] = np.frombuffer(blob, dtype=np.float32)
        except sqlite3.Error as e:
            logging.error(f"Error reading from embedding cache: {e}")

        # 2. Encode sentences that were not in the cache
        new_sentences_to_encode = [s for s in unique_sentences if s not in final_embeddings]
        if new_sentences_to_encode:
            inputs = self.tokenizer(new_sentences_to_encode, padding=True, truncation=True, return_tensors='np')
            model_output = self.model(**inputs)
            pooled_output = mean_pooling(model_output, inputs['attention_mask'])
            new_embeddings = pooled_output / np.linalg.norm(pooled_output, axis=1, keepdims=True)

            db_insert_data = []
            for i, sentence in enumerate(new_sentences_to_encode):
                embedding = new_embeddings[i].astype(np.float32)
                final_embeddings[sentence] = embedding
                db_insert_data.append((sentence, embedding.tobytes()))

            # 3. Write new embeddings to cache
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.executemany("INSERT OR IGNORE INTO embeddings (sentence, embedding) VALUES (?, ?)", db_insert_data)
                    conn.commit()
            except sqlite3.Error as e:
                logging.error(f"Error writing to embedding cache: {e}")

        # 4. Return embeddings in the original order
        ordered_embeddings = [final_embeddings[s] for s in sentences if s in final_embeddings]
        return np.vstack(ordered_embeddings) if ordered_embeddings else np.array([])