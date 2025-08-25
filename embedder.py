# In embedder.py
"""
Handles text embedding using a cached, ONNX-optimized sentence transformer.
"""
import os
import logging
import sqlite3
import numpy as np
from pathlib import Path
from typing import List
from sentence_transformers import SentenceTransformer
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
        """
        Initializes the Embedder service.

        Args:
            model_name: The name of the SentenceTransformer model to use.
            db_path: The file path for the SQLite cache database.
        """
        self.model_name = model_name
        self.db_path = db_path
        self.model_dir = Path("models")
        self.onnx_path = self.model_dir / f"{self.model_name.split('/')[-1]}-onnx"
        
        self.model_dir.mkdir(exist_ok=True, parents=True)
        Path(os.path.dirname(self.db_path)).mkdir(exist_ok=True, parents=True)

        self._init_model()
        self._init_db()

    def _init_model(self):
        """Initializes the model, converting to ONNX format if it doesn't exist."""
        if not self.onnx_path.exists() or not list(self.onnx_path.glob("*.onnx")):
            logging.info("ONNX model not found. Starting one-time export process...")
            original_model = SentenceTransformer(self.model_name)
            original_model.save(str(self.onnx_path))

            original_model.save_pretrained(path=str(self.onnx_path))
            self.model = ORTModelForFeatureExtraction.from_pretrained(self.onnx_path, export=True)
            self.model.save_pretrained(save_directory=str(self.onnx_path))
            logging.info("Export complete.")

        logging.info("Loading ONNX model for inference...")
        self.model = ORTModelForFeatureExtraction.from_pretrained(self.onnx_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.onnx_path)

    def _init_db(self):
        """Initializes the SQLite database for caching embeddings."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("CREATE TABLE IF NOT EXISTS embeddings (sentence TEXT PRIMARY KEY, embedding BLOB)")
            conn.commit()

    def encode_cached(self, sentences: List[str]) -> np.ndarray:
        """
        Encodes a list of sentences, using a cache to avoid re-computing existing embeddings.

        Args:
            sentences: A list of strings to encode.

        Returns:
            A numpy array of embeddings, ordered according to the input list.
        """
        if not sentences:
            return np.array([])

        final_embeddings = {}
        unique_sentences = set(s for s in sentences if s)
        
        cached_sentences = {}
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            placeholders = ','.join('?' for _ in unique_sentences)
            if placeholders:
                cursor.execute(f"SELECT sentence, embedding FROM embeddings WHERE sentence IN ({placeholders})", list(unique_sentences))
                for sentence, embedding_blob in cursor.fetchall():
                    cached_sentences[sentence] = np.frombuffer(embedding_blob, dtype=np.float32).reshape(1, -1)

        final_embeddings.update(cached_sentences)

        new_sentences_to_encode = list(unique_sentences - set(cached_sentences.keys()))

        if new_sentences_to_encode:
            inputs = self.tokenizer(new_sentences_to_encode, padding=True, truncation=True, return_tensors='np')
            model_output = self.model(**inputs)
            pooled_output = mean_pooling(model_output, inputs['attention_mask'])
            new_embeddings = pooled_output / np.linalg.norm(pooled_output, axis=1, keepdims=True)
            new_embeddings = new_embeddings.astype(np.float32)

            db_insert_data = []
            for sentence, embedding_array in zip(new_sentences_to_encode, new_embeddings):
                final_embeddings[sentence] = embedding_array.reshape(1, -1)
                db_insert_data.append((sentence, embedding_array.tobytes()))

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.executemany("INSERT OR IGNORE INTO embeddings (sentence, embedding) VALUES (?, ?)", db_insert_data)
                conn.commit()

        ordered_embeddings = [final_embeddings[s] for s in sentences if s in final_embeddings]
        return np.vstack(ordered_embeddings) if ordered_embeddings else np.array([])

embedder_service = Embedder()