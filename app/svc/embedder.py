import os
import sqlite3
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

# Optimum and ONNX libraries for quantization and inference
from optimum.onnxruntime import ORTQuantizer, ORTModelForFeatureExtraction
from optimum.onnxruntime.configuration import AutoQuantizationConfig

# We still need SentenceTransformer for the initial conversion
from sentence_transformers import SentenceTransformer

def mean_pooling(model_output, attention_mask):
    """Helper function for pooling token embeddings."""
    token_embeddings = model_output[0]
    input_mask_expanded = np.expand_dims(attention_mask, axis=-1).repeat(token_embeddings.shape[-1], axis=-1)
    sum_embeddings = np.sum(token_embeddings * input_mask_expanded, axis=1)
    sum_mask = np.clip(input_mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
    return sum_embeddings / sum_mask

class Embedder:
    """
    A service to generate text embeddings using a quantized INT8 ONNX model.
    Includes a persistent SQLite cache.
    """
    def __init__(self, model_name: str = 'BAAI/bge-small-en-v1.5', db_path: str = "app/data/embedding_cache.sqlite"):
        self.model_name = model_name
        self.db_path = db_path
        self.model_dir = Path("app/models")
        self.onnx_path = self.model_dir / f"{self.model_name.split('/')[-1]}-onnx"
        self.quantized_path = self.onnx_path / "model_quantized.onnx"

        self.model_dir.mkdir(exist_ok=True)
        self.onnx_path.mkdir(exist_ok=True)

        self._init_model()
        self._init_db()

    def _init_model(self):
        """
        Initializes the quantized model. If the quantized model file doesn't exist,
        it performs a one-time conversion and quantization process.
        """
        if not self.quantized_path.exists():
            print("Quantized model not found. Starting one-time conversion process...")
            # 1. Export the base model to ONNX
            print("Exporting base model to ONNX...")
            original_model = SentenceTransformer(self.model_name)
            original_model.save(str(self.onnx_path))

            # 2. Create the INT8 quantized model from the ONNX export
            print("Quantizing ONNX model to INT8...")
            onnx_model = ORTModelForFeatureExtraction.from_pretrained(self.onnx_path)
            quantizer = ORTQuantizer.from_pretrained(onnx_model)
            dqconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)
            quantizer.quantize(save_dir=self.onnx_path, quantization_config=dqconfig)
            # The quantized model is saved as model_quantized.onnx by default in the same dir
            print("Quantization complete.")

        print("Loading quantized INT8 model for inference...")
        self.model = ORTModelForFeatureExtraction.from_pretrained(self.onnx_path)
        # We also need the tokenizer from the original model
        self.tokenizer = SentenceTransformer(self.model_name).tokenizer


    def _init_db(self):
        """Initializes the database connection and table."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("CREATE TABLE IF NOT EXISTS embeddings (sentence TEXT PRIMARY KEY, embedding BLOB)")
            conn.commit()

    def encode_cached(self, sentences: List[str]) -> np.ndarray:
        """
        Generates embeddings using the quantized model, with SQLite caching.
        """
        final_embeddings = {}
        new_sentences_to_encode = []

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            for sentence in sentences:
                cursor.execute("SELECT embedding FROM embeddings WHERE sentence = ?", (sentence,))
                result = cursor.fetchone()
                if result:
                    final_embeddings[sentence] = np.frombuffer(result[0], dtype=np.float32).reshape(1, -1)
                else:
                    if sentence not in new_sentences_to_encode:
                        new_sentences_to_encode.append(sentence)

        if new_sentences_to_encode:
            print(f"Encoding {len(new_sentences_to_encode)} new sentences with quantized model...")

            # Tokenize sentences
            inputs = self.tokenizer(new_sentences_to_encode, padding=True, truncation=True, return_tensors='np')

            # Get model outputs using the ONNX runtime
            model_output = self.model(**inputs)

            # Perform mean pooling
            pooled_output = mean_pooling(model_output, inputs['attention_mask'])

            # Normalize embeddings
            new_embeddings = pooled_output / np.linalg.norm(pooled_output, axis=1, keepdims=True)
            new_embeddings = new_embeddings.astype(np.float32)

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                for sentence, embedding_array in zip(new_sentences_to_encode, new_embeddings):
                    embedding_blob = embedding_array.tobytes()
                    cursor.execute("INSERT INTO embeddings (sentence, embedding) VALUES (?, ?)", (sentence, embedding_blob))
                    final_embeddings[sentence] = embedding_array.reshape(1, -1)
                conn.commit()

        ordered_embeddings = [final_embeddings[s] for s in sentences]
        return np.vstack(ordered_embeddings) if ordered_embeddings else np.array([])

# Global instance
embedder_service = Embedder()

def get_embedder():
    return embedder_service
