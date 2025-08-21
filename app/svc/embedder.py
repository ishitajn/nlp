from sentence_transformers import SentenceTransformer
import numpy as np
import sqlite3
import os
from typing import List, Dict, Any

class Embedder:
    """
    A service to generate text embeddings with a persistent SQLite cache.
    """
    def __init__(self, model_name: str = 'BAAI/bge-small-en-v1.5', db_path: str = "app/data/embedding_cache.sqlite"):
        """
        Initializes the Embedder, loads the model, and sets up the SQLite cache.
        """
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

        # Ensure the data directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path

        # Set up the database connection and create the table if it doesn't exist
        self._init_db()

    def _init_db(self):
        """Initializes the database connection and table."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS embeddings (
                    sentence TEXT PRIMARY KEY,
                    embedding BLOB
                )
            """)
            conn.commit()

    def encode_cached(self, sentences: List[str]) -> np.ndarray:
        """
        Generates embeddings for a list of sentences, using a persistent SQLite
        cache to retrieve previously computed embeddings.
        """
        final_embeddings = {}
        new_sentences_to_encode = []

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            for sentence in sentences:
                cursor.execute("SELECT embedding FROM embeddings WHERE sentence = ?", (sentence,))
                result = cursor.fetchone()
                if result:
                    # Deserialize the embedding from BLOB to numpy array
                    final_embeddings[sentence] = np.frombuffer(result[0], dtype=np.float32).reshape(1, -1)
                else:
                    if sentence not in new_sentences_to_encode:
                        new_sentences_to_encode.append(sentence)

        if new_sentences_to_encode:
            print(f"Encoding {len(new_sentences_to_encode)} new sentences...")
            new_embeddings = self.model.encode(
                new_sentences_to_encode,
                normalize_embeddings=True
            ).astype(np.float32)

            # Add the new embeddings to the cache and the final result list
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                for sentence, embedding_array in zip(new_sentences_to_encode, new_embeddings):
                    # Serialize the numpy array to bytes
                    embedding_blob = embedding_array.tobytes()
                    cursor.execute("INSERT INTO embeddings (sentence, embedding) VALUES (?, ?)", (sentence, embedding_blob))
                    final_embeddings[sentence] = embedding_array.reshape(1, -1)
                conn.commit()

        # Return the embeddings in the original order
        ordered_embeddings = [final_embeddings[s] for s in sentences]

        if not ordered_embeddings:
             return np.array([])

        return np.vstack(ordered_embeddings)

# Global instance
embedder_service = Embedder()

def get_embedder():
    return embedder_service
