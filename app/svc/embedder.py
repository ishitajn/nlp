from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Any

class Embedder:
    """
    A service to generate text embeddings using a sentence transformer model.
    It includes an in-memory cache to avoid re-computing embeddings for the same text.
    """
    def __init__(self, model_name: str = 'BAAI/bge-small-en-v1.5'):
        """
        Initializes the Embedder and loads the sentence transformer model.

        Args:
            model_name: The name of the model to load from Hugging Face.
        """
        # In a production system, model loading should be handled carefully.
        # For instance, the model could be downloaded during the build process
        # or loaded into memory once when the application starts.
        self.model = SentenceTransformer(model_name)
        self.cache: Dict[str, np.ndarray] = {}
        # For a more persistent cache, a database like SQLite could be used, as suggested
        # in the architecture document.
        # e.g., self.cache = Cache(db_path='/app/data/embedding_cache.db')

    def encode_cached(self, sentences: List[str]) -> np.ndarray:
        """
        Generates embeddings for a list of sentences, using a cache to retrieve
        previously computed embeddings.

        Args:
            sentences: A list of strings to be encoded.

        Returns:
            A numpy array of shape (n_sentences, embedding_dim) containing the embeddings.
        """
        # Find which sentences are not in the cache
        new_sentences = [s for s in sentences if s not in self.cache]

        if new_sentences:
            print(f"Encoding {len(new_sentences)} new sentences...")
            # Generate embeddings for the new sentences
            # The prompt mentions int8 quantization. For sentence-transformers, this is not
            # a direct option, but techniques like `model.encode(..., convert_to_tensor=True, normalize_embeddings=True)`
            # followed by quantization through a library like `optimum` or custom PyTorch code
            # would be the way to go. For now, we use standard float32 embeddings.
            new_embeddings = self.model.encode(new_sentences, normalize_embeddings=True)

            # Add the new embeddings to the cache
            for sentence, embedding in zip(new_sentences, new_embeddings):
                self.cache[sentence] = embedding

        # Retrieve all embeddings (from cache or newly computed)
        final_embeddings = [self.cache[s] for s in sentences]

        return np.array(final_embeddings)

# It's good practice to instantiate the service once and reuse it.
# In a FastAPI app, this can be managed with dependency injection.
# For now, we can create a global instance.
embedder_service = Embedder()

def get_embedder():
    """Dependency injection getter for the embedder service."""
    return embedder_service
