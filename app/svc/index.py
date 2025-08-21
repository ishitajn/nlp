import faiss
import numpy as np
from typing import List, Dict, Any, Tuple

class FaissIndex:
    """
    A wrapper for a FAISS HNSW index to perform efficient similarity searches
    on conversation embeddings.
    """
    def __init__(self, embedding_dim: int = 384):
        """
        Initializes the FAISS index.

        Args:
            embedding_dim: The dimensionality of the vectors to be indexed.
                           For 'bge-small-en-v1.5', this is 384.
        """
        self.embedding_dim = embedding_dim
        # Using HNSW (Hierarchical Navigable Small World) for its balance of speed and accuracy.
        # It's good for production systems where search speed is important.
        # The index is not memory-mapped as per the prompt, but this can be done
        # by saving/loading the index to/from disk for persistence across restarts.
        self.index = faiss.IndexHNSWFlat(self.embedding_dim, 32)  # 32 is the number of neighbors
        self.index.hnsw.efSearch = 128 # efSearch controls the trade-off between speed and accuracy

        # We need a way to map the index's internal IDs back to our conversation turns.
        self.id_to_turn_map: Dict[int, Dict[str, Any]] = {}
        self.next_id = 0

    def ensure_added(self, turns: List[Dict[str, Any]], vectors: np.ndarray):
        """
        Adds new turns and their vectors to the index if they are not already present.

        Note: This is a simple implementation that assumes a stateless index for each
        API call. A production system would need a persistent index and a way to
        uniquely identify turns across calls (e.g., using a database ID).
        """
        if vectors.shape[0] == 0:
            return

        # For this stateless implementation, we clear and rebuild the index for each call.
        self.index.reset()
        self.id_to_turn_map.clear()
        self.next_id = 0

        # Add vectors to the index
        self.index.add(vectors)

        # Map IDs to turns
        for i, turn in enumerate(turns):
            self.id_to_turn_map[i] = turn
        self.next_id = len(turns)

    def search(self, query_vector: np.ndarray, k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        """
        Performs a similarity search on the index.

        Args:
            query_vector: The vector to search for. Must be 2D array.
            k: The number of nearest neighbors to return.

        Returns:
            A list of tuples, where each tuple contains the similar turn and its distance.
        """
        if self.index.ntotal == 0:
            return []

        distances, indices = self.index.search(query_vector, k)

        results = []
        for i in range(k):
            idx = indices[0][i]
            if idx != -1: # FAISS returns -1 for invalid indices
                turn = self.id_to_turn_map.get(idx)
                if turn:
                    results.append((turn, distances[0][i]))
        return results

# Global instance for the service
faiss_index_service = FaissIndex()

def get_faiss_index():
    """Dependency injection getter for the FAISS index service."""
    return faiss_index_service
