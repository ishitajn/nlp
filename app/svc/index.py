import faiss
import numpy as np
import os
import json
import hashlib
from typing import List, Dict, Any, Tuple

class FaissIndex:
    """
    A wrapper for a persistent, global FAISS HNSW index. The index and its
    mappings are loaded from disk on startup and saved after modification.
    """
    def __init__(self, embedding_dim: int = 384, index_dir: str = "app/index"):
        self.embedding_dim = embedding_dim
        self.index_path = os.path.join(index_dir, "global.faiss")
        self.map_path = os.path.join(index_dir, "global_map.json")
        self.keys_path = os.path.join(index_dir, "added_keys.json")

        os.makedirs(index_dir, exist_ok=True)
        self._load()

    def _load(self):
        """Loads the index and associated data from disk if they exist."""
        if os.path.exists(self.index_path):
            print("Loading persistent FAISS index from disk...")
            self.index = faiss.read_index(self.index_path)
            with open(self.map_path, 'r') as f:
                self.id_to_turn_map = {int(k): v for k, v in json.load(f).items()}
            with open(self.keys_path, 'r') as f:
                self.added_keys = set(json.load(f))
            self.next_id = self.index.ntotal
        else:
            print("No persistent FAISS index found, initializing a new one.")
            self.index = faiss.IndexHNSWFlat(self.embedding_dim, 32)
            self.index.hnsw.efSearch = 128
            self.id_to_turn_map = {}
            self.added_keys = set()
            self.next_id = 0

    def _save(self):
        """Saves the index and associated data to disk."""
        print(f"Saving FAISS index with {self.index.ntotal} vectors to disk...")
        faiss.write_index(self.index, self.index_path)
        with open(self.map_path, 'w') as f:
            json.dump(self.id_to_turn_map, f)
        with open(self.keys_path, 'w') as f:
            json.dump(list(self.added_keys), f)

    def _generate_turn_key(self, turn: Dict[str, Any], match_id: str) -> str:
        """Creates a deterministic, unique identifier for a conversation turn."""
        content = turn.get('content', '')
        # Use a deterministic hash function like SHA256 instead of the built-in hash()
        content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()
        return f"{match_id}|{turn.get('role')}|{turn.get('date')}|{content_hash}"

    def ensure_added(self, turns: List[Dict[str, Any]], vectors: np.ndarray, match_id: str):
        """
        Incrementally adds new, unique turns and their vectors to the persistent index.
        """
        if vectors.shape[0] == 0:
            return

        new_vectors_to_add = []
        new_turns_to_map = []

        for i, turn in enumerate(turns):
            turn_key = self._generate_turn_key(turn, match_id)
            if turn_key not in self.added_keys:
                new_turns_to_map.append({'turn': turn, 'key': turn_key})
                new_vectors_to_add.append(vectors[i])

        if not new_vectors_to_add:
            return

        new_vectors_np = np.array(new_vectors_to_add).astype('float32')
        self.index.add(new_vectors_np)

        for turn_info in new_turns_to_map:
            self.id_to_turn_map[self.next_id] = turn_info['turn']
            self.added_keys.add(turn_info['key'])
            self.next_id += 1

        self._save()

    def search(self, query_vector: np.ndarray, k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        """Performs a similarity search on the index."""
        if self.index.ntotal == 0:
            return []

        k = min(k, self.index.ntotal)
        distances, indices = self.index.search(query_vector.astype('float32'), k)

        results = []
        for i in range(k):
            idx = indices[0][i]
            if idx != -1 and idx in self.id_to_turn_map:
                results.append((self.id_to_turn_map[idx], distances[0][i]))
        return results

# Global service instance that gets initialized on application startup.
faiss_index_service = FaissIndex()

def get_faiss_index():
    return faiss_index_service
