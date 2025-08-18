from sentence_transformers import SentenceTransformer

class EmbeddingModel:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        """
        Generates embeddings for a list of texts.
        """
        return self.model.encode(texts, convert_to_tensor=False).tolist()
