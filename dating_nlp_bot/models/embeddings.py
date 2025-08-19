import torch
from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer
import numpy as np
from dating_nlp_bot import config

def mean_pooling(model_output, attention_mask):
    """
    Mean pooling to get sentence embeddings.
    From: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
    """
    token_embeddings = model_output[0] # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

class EmbeddingModel:
    def __init__(self, model_name=config.EMBEDDING_MODEL):
        # Using a pre-converted ONNX model for faster inference
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = ORTModelForFeatureExtraction.from_pretrained(
            model_name,
            file_name="onnx/model_quantized.onnx",
            provider="CPUExecutionProvider"
        )


    def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        """
        Generates embeddings for a list of texts using ONNX Runtime.
        """
        # The model from Xenova is already quantized and optimized for CPU
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        outputs = self.model(**inputs)

        # Perform pooling
        sentence_embeddings = mean_pooling(outputs, inputs['attention_mask'])

        # Normalize embeddings
        normalized_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)

        return normalized_embeddings.tolist()
