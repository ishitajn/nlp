from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from .. import config

class EnhancedSentimentModel:
    def __init__(self, model_name=config.ENHANCED_SENTIMENT_MODEL):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.labels = ["negative", "neutral", "positive"]

    def predict(self, text: str) -> tuple[str, float]:
        """
        Predicts the sentiment of a given text.
        Returns a tuple of (sentiment_label, confidence_score).
        """
        if not isinstance(text, str) or not text.strip():
            return "neutral", 1.0

        # Truncate text to avoid errors with long inputs
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)

        with torch.no_grad():
            outputs = self.model(**inputs)

        scores = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
        sentiment_idx = torch.argmax(scores).item()

        label = self.labels[sentiment_idx]
        confidence = scores[sentiment_idx].item()

        return label, confidence
