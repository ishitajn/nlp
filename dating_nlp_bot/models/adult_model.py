from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from .. import config

class EnhancedAdultModel:
    def __init__(self, model_name=config.ENHANCED_ADULT_MODEL):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

    def predict(self, text: str) -> dict[str, float]:
        """
        Predicts the toxicity scores for a given text.
        Returns a dictionary of labels to confidence scores.
        """
        if not isinstance(text, str) or not text.strip():
            return {label: 0.0 for label in self.labels}

        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)

        with torch.no_grad():
            outputs = self.model(**inputs)

        scores = torch.sigmoid(outputs.logits)[0]

        return {self.labels[i]: scores[i].item() for i in range(len(self.labels))}
