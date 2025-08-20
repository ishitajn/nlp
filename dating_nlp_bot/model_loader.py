import logging
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from dating_nlp_bot.models.embeddings import EmbeddingModel

# Configure logging
logger = logging.getLogger(__name__)

class ModelLoader:
    def __init__(self):
        self.sentiment_analyzer_fast = None
        self.embedding_model = None

    def load_models(self):
        logger.info("Loading models...")
        self.sentiment_analyzer_fast = SentimentIntensityAnalyzer()
        self.embedding_model = EmbeddingModel()
        logger.info("Models loaded successfully.")

models = ModelLoader()

def get_models():
    return models
