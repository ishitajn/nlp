import logging
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from dating_nlp_bot.models.embeddings import EmbeddingModel

# Try to import transformers, but don't fail if it's not installed
try:
    from transformers import pipeline
except ImportError:
    pipeline = None

# Configure logging
logger = logging.getLogger(__name__)

class ModelLoader:
    def __init__(self):
        self.sentiment_analyzer_fast = None
        self.embedding_model = None
        self.topic_classifier_enhanced = None
        self.text_generator_enhanced = None

    def load_models(self):
        logger.info("Loading models...")
        self.sentiment_analyzer_fast = SentimentIntensityAnalyzer()
        self.embedding_model = EmbeddingModel()

        # Load enhanced models only if transformers is available
        if pipeline:
            try:
                logger.info("Loading zero-shot classification model for topic analysis...")
                self.topic_classifier_enhanced = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
                logger.info("Zero-shot classification model loaded.")
            except Exception as e:
                logger.error(f"Failed to load zero-shot classification model: {e}")
                self.topic_classifier_enhanced = None

            try:
                logger.info("Loading text generation model for brain analysis...")
                self.text_generator_enhanced = pipeline("text-generation", model="distilgpt2")
                logger.info("Text generation model loaded.")
            except Exception as e:
                logger.error(f"Failed to load text generation model: {e}")
                self.text_generator_enhanced = None
        else:
            logger.warning("Transformers library not found. Enhanced NLP features will be disabled.")

        logger.info("Models loaded successfully.")

models = ModelLoader()

def get_models():
    return models
