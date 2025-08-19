import logging
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from dating_nlp_bot.models.sentiment_model import EnhancedSentimentModel
from dating_nlp_bot.models.topic_model import EnhancedTopicModel
from dating_nlp_bot.models.embeddings import EmbeddingModel
from dating_nlp_bot.models.llm_generator import LLMGenerator

# Configure logging
logger = logging.getLogger(__name__)

class ModelLoader:
    def __init__(self):
        self.sentiment_analyzer_fast = None
        self.sentiment_model_enhanced = None
        self.topic_model_enhanced = None
        self.embedding_model = None
        self.llm_generator = None

    def load_models(self):
        logger.info("Loading models...")
        self.sentiment_analyzer_fast = SentimentIntensityAnalyzer()
        self.sentiment_model_enhanced = EnhancedSentimentModel()
        self.topic_model_enhanced = EnhancedTopicModel()
        self.embedding_model = EmbeddingModel()
        try:
            self.llm_generator = LLMGenerator()
        except Exception as e:
            logger.error(f"Failed to load LLM Generator: {e}", exc_info=True)
            # Depending on requirements, you might want to handle this more gracefully
            # For now, we'll proceed without it if it fails
            self.llm_generator = None
        logger.info("Models loaded successfully.")

models = ModelLoader()

def get_models():
    return models
