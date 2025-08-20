import logging
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from dating_nlp_bot.models.embeddings import EmbeddingModel

# Try to import libraries, but don't fail if they're not installed
try:
    from ctransformers import AutoModelForCausalLM
    from transformers import pipeline
except ImportError:
    pipeline = None
    AutoModelForCausalLM = None

# Configure logging
logger = logging.getLogger(__name__)

class ModelLoader:
    def __init__(self):
        self.sentiment_analyzer_fast = None
        self.embedding_model = None
        self.text_generator_enhanced = None

    def load_models(self):
        logger.info("Loading models...")
        self.sentiment_analyzer_fast = SentimentIntensityAnalyzer()
        self.embedding_model = EmbeddingModel()

        # Load enhanced models only if libraries are available
        if AutoModelForCausalLM:
            try:
                logger.info("Loading TinyLlama model for brain analysis...")
                # Using ctransformers for GGUF model
                self.text_generator_enhanced = AutoModelForCausalLM.from_pretrained(
                    "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
                    model_file="tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
                    model_type="llama"
                )
                logger.info("TinyLlama model loaded.")
            except Exception as e:
                logger.error(f"Failed to load TinyLlama model: {e}")
                self.text_generator_enhanced = None
        else:
            logger.warning("ctransformers library not found. Enhanced brain analysis will be disabled.")

        logger.info("Models loaded successfully.")

models = ModelLoader()

def get_models():
    return models
