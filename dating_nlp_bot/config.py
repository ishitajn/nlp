# Centralized configuration for model names and other constants.

# --- Model Names ---
ENHANCED_ADULT_MODEL = "unitary/toxic-bert"
EMBEDDING_MODEL = "Xenova/all-MiniLM-L6-v2"

# --- Sentiment Analysis ---
SENTIMENT_THRESHOLD_FAST_POSITIVE = 0.05
SENTIMENT_THRESHOLD_FAST_NEGATIVE = -0.05
SENTIMENT_THRESHOLD_ENHANCED_POSITIVE = 0.1
SENTIMENT_THRESHOLD_ENHANCED_NEGATIVE = -0.1

# --- Topic Analysis ---
TOPIC_SENTIMENT_THRESHOLD_POSITIVE = 0.2
TOPIC_SENTIMENT_THRESHOLD_NEGATIVE = -0.2
GENERAL_TOPICS = ["travel", "food", "sports", "career", "flirt", "sexual", "emotions"]
FEMALE_CENTRIC_TOPICS = ["fashion", "wellness", "hobbies", "social", "relationships"]
