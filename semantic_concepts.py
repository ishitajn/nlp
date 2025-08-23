# In semantic_concepts.py
"""
A centralized library for defining and pre-computing embeddings for core conversational concepts.
This allows for robust, semantic-based checks instead of relying on brittle keywords or regex.
"""
from embedder import embedder_service

# Define the concepts with descriptive strings that capture their semantic meaning.
CONCEPT_DEFINITIONS = {
    "GREETING": "saying hi, hello, how are you, hey there, good morning, good afternoon, good evening, what's up",
    "ASKING_A_QUESTION": "asking a question, being curious, inquisitive, what, how, when, where, why, do you, are you, can you, will you",
    "FLIRTATION": "flirting, being romantic, expressing sexual desire, passion, attraction, chemistry, saying someone is cute, hot, or beautiful",
    "TIME_REFERENCE": "talking about time, dates, scheduling, today, tomorrow, this weekend, next week, morning, evening, hours, minutes",
    "LOCATION_REFERENCE": "talking about places, locations, meeting up, distance, city, country, here, there, neighborhood, area",
    "DISENGAGEMENT": "being unsure, ambiguous, non-committal, I don't know, maybe, we'll see, lol, haha, okay",
    "PLANNING_LOGISTICS": "making plans, scheduling, organizing a meetup, suggesting a date, asking about availability"
}

# Pre-compute the embeddings for each concept for efficient reuse.
# The result is a dictionary mapping the concept name to its numpy array embedding.
CONCEPT_EMBEDDINGS = {
    name: embedder_service.encode_cached([description])[0]
    for name, description in CONCEPT_DEFINITIONS.items()
}
