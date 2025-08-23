# In suggestion_generator.py
import numpy as np
from typing import List, Dict
from sklearn.metrics.pairwise import cosine_similarity

from embedder import embedder_service

# A predefined vocabulary of potential conversation topics.
# This list can be expanded or refined.
SUGGESTION_VOCABULARY = [
    # Hobbies & Interests
    "playing any sports", "favorite music genres", "a book you've read recently",
    "favorite movies or TV shows", "art or museums", "cooking or baking",
    "playing video games", "gardening", "photography", "learning a new skill",
    "playing a musical instrument",

    # Travel
    "dream travel destinations", "most memorable trip", "weekend getaways",
    "beaches or mountains", "cultural experiences while traveling",

    # Career & Ambition
    "your career goals", "what you love about your job", "work-life balance",
    "a project you're proud of",

    # Personal Growth & Values
    "what you're passionate about", "personal values", "learning from challenges",
    "what makes you happy",

    # Fun & Lighthearted
    "your favorite food", "pets or animals", "your go-to karaoke song",
    "hidden talents", "your favorite season and why", "coffee or tea",
    "a funny personal story",

    # Deeper Connection
    "your love language", "what you look for in a partner", "family and friends",
    "your definition of a perfect day",
]

def generate_topic_suggestions(
    discovered_topics: Dict[str, List[str]],
    existing_keyphrases: List[str],
    similarity_threshold=0.7,
    top_n=5
) -> List[str]:
    """
    Generates future conversation topic suggestions based on existing topics.
    """
    if not discovered_topics:
        # Return generic suggestions if no topics were discovered
        return [
            "Ask about something they are passionate about.",
            "Bring up a recent experience you enjoyed.",
            "Talk about favorite weekend activities."
        ][:top_n]

    seed_topics = list(discovered_topics.keys())

    # Embed the seed topics, the suggestion vocabulary, and existing phrases
    seed_embeddings = embedder_service.encode_cached(seed_topics)
    vocab_embeddings = embedder_service.encode_cached(SUGGESTION_VOCABULARY)

    # Calculate similarity between seed topics and the suggestion vocabulary
    similarity_matrix = cosine_similarity(seed_embeddings, vocab_embeddings)

    # Find the best suggestion for each seed topic, store with its similarity score
    potential_suggestions = {}
    for i in range(similarity_matrix.shape[0]):
        for j in range(similarity_matrix.shape[1]):
            score = similarity_matrix[i][j]
            suggestion = SUGGESTION_VOCABULARY[j]
            # If suggestion is already there, update if the new score is higher
            if suggestion not in potential_suggestions or score > potential_suggestions[suggestion]:
                potential_suggestions[suggestion] = score

    # Sort suggestions by similarity score in descending order
    sorted_suggestions = sorted(potential_suggestions.items(), key=lambda item: item[1], reverse=True)

    # Filter out suggestions that are too similar to what has already been said
    final_suggestions = []
    if existing_keyphrases:
        existing_phrase_embeddings = embedder_service.encode_cached(existing_keyphrases)
        for suggestion, score in sorted_suggestions:
            if len(final_suggestions) >= top_n:
                break

            suggestion_embedding = embedder_service.encode_cached([suggestion])
            similarity_to_existing = cosine_similarity(suggestion_embedding, existing_phrase_embeddings)

            if np.max(similarity_to_existing) < similarity_threshold:
                final_suggestions.append(suggestion)
    else: # If no existing keyphrases, just take the top N
        final_suggestions = [s for s, score in sorted_suggestions[:top_n]]


    if not final_suggestions:
        return [
            "Ask about something they are passionate about.",
            "Bring up a recent experience you enjoyed."
        ]

    return final_suggestions
