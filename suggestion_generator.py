# In suggestion_generator.py
import numpy as np
from typing import List, Dict
from sklearn.metrics.pairwise import cosine_similarity

from embedder import embedder_service
# Important: We need to import the keyphrase extractor and the nlp model from the other module
try:
    from topic_discoverer import _extract_keyphrases, nlp
except ImportError:
    # Handle case where nlp might not be initialized in time or other issues.
    nlp = None
    _extract_keyphrases = None


def generate_topic_suggestions(
    discovered_topics: Dict[str, List[str]],
    existing_keyphrases: List[str],
    my_profile: str,
    their_profile: str,
    similarity_threshold=0.7,
    top_n=5
) -> List[str]:
    """
    Generates future conversation topic suggestions based on existing topics
    and profile information.
    """
    if nlp is None or _extract_keyphrases is None:
        return ["Error: NLP model not loaded."]

    # 1. Create a dynamic suggestion vocabulary from profiles
    profile_text = f"{my_profile}\n{their_profile}"
    profile_doc = nlp(profile_text)
    suggestion_vocabulary = _extract_keyphrases(profile_doc)

    if not suggestion_vocabulary:
        return ["Could not find any topics in the profiles to suggest."]

    # 2. If no topics were discovered in convo, suggest top topics from profiles
    # that haven't been discussed.
    if not discovered_topics:
        final_suggestions = []
        for suggestion in suggestion_vocabulary:
            if suggestion not in existing_keyphrases:
                final_suggestions.append(suggestion)
            if len(final_suggestions) >= top_n:
                break
        return final_suggestions if final_suggestions else ["No new topics to suggest from profiles."]

    # 3. Find suggestions from profile vocab that are related to conversation topics
    seed_topics = list(discovered_topics.keys())

    seed_embeddings = embedder_service.encode_cached(seed_topics)
    vocab_embeddings = embedder_service.encode_cached(suggestion_vocabulary)

    similarity_matrix = cosine_similarity(seed_embeddings, vocab_embeddings)

    potential_suggestions = {}
    for i in range(similarity_matrix.shape[0]): # For each seed topic
        for j in range(similarity_matrix.shape[1]): # For each profile phrase
            score = similarity_matrix[i][j]
            suggestion = suggestion_vocabulary[j]
            if suggestion not in potential_suggestions or score > potential_suggestions[suggestion]:
                potential_suggestions[suggestion] = score

    sorted_suggestions = sorted(potential_suggestions.items(), key=lambda item: item[1], reverse=True)

    # 4. Filter out suggestions that have already been discussed
    final_suggestions = []
    for suggestion, score in sorted_suggestions:
        if len(final_suggestions) >= top_n:
            break
        # A simple direct check is better now that we use the same keyphrase extractor
        if suggestion not in existing_keyphrases:
            final_suggestions.append(suggestion)

    if not final_suggestions:
        return ["You've already discussed the main topics from the profiles! Try asking an open-ended question."]

    return final_suggestions
