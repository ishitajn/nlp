# In topic_engine.py
import re
import numpy as np
from typing import List, Dict, Any, Set, Tuple
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import fuzz

from embedder import embedder_service
from preprocessor import extract_canonical_phrases
from semantic_concepts import CONCEPT_EMBEDDINGS

# --- Constants ---
CATEGORY_PRIORITY = {"sensitive": 5, "fetish": 4, "sexual": 4, "romantic": 3, "focus": 2, "avoid": 1, "neutral": 0}
ROMANTIC_INDICATORS = {'chemistry', 'date', 'connection', 'cuddle', 'kiss', 'heart', 'stargazing', 'cozy'}
LOGISTICS_INDICATORS = {'weekend', 'number', 'schedule', 'time', 'day'}

# Pre-compiled regex for keyword-based categorization
KEYWORD_CATEGORIES = {
    "avoid": [re.compile(p, re.IGNORECASE) for p in [r'\b(politics|religion|government|election|vote|biden|trump|conservative|liberal|democrat|republican|church|god|bible)\b']],
    "sensitive": [re.compile(p, re.IGNORECASE) for p in [r'\b(autism|adhd|ocd|bpd|trauma|disability|mental health|therapy|depression|anxiety|grief|loss|death|divorce|illness|disorder|neurodivergent)\b']],
    "fetish": [re.compile(p, re.IGNORECASE) for p in [r'\b(kink|fetish|bdsm|dom|sub|foot|feet|choke|spank|daddy|kitten|leash|collar|submission)\b']],
}

def _filter_and_correct_phrases(phrases: List[str]) -> List[str]:
    noise_blocklist = {'a big fan', 'my life', 'weeks', 'my bestie', 'a side', 'the first thing', 'a good food'}
    typo_map = {"favorite foodv": "favorite food"}
    corrected_phrases = [typo_map.get(p, p) for p in phrases]
    return [p for p in corrected_phrases if p not in noise_blocklist and (len(p.split()) > 1 or len(p) > 5)]

def _consolidate_topic_groups(topics: List[str], threshold=80) -> List[List[str]]:
    if not topics: return []
    topics.sort(key=len, reverse=True)
    groups, processed = [], set()
    for topic in topics:
        if topic in processed: continue
        similar_group = {t for t in topics if fuzz.token_set_ratio(topic, t) > threshold}
        groups.append(list(similar_group))
        processed.update(similar_group)
    return groups

def _consolidate_topic_groups_semantic(topics: List[str], threshold=0.85) -> List[List[str]]:
    if not topics:
        return []

    embeddings = embedder_service.encode_cached(topics)
    if embeddings.size == 0:
        return []

    similarity_matrix = cosine_similarity(embeddings)

    groups = []
    processed_indices = set()

    for i in range(len(topics)):
        if i in processed_indices:
            continue

        # Find indices of similar topics (where similarity is above the threshold)
        similar_indices = np.where(similarity_matrix[i] > threshold)[0]

        # Create a group of topic strings from the indices
        new_group = [topics[j] for j in similar_indices if j not in processed_indices]

        if new_group:
            groups.append(new_group)
            # Add all indices from this new group to the processed set
            for j in similar_indices:
                processed_indices.add(j)

    return groups

def identify_and_canonicalize_topics(
    conversation_turns: List[Dict[str, Any]],
    their_profile: str,
    use_enhanced_nlp: bool = False
) -> Tuple[Dict[str, List[Dict[str, Any]]], List[str]]:
    profile_topics = extract_canonical_phrases(their_profile)
    all_candidate_phrases, phrase_to_source_turns = [], defaultdict(list)
    for turn in conversation_turns:
        content = turn.get("content", "")
        if not content: continue
        phrases = extract_canonical_phrases(content)
        for phrase in phrases:
            all_candidate_phrases.append(phrase)
            phrase_to_source_turns[phrase].append(turn)

    filtered_phrases = _filter_and_correct_phrases(list(set(all_candidate_phrases + profile_topics)))

    if use_enhanced_nlp:
        topic_groups = _consolidate_topic_groups_semantic(filtered_phrases)
    else:
        topic_groups = _consolidate_topic_groups(filtered_phrases)

    final_topic_map = defaultdict(list)
    for group in topic_groups:
        if not group: continue
        canonical = min(group, key=len)
        for phrase in group:
            if phrase in phrase_to_source_turns:
                # Use extend to add all turns from the source phrase
                final_topic_map[canonical].extend(phrase_to_source_turns[phrase])

    # Ensure final map has unique turns per topic
    for topic in final_topic_map:
        # Create a set of tuples to find unique dicts, then convert back to list
        unique_turns = list({frozenset(item.items()): item for item in final_topic_map[topic]}.values())
        final_topic_map[topic] = unique_turns

    return final_topic_map, profile_topics


def _categorize_topic_enhanced(topic: str, source_turns: List[Dict[str, Any]]) -> str:
    """Categorizes a single topic using enhanced semantic analysis."""
    source_contents = [turn['content'] for turn in source_turns]
    if not source_contents:
        return 'neutral'

    contextual_embedding = np.mean(embedder_service.encode_cached(source_contents), axis=0).reshape(1, -1)

    # Define a helper for similarity checks
    def get_similarity(concept_name):
        concept_emb = CONCEPT_EMBEDDINGS.get(concept_name)
        if concept_emb is None: return 0.0
        return cosine_similarity(contextual_embedding, concept_emb.reshape(1, -1))[0][0]

    # Check against high-priority semantic concepts
    if get_similarity("AVOID_TOPICS") > 0.6: return 'avoid'
    if get_similarity("SENSITIVE_TOPICS") > 0.6: return 'sensitive'
    if get_similarity("FETISH_TOPICS") > 0.65: return 'fetish'

    is_logistics_indicator = any(ind in topic for ind in LOGISTICS_INDICATORS)
    if not is_logistics_indicator and get_similarity("SEXUAL_ADVANCE") > 0.65:
        return 'sexual'

    is_romantic_indicator = any(ind in topic for ind in ROMANTIC_INDICATORS)
    romantic_score = max(get_similarity("ROMANTIC"), get_similarity("FLIRTATION"))
    if romantic_score > 0.62 or is_romantic_indicator:
        return 'romantic'

    return 'neutral'

def _categorize_topic_standard(topic: str) -> str:
    """Categorizes a single topic using standard keyword-based checks."""
    for category, patterns in KEYWORD_CATEGORIES.items():
        if any(pattern.search(topic) for pattern in patterns):
            return category

    is_romantic_indicator = any(ind in topic for ind in ROMANTIC_INDICATORS)
    if is_romantic_indicator:
        return 'romantic'

    return 'neutral'


def score_and_categorize_topics(
    topic_map: Dict[str, List[Dict[str, Any]]],
    profile_topics: List[str],
    focus_topic: str, # The single, most recent topic
    topic_salience: Dict[str, float], # Pre-calculated scores for ranking
    use_enhanced_nlp: bool = False
) -> Dict[str, List[str]]:
    if not topic_map: return {cat: [] for cat in CATEGORY_PRIORITY}
    
    topic_to_category = {}

    for topic, source_turns in topic_map.items():
        if topic == focus_topic:
            topic_to_category[topic] = 'focus'
            continue

        if use_enhanced_nlp:
            # The enhanced path uses full semantic analysis for categorization
            topic_to_category[topic] = _categorize_topic_enhanced(topic, source_turns)
        else:
            # The standard path uses keywords first, then limited semantics
            category = _categorize_topic_standard(topic)
            if category == 'neutral': # Only run semantics if keywords don't match
                source_contents = [turn['content'] for turn in source_turns]
                contextual_embedding = np.mean(embedder_service.encode_cached(source_contents), axis=0)
                is_logistics_indicator = any(ind in topic for ind in LOGISTICS_INDICATORS)

                if not is_logistics_indicator:
                    sexual_emb = CONCEPT_EMBEDDINGS.get("SEXUAL_ADVANCE")
                    if sexual_emb is not None and cosine_similarity(contextual_embedding.reshape(1, -1), sexual_emb.reshape(1, -1))[0][0] > 0.65:
                        category = 'sexual'

                if category == 'neutral':
                     romantic_embs = [CONCEPT_EMBEDDINGS.get("ROMANTIC"), CONCEPT_EMBEDDINGS.get("FLIRTATION")]
                     romantic_score = max((cosine_similarity(contextual_embedding.reshape(1, -1), emb.reshape(1, -1))[0][0] if emb is not None else 0.0) for emb in romantic_embs)
                     if romantic_score > 0.62:
                         category = 'romantic'

            topic_to_category[topic] = category

    # Assemble, Rank, and Limit
    final_output = defaultdict(list)
    for topic, category in topic_to_category.items():
        final_output[category].append(topic)
    
    # Rank topics within each category by their salience score and limit to 10
    ranked_and_limited_output = {cat: [] for cat in CATEGORY_PRIORITY}
    for category, topics in final_output.items():
        # Sort topics based on the pre-calculated salience score
        sorted_topics = sorted(topics, key=lambda t: topic_salience.get(t, 0), reverse=True)
        ranked_and_limited_output[category] = [t.title() for t in sorted_topics[:10]]
        
    return ranked_and_limited_output