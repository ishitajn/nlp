"""
This module is responsible for identifying, categorizing, and scoring topics
from the conversation history.
"""
import re
import numpy as np
from typing import List, Dict, Any, Tuple
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import fuzz

from preprocessor import extract_canonical_phrases

# --- Constants ---
CATEGORY_PRIORITY = {"sensitive": 5, "fetish": 4, "sexual": 4, "romantic": 3, "focus": 2, "avoid": 1, "neutral": 0}
ROMANTIC_INDICATORS = {'chemistry', 'date', 'connection', 'cuddle', 'kiss', 'heart', 'stargazing', 'cozy'}
LOGISTICS_INDICATORS = {'weekend', 'number', 'schedule', 'time', 'day'}
TOPIC_SIMILARITY_THRESHOLDS = {
    "AVOID_TOPICS": 0.60, "SENSITIVE_TOPICS": 0.60, "FETISH_TOPICS": 0.65,
    "SEXUAL_ADVANCE": 0.65, "ROMANTIC": 0.62, "FLIRTATION": 0.62,
}
KEYWORD_CATEGORIES = {
    "avoid": [re.compile(p, re.IGNORECASE) for p in [r'\b(politics|religion|government|election|vote|biden|trump|conservative|liberal|democrat|republican|church|god|bible)\b']],
    "sensitive": [re.compile(p, re.IGNORECASE) for p in [r'\b(autism|adhd|ocd|bpd|trauma|disability|mental health|therapy|depression|anxiety|grief|loss|death|divorce|illness|disorder|neurodivergent)\b']],
    "fetish": [re.compile(p, re.IGNORECASE) for p in [r'\b(kink|fetish|bdsm|dom|sub|foot|feet|choke|spank|daddy|kitten|leash|collar|submission)\b']],
}

# --- Helper Functions ---
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

def _consolidate_topic_groups_semantic(topics: List[str], services, threshold=0.85) -> List[List[str]]:
    if not topics: return []
    embeddings = services.embedder.encode_cached(topics)
    if embeddings.size == 0: return []
    similarity_matrix = cosine_similarity(embeddings)
    groups, processed_indices = [], set()
    for i in range(len(topics)):
        if i in processed_indices: continue
        similar_indices = np.where(similarity_matrix[i] > threshold)[0]
        new_group = [topics[j] for j in similar_indices if j not in processed_indices]
        if new_group:
            groups.append(new_group)
            processed_indices.update(similar_indices)
    return groups

# --- Main Topic Engine Functions ---
def identify_and_canonicalize_topics(
    services, conversation_turns: List[Dict[str, Any]], their_profile: str, use_enhanced_nlp: bool = False
) -> Tuple[Dict[str, List[Dict[str, Any]]], List[str]]:
    """Identifies and consolidates topics from conversation and profile text."""
    profile_topics = extract_canonical_phrases(
        their_profile, services.nlp, services.kw_extractor, services.slang_handler
    )
    all_candidate_phrases, phrase_to_source_turns = [], defaultdict(list)
    for turn in conversation_turns:
        content = turn.get("content", "")
        if not content: continue
        phrases = extract_canonical_phrases(
            content, services.nlp, services.kw_extractor, services.slang_handler
        )
        for phrase in phrases:
            all_candidate_phrases.append(phrase)
            phrase_to_source_turns[phrase].append(turn)

    filtered_phrases = _filter_and_correct_phrases(list(set(all_candidate_phrases + profile_topics)))

    if use_enhanced_nlp:
        topic_groups = _consolidate_topic_groups_semantic(filtered_phrases, services)
    else:
        topic_groups = _consolidate_topic_groups(filtered_phrases)

    final_topic_map = defaultdict(list)
    for group in topic_groups:
        if not group: continue
        canonical = min(group, key=len)
        for phrase in group:
            if phrase in phrase_to_source_turns:
                final_topic_map[canonical].extend(phrase_to_source_turns[phrase])

    for topic in final_topic_map:
        unique_turns = list({frozenset(item.items()): item for item in final_topic_map[topic]}.values())
        final_topic_map[topic] = unique_turns

    return final_topic_map, profile_topics

def _categorize_topic_enhanced(topic: str, source_turns: List[Dict[str, Any]], services) -> str:
    """Categorizes a single topic using enhanced semantic analysis."""
    source_contents = [turn['content'] for turn in source_turns]
    if not source_contents: return 'neutral'

    contextual_embedding = np.mean(services.embedder.encode_cached(source_contents), axis=0).reshape(1, -1)

    def get_similarity(concept_name):
        concept_emb = services.concept_embeddings.get(concept_name)
        if concept_emb is None: return 0.0
        return cosine_similarity(contextual_embedding, concept_emb.reshape(1, -1))[0][0]

    if get_similarity("AVOID_TOPICS") > TOPIC_SIMILARITY_THRESHOLDS["AVOID_TOPICS"]: return 'avoid'
    if get_similarity("SENSITIVE_TOPICS") > TOPIC_SIMILARITY_THRESHOLDS["SENSITIVE_TOPICS"]: return 'sensitive'
    if get_similarity("FETISH_TOPICS") > TOPIC_SIMILARITY_THRESHOLDS["FETISH_TOPICS"]: return 'fetish'
    if not any(ind in topic for ind in LOGISTICS_INDICATORS) and get_similarity("SEXUAL_ADVANCE") > TOPIC_SIMILARITY_THRESHOLDS["SEXUAL_ADVANCE"]: return 'sexual'
    if any(ind in topic for ind in ROMANTIC_INDICATORS) or max(get_similarity("ROMANTIC"), get_similarity("FLIRTATION")) > TOPIC_SIMILARITY_THRESHOLDS["ROMANTIC"]: return 'romantic'

    return 'neutral'

def _categorize_topic_standard(topic: str, source_turns: List[Dict[str, Any]], services) -> str:
    """Categorizes a single topic using standard keyword-based checks."""
    for category, patterns in KEYWORD_CATEGORIES.items():
        if any(pattern.search(topic) for pattern in patterns):
            return category
    if any(ind in topic for ind in ROMANTIC_INDICATORS): return 'romantic'

    # Fallback semantic check for sexual/romantic for standard mode
    source_contents = [turn['content'] for turn in source_turns]
    if not source_contents: return 'neutral'
    contextual_embedding = np.mean(services.embedder.encode_cached(source_contents), axis=0).reshape(1, -1)

    sexual_emb = services.concept_embeddings.get("SEXUAL_ADVANCE")
    if not any(ind in topic for ind in LOGISTICS_INDICATORS) and sexual_emb is not None and \
       cosine_similarity(contextual_embedding, sexual_emb.reshape(1, -1))[0][0] > TOPIC_SIMILARITY_THRESHOLDS["SEXUAL_ADVANCE"]:
        return 'sexual'

    return 'neutral'

def score_and_categorize_topics(
    services, topic_map: Dict[str, List[Dict[str, Any]]], profile_topics: List[str], focus_topic: str, topic_salience: Dict[str, float], use_enhanced_nlp: bool = False
) -> Dict[str, List[str]]:
    """Categorizes all identified topics and ranks them."""
    if not topic_map: return {cat: [] for cat in CATEGORY_PRIORITY}
    
    topic_to_category = {}
    for topic, source_turns in topic_map.items():
        if topic == focus_topic:
            topic_to_category[topic] = 'focus'
            continue
        if use_enhanced_nlp:
            topic_to_category[topic] = _categorize_topic_enhanced(topic, source_turns, services)
        else:
            topic_to_category[topic] = _categorize_topic_standard(topic, source_turns, services)

    final_output = defaultdict(list)
    for topic, category in topic_to_category.items():
        final_output[category].append(topic)
    
    ranked_and_limited_output = {cat: [] for cat in CATEGORY_PRIORITY}
    for category, topics in final_output.items():
        sorted_topics = sorted(topics, key=lambda t: topic_salience.get(t, 0), reverse=True)
        ranked_and_limited_output[category] = [t.title() for t in sorted_topics[:10]]
        
    return ranked_and_limited_output