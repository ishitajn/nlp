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

def identify_and_canonicalize_topics(
    conversation_turns: List[Dict[str, Any]],
    their_profile: str
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


def score_and_categorize_topics(
    topic_map: Dict[str, List[Dict[str, Any]]],
    profile_topics: List[str],
    focus_topic: str, # The single, most recent topic
    topic_salience: Dict[str, float] # Pre-calculated scores for ranking
) -> Dict[str, List[str]]:
    if not topic_map: return {cat: [] for cat in CATEGORY_PRIORITY}
    
    topic_to_category = {}

    for topic, source_turns in topic_map.items():
        if topic == focus_topic:
            topic_to_category[topic] = 'focus'
            continue

        # Keyword-based categorization first
        categorized = False
        for category, patterns in KEYWORD_CATEGORIES.items():
            if any(pattern.search(topic) for pattern in patterns):
                topic_to_category[topic] = category
                categorized = True
                break
        if categorized:
            continue

        # Semantic categorization
        source_contents = [turn['content'] for turn in source_turns]
        contextual_embedding = np.mean(embedder_service.encode_cached(source_contents), axis=0)
        is_romantic_indicator = any(ind in topic for ind in ROMANTIC_INDICATORS)
        is_logistics_indicator = any(ind in topic for ind in LOGISTICS_INDICATORS)

        if not is_logistics_indicator:
            sexual_emb = CONCEPT_EMBEDDINGS.get("SEXUAL_ADVANCE")
            if sexual_emb is not None and cosine_similarity(contextual_embedding.reshape(1, -1), sexual_emb.reshape(1, -1))[0][0] > 0.65:
                topic_to_category[topic] = 'sexual'
        
        if topic not in topic_to_category:
            romantic_embs = [CONCEPT_EMBEDDINGS.get("ROMANTIC"), CONCEPT_EMBEDDINGS.get("FLIRTATION")]
            romantic_score = max((cosine_similarity(contextual_embedding.reshape(1, -1), emb.reshape(1, -1))[0][0] if emb is not None else 0.0) for emb in romantic_embs)
            if romantic_score > 0.62 or is_romantic_indicator:
                topic_to_category[topic] = 'romantic'
        
        if topic not in topic_to_category:
            topic_to_category[topic] = 'neutral'

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