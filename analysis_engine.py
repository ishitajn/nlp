# In app/svc/analysis_engine.py

import re
import spacy
import numpy as np
from collections import Counter
from typing import List, Dict, Any
from sklearn.metrics.pairwise import cosine_similarity
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from embedder import embedder_service
from topic_engine_v2 import run_topic_engine

try:
    nlp = spacy.load("en_core_web_trf")
    print("Successfully loaded high-accuracy transformer model: en_core_web_trf")
except OSError:
    print("Spacy model 'en_core_web_trf' not found. Please run 'pip install spacy-transformers && python -m spacy download en_core_web_trf'")
    nlp = None

CANONICAL_TOPICS = {
    "Career & Ambition": "Talking about jobs, career paths, professional goals, work-life balance, and ambition.",
    "Family & Background": "Discussing family, parents, siblings, nieces, nephews, upbringing, and hometown.",
    "Hobbies & Passions": "Chatting about personal interests, hobbies, creative pursuits, sports, and what we do for fun.",
    "Travel & Adventure": "Sharing stories about past travels, dream destinations, and future adventure plans.",
    "Fitness & Health": "Discussing workouts, gym, jogging, physical activities, and general wellness.",
    "Food & Drink": "Talking about favorite foods, cooking, restaurants, coffee, or cocktails.",
    "Flirting & Compliments": "Exchanging compliments, playful banter, and expressing romantic or physical attraction.",
    "Deeper Connection": "Sharing vulnerabilities, personal feelings, and building a deeper emotional connection.",
    "Making Plans & Logistics": "Suggesting meeting up, discussing availability, and planning the logistics for a date.",
    "Sexual Escalation & Kinks": "Discussing sexual preferences, kinks, fantasies, sexting, and other explicit topics.",
    "Pop Culture & Media": "Talking about movies, music, TV shows, books, and other pop culture.",
    "Inside Jokes & Nicknames": "Using repeated, unique phrases or nicknames specific to the conversation."
}

TOPIC_TO_CATEGORY_MAP = {
    "Career & Ambition": "neutral", "Family & Background": "neutral", "Hobbies & Passions": "neutral",
    "Travel & Adventure": "neutral", "Fitness & Health": "neutral", "Food & Drink": "neutral",
    "Pop Culture & Media": "neutral", "Making Plans & Logistics": "neutral",
    "Flirting & Compliments": "sexual", "Deeper Connection": "intimacy",
    "Sexual Escalation & Kinks": "sexual", "Inside Jokes & Nicknames": "neutral"
}

ANALYSIS_SCHEMA = {
    "Phases": { "Icebreaker": [r'\b(h(i|ey|ello)|how are you|your profile|we matched)\b'], "Rapport Building": [r'\b(tell me more|what about you|hobbies|passions|family|career)\b'], "Escalation": [r'\b(tension|desire|imagining|in person|what if|chemistry)\b'], "Explicit Banter": [r'\b(fuck|sex|nude|kink|sexting|horny|aroused)\b'], "Logistics": [r'\b(when are you free|let\'s meet|what\'s your number|schedule|date)\b'], },
    "Tones": { "Playful": [r'\b(haha|lol|lmao|kidding|teasing|banter|playful|cheeky)\b', r'[😉😜😏]'], "Serious": [r'\b(to be honest|actually|my values|looking for|seriously)\b'], "Romantic": [r'\b(connection|special|beautiful|chemistry|heart|adore|lovely)\b'], "Complimentary": [r'\b(great|amazing|impressive|gorgeous|handsome|hot|sexy|cute)\b'], "Vulnerable": [r'\b(my feelings|i feel|struggle|opening up is hard|i feel safe with you)\b'], },
    "Intents": { "Gathering Information": [r'\?'], "Building Comfort": [r'\b(that makes sense|i understand|thank you for sharing)\b'], "Testing Boundaries": [r'\b(what are you into|how adventurous|are you open to)\b'], "Making Plans": [r'\b(we should|let\'s|are you free|wanna grab)\b'], "Expressing Desire": [r'\b(i want you|i need you|can\'t stop thinking about you|i desire you)\b'], }
}

sentiment_analyzer = SentimentIntensityAnalyzer()

if nlp:
    canonical_topic_names = list(CANONICAL_TOPICS.keys())
    canonical_topic_vectors = embedder_service.encode_cached(list(CANONICAL_TOPICS.values()))

def _clean_topic_label(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"'s\b", "", text)
    if text.startswith("the ") or text.startswith("a ") or text.startswith("an "):
        text = text.split(" ", 1)[1]
    return text

def _map_conversation_to_concepts(doc: spacy.tokens.Doc, similarity_threshold=0.6) -> Dict[str, List[str]]:
    """Performs semantic search to map canonical concepts to relevant sentences."""
    sentences = [sent.text for sent in doc.sents if len(sent.text.split()) > 3]
    if not sentences: return {}

    sentence_vectors = embedder_service.encode_cached(sentences)
    similarity_matrix = cosine_similarity(sentence_vectors, canonical_topic_vectors)

    concept_to_sentences_map = {name: [] for name in canonical_topic_names}
    for i, sentence in enumerate(sentences):
        best_match_index = np.argmax(similarity_matrix[i])
        if similarity_matrix[i][best_match_index] > similarity_threshold:
            canonical_topic = canonical_topic_names[best_match_index]
            concept_to_sentences_map[canonical_topic].append(sentence)
            
    return {k: v for k, v in concept_to_sentences_map.items() if v}

def _extract_raw_topics_from_sentences(sentences: List[str]) -> List[str]:
    """Runs noun phrase extraction on a targeted list of sentences."""
    if not sentences: return []
    
    text = " ".join(sentences)
    doc = nlp(text)
    
    discovered_chunks = set()
    stopwords = {'i', 'you', 'me', 'my', 'it', 'that', 'what', 'wbu', 'hmmmm', 'lol', 'haha', 'the first thing', 'a bit', 'thing', 'hihi'}
    for chunk in doc.noun_chunks:
        clean_chunk = _clean_topic_label(chunk.text)
        if chunk.root.pos_ != 'PRON' and clean_chunk not in stopwords and len(clean_chunk) > 3:
            discovered_chunks.add(clean_chunk)
    return list(discovered_chunks)

def run_full_analysis(my_profile: str, their_profile: str, turns: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not nlp: raise RuntimeError("spaCy model is not loaded.")

    conversation_history_str = "\n".join([f"{t.get('content', '')}" for t in turns])
    full_text_for_rules = f"{my_profile} {their_profile} {conversation_history_str}"
    full_text_lower = full_text_for_rules.lower()
    doc = nlp(conversation_history_str)
    
    # --- 1. Concept-Centric Semantic Search ---
    concept_map = _map_conversation_to_concepts(doc)

    # --- 2. Build the Definitive Topic Structure ---
    topic_mapping = {}
    categorized_raw_topics = { "neutral": set(), "sensitive": set(), "fetish": set(), "sexual": set(), "intimacy": set() }
    
    for canonical_topic, sentences in concept_map.items():
        raw_topics = _extract_raw_topics_from_sentences(sentences)
        if not raw_topics: continue
        
        topic_mapping[canonical_topic] = raw_topics
        category = TOPIC_TO_CATEGORY_MAP.get(canonical_topic, "neutral")
        categorized_raw_topics[category].update(raw_topics)

    # --- 3. Heatmap Calculation on Canonical Topics ---
    occurrence_heatmap = dict(Counter(concept_map.keys()).most_common(15))
    
    last_occurrence = {}
    for i, turn in enumerate(turns):
        turn_doc = nlp(turn['content'])
        turn_sentences = [sent.text for sent in turn_doc.sents]
        if not turn_sentences: continue
        
        turn_vectors = embedder_service.encode_cached(turn_sentences)
        similarity_matrix = cosine_similarity(turn_vectors, canonical_topic_vectors)
        
        if np.max(similarity_matrix) > 0.6:
            best_match_index = np.unravel_index(np.argmax(similarity_matrix, axis=None), similarity_matrix.shape)[1]
            canonical_topic = canonical_topic_names[best_match_index]
            last_occurrence[canonical_topic] = i
            
    sorted_by_recency = sorted(last_occurrence.items(), key=lambda item: item[1], reverse=True)
    recency_heatmap = {topic: rank + 1 for rank, (topic, index) in enumerate(sorted_by_recency[:15])}

    # --- New V2 Topic Engine ---
    v2_topic_results = run_topic_engine(turns)

    # --- 4. Build the Final Conversation State ---
    focus_topics = list(categorized_raw_topics["neutral"] | categorized_raw_topics["sexual"] | categorized_raw_topics["intimacy"])
    top_20_topics = [t for t, c in Counter(focus_topics).most_common(20)]
    inside_jokes = [chunk.text for chunk in doc.noun_chunks if chunk.root.pos_ == 'PROPN' and len(chunk.text) > 2 and conversation_history_str.count(chunk.text) > 1]

    conversation_state = {
        "v2_topics": v2_topic_results,
        "topics": {
            "focus": top_20_topics, "avoid": [],
            "neutral": [t for t in categorized_raw_topics["neutral"] if t in top_20_topics],
            "sensitive": [t for t in categorized_raw_topics["sensitive"] if t in top_20_topics],
            "fetish": [t for t in categorized_raw_topics["fetish"] if t in top_20_topics],
            "sexual": [t for t in categorized_raw_topics["sexual"] if t in top_20_topics],
            "inside_jokes": list(set(inside_jokes))
        },
        "recent_topics": list(recency_heatmap.keys()),
        "topic_occurrence_heatmap": occurrence_heatmap,
        "topic_recency_heatmap": recency_heatmap,
        "topic_mapping": {k: v for k, v in topic_mapping.items() if v}
    }

    # --- 5. Schema-Based Tagging (for Structure) ---
    detected_tags = { "detected_phases": set(), "detected_tones": set(), "detected_intents": set() }
    for category, rules in ANALYSIS_SCHEMA.items():
        for tag_name, patterns in rules.items():
            if any(re.search(pattern, full_text_lower) for pattern in patterns):
                detected_tags[f"detected_{category.lower()}"].add(tag_name)

    analysis = {k: list(v) for k, v in detected_tags.items()}
    analysis["conversation_state"] = conversation_state

    # --- 6. Quantitative & Sentiment Analysis ---
    sentiment_scores = sentiment_analyzer.polarity_scores(conversation_history_str)
    compound_score = sentiment_scores['compound']
    sentiment = "neutral"
    if compound_score > 0.5: sentiment = "very positive"
    elif compound_score > 0.05: sentiment = "positive"
    elif compound_score < -0.5: sentiment = "very negative"
    elif compound_score < -0.05: sentiment = "negative"

    num_turns = len(turns)
    question_count = conversation_history_str.count('?')
    avg_msg_len = len(conversation_history_str) / num_turns if num_turns > 0 else 0
    
    engagement = "low"
    if num_turns > 4 and question_count > 2: engagement = "medium"
    if num_turns > 8 and avg_msg_len > 50 and question_count > 4: engagement = "high"
    if num_turns > 12 and avg_msg_len > 70 and question_count > 6: engagement = "very high"

    pace = "steady"
    if avg_msg_len < 40 and num_turns > 10: pace = "fast"
    if avg_msg_len > 120: pace = "slow and thoughtful"
    if "Escalation" in analysis["detected_phases"]: pace = "steady with potential for escalation"

    flirt_keywords = ['flirt', 'teasing', 'sexy', 'hot', 'desire', 'tension', 'imagining', 'irresistible', '😉', '😏', 'cuddle', 'kiss']
    flirt_score = sum(full_text_lower.count(kw) for kw in flirt_keywords)
    flirtation_level = "low"
    if flirt_score > 5 or "Explicit Banter" in analysis["detected_phases"]: flirtation_level = "very high"
    elif flirt_score > 2 or "Escalation" in analysis["detected_phases"]: flirtation_level = "high"
    elif flirt_score > 0: flirtation_level = "medium"

    analysis["sentiment_analysis"] = { "overall": sentiment, "compound_score": compound_score }
    analysis["engagement_metrics"] = { "level": engagement, "pace": pace, "flirtation_level": flirtation_level }

    return analysis