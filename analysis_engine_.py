# In app/svc/analysis_engine.py

import re
import spacy
import numpy as np
from collections import Counter
from typing import List, Dict, Any
from sklearn.metrics.pairwise import cosine_similarity
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from embedder import embedder_service

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

def _deduplicate_and_merge_topics(topics: List[str], similarity_threshold=0.85) -> List[str]:
    """Intelligently merges semantically similar topics."""
    if not topics: return []
    
    vectors = embedder_service.encode_cached(topics)
    if vectors.shape[0] == 0: return []

    similarity_matrix = cosine_similarity(vectors)
    
    merged_topics = set()
    processed_indices = set()

    for i in range(len(topics)):
        if i in processed_indices: continue
        
        similar_indices = [j for j, score in enumerate(similarity_matrix[i]) if score > similarity_threshold]
        
        representative_topic = max([topics[k] for k in similar_indices], key=len)
        merged_topics.add(representative_topic)
        
        processed_indices.update(similar_indices)
        
    return list(merged_topics)

def _detect_inside_jokes(doc: spacy.tokens.Doc) -> List[str]:
    propn_candidates = [chunk.text.strip() for chunk in doc.noun_chunks if chunk.root.pos_ == 'PROPN']
    counts = Counter(propn_candidates)
    common_words = {'orion', 'iceland', 'america'}
    return [text for text, count in counts.items() if count > 1 and text.lower() not in common_words]

def run_full_analysis(my_profile: str, their_profile: str, turns: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not nlp: raise RuntimeError("spaCy model is not loaded.")

    conversation_history_str = "\n".join([f"{t.get('content', '')}" for t in turns])
    full_text_for_rules = f"{my_profile} {their_profile} {conversation_history_str}"
    full_text_lower = full_text_for_rules.lower()
    doc = nlp(conversation_history_str)
    
    # --- 1. Dynamic Topic Discovery ---
    discovered_chunks = set()
    stopwords = {'i', 'you', 'me', 'my', 'it', 'that', 'what', 'wbu', 'hmmmm', 'lol', 'haha', 'the first thing', 'a bit', 'thing', 'hihi'}
    for chunk in doc.noun_chunks:
        clean_chunk = _clean_topic_label(chunk.text)
        if chunk.root.pos_ != 'PRON' and clean_chunk not in stopwords and len(clean_chunk) > 3:
            discovered_chunks.add(clean_chunk)
    
    if not discovered_chunks:
        # Return a default empty structure if no topics are found
        empty_state = { "topics": {"focus": [], "avoid": [], "neutral": [], "sensitive": [], "fetish": [], "sexual": [], "inside_jokes": []}, "recent_topics": [], "topic_occurrence_heatmap": {}, "topic_recency_heatmap": {}, "topic_mapping": {} }
        return { "conversation_state": empty_state, "analysis": {}, "engagement_metrics": {}, "sentiment_analysis": {} }

    # --- 2. Semantic Deduplication ---
    unique_topics = _deduplicate_and_merge_topics(list(discovered_chunks))

    # --- 3. Semantic Mapping & Categorization ---
    unique_topic_vectors = embedder_service.encode_cached(unique_topics)
    if unique_topic_vectors.shape[0] == 0: # Check if vectors were generated
        empty_state = { "topics": {"focus": [], "avoid": [], "neutral": [], "sensitive": [], "fetish": [], "sexual": [], "inside_jokes": []}, "recent_topics": [], "topic_occurrence_heatmap": {}, "topic_recency_heatmap": {}, "topic_mapping": {} }
        return { "conversation_state": empty_state, "analysis": {}, "engagement_metrics": {}, "sentiment_analysis": {} }

    similarity_matrix = cosine_similarity(unique_topic_vectors, canonical_topic_vectors)
    
    categorized_raw_topics = { "neutral": set(), "sensitive": set(), "fetish": set(), "sexual": set(), "intimacy": set() }
    topic_mapping = {name: [] for name in canonical_topic_names}
    canonical_topic_list_for_heatmaps = []

    similarity_threshold = 0.4
    for i, chunk in enumerate(unique_topics):
        best_match_index = np.argmax(similarity_matrix[i])
        if similarity_matrix[i][best_match_index] > similarity_threshold:
            canonical_topic = canonical_topic_names[best_match_index]
            category = TOPIC_TO_CATEGORY_MAP.get(canonical_topic, "neutral")
            categorized_raw_topics[category].add(chunk)
            topic_mapping[canonical_topic].append(chunk)
            canonical_topic_list_for_heatmaps.append(canonical_topic)

    # --- 4. Heatmap Calculation on Canonical Topics ---
    occurrence_heatmap = dict(Counter(canonical_topic_list_for_heatmaps).most_common(15))
    
    last_occurrence = {}
    for i, turn in enumerate(turns):
        turn_lower = turn['content'].lower()
        for canonical_topic, chunks in topic_mapping.items():
            if any(chunk.lower() in turn_lower for chunk in chunks):
                last_occurrence[canonical_topic] = i
    sorted_by_recency = sorted(last_occurrence.items(), key=lambda item: item[1], reverse=True)
    recency_heatmap = {topic: rank + 1 for rank, (topic, index) in enumerate(sorted_by_recency[:15])}

    # --- 5. Build the Final Conversation State ---
    all_categorized_topics = list(categorized_raw_topics["neutral"] | categorized_raw_topics["sexual"] | categorized_raw_topics["intimacy"])
    
    # *** THE FIX IS HERE ***
    # The variable `all_topics` is now correctly named `all_categorized_topics`.
    top_20_topics = [t for t, c in Counter(all_categorized_topics).most_common(20)]

    inside_jokes = _detect_inside_jokes(doc)

    conversation_state = {
        "topics": {
            "focus": top_20_topics, "avoid": [],
            "neutral": [t for t in categorized_raw_topics["neutral"] if t in top_20_topics],
            "sensitive": [t for t in categorized_raw_topics["sensitive"] if t in top_20_topics],
            "fetish": [t for t in categorized_raw_topics["fetish"] if t in top_20_topics],
            "sexual": [t for t in categorized_raw_topics["sexual"] if t in top_20_topics],
            "inside_jokes": inside_jokes
        },
        "recent_topics": [topic for topic, rank in recency_heatmap.items()],
        "topic_occurrence_heatmap": occurrence_heatmap,
        "topic_recency_heatmap": recency_heatmap,
        "topic_mapping": {k: v for k, v in topic_mapping.items() if v}
    }

    # --- 6. Schema-Based Tagging (for Structure) ---
    detected_tags = { "detected_phases": set(), "detected_tones": set(), "detected_intents": set() }
    for category, rules in ANALYSIS_SCHEMA.items():
        for tag_name, patterns in rules.items():
            if any(re.search(pattern, full_text_lower) for pattern in patterns):
                detected_tags[f"detected_{category.lower()}"].add(tag_name)

    analysis = {k: list(v) for k, v in detected_tags.items()}
    analysis["conversation_state"] = conversation_state

    # --- 7. Quantitative & Sentiment Analysis ---
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