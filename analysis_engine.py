# In app/svc/analysis_engine.py

import re
import spacy
from spacy.matcher import PhraseMatcher
from collections import Counter
from typing import List, Dict, Any
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# --- Load the HIGH-ACCURACY Transformer NLP model once on startup ---
try:
    # This is the key upgrade: using the transformer model for SOTA accuracy.
    nlp = spacy.load("en_core_web_trf")
    print("Successfully loaded high-accuracy transformer model: en_core_web_trf")
except OSError:
    print("Spacy model 'en_core_web_trf' not found. Please run 'pip install spacy-transformers && python -m spacy download en_core_web_trf'")
    nlp = None

# ==============================================================================
# == THE DEFINITIVE, MASSIVELY EXPANDED CUSTOM VOCABULARY                     ==
# ==============================================================================
# This vocabulary is curated from internet slang, dating advice, and common conversation patterns.
CUSTOM_VOCAB = {
    # Core Activities & Concepts
    "Stargazing": ["stargazing", "constellation", "big dipper", "orion", "the stars", "under the stars"],
    "Cuddling": ["cuddle", "cuddling", "snuggling", "snuggle", "cozy night in", "getting cozy"],
    "Coffee Date": ["coffee date", "grab a coffee", "getting coffee"],
    "Smoking Weed": ["smoking weed", "getting high", "420 friendly", "edibles", "cruising smoking"],
    "Deep Conversations": ["deep conversations", "meaningful chat", "real talk"],
    "Getting to Know You": ["getting to know you", "getting to know each other", "learning about you"],
    "Taking It Slow": ["take it slow", "slowing down", "let's slow down", "no rush", "at your pace"],
    "Long Distance": ["live far away", "long distance", "the distance"],
    "Making Plans": ["this weekend", "next few weeks", "planning our night", "making it happen", "let's plan something"],
    "Chemistry": ["our chemistry", "the chemistry", "a spark", "a connection", "the vibe"],
    "Late-Night Vibe": ["late-night vibe", "ideal late-night"],
    "Adventurous Things": ["adventurous thing", "most adventurous", "sense of adventure"],
    "Warm Blankets": ["warm blankets"],
    "Life in Iceland": ["in iceland"],
    "Autism": ["my autism"],

    # Modern Dating & Social Media Slang
    "Rizz": ["rizz", "unspoken rizz"],
    "Sliding into DMs": ["slide into dms", "slid into your dms"],
    "Shooting Your Shot": ["shooting my shot", "shoot my shot"],
    "Vibe Check": ["vibe check", "passing the vibe check"],
    "The Ick": ["the ick", "got the ick"],
    "Ghosting": ["ghosting", "ghosted me"],
    "Breadcrumbing": ["breadcrumbing"],
    "Love Bombing": ["love bombing"],
    "Situationship": ["situationship"],
    "FWB (Friends with Benefits)": ["fwb", "friends with benefits"],
    "DTF (Down to Fuck)": ["dtf", "down to fuck"],
    "Body Count": ["body count"],
    "OnlyFans": ["onlyfans", "of content"],
    "Spicy Content": ["spicy content", "spicy pics"],
    "Link in Bio": ["link in bio"],

    # Flirting & Escalation
    "Playful Banter": ["playful banter", "good banter"],
    "Sexual Tension": ["sexual tension", "the tension"],
    "Testing the Waters": ["testing the waters", "feeling it out"],
    "Making a Move": ["making a move", "make the first move"],
    "Intimate Compliments": ["intimate compliments", "personal compliment"],

    # Sexual Health & Boundaries
    "Consent": ["consent", "enthusiastic consent", "is this okay?"],
    "Boundaries": ["my boundaries", "hard limits", "soft limits"],
    "Safe Words": ["safe word", "safewording"],
    "Aftercare": ["aftercare"],
    "Sexual Health": ["sexual health", "std testing", "sti check"],
}

# ==============================================================================
# == ANALYSIS SCHEMA (For Phases, Tones, and Intents ONLY)                    ==
# ==============================================================================
ANALYSIS_SCHEMA = {
    "Phases": {
        "Icebreaker": [r'\b(h(i|ey|ello)|how are you|your profile|we matched)\b'],
        "Rapport Building": [r'\b(tell me more|what about you|hobbies|passions|family|career)\b'],
        "Escalation": [r'\b(tension|desire|imagining|in person|what if|chemistry)\b'],
        "Explicit Banter": [r'\b(fuck|sex|nude|kink|sexting|horny|aroused)\b'],
        "Logistics": [r'\b(when are you free|let\'s meet|what\'s your number|schedule|date)\b'],
    },
    "Tones": {
        "Playful": [r'\b(haha|lol|lmao|kidding|teasing|banter|playful|cheeky)\b', r'[😉😜😏]'],
        "Serious": [r'\b(to be honest|actually|my values|looking for|seriously)\b'],
        "Romantic": [r'\b(connection|special|beautiful|chemistry|heart|adore|lovely)\b'],
        "Complimentary": [r'\b(great|amazing|impressive|gorgeous|handsome|hot|sexy|cute)\b'],
        "Vulnerable": [r'\b(my feelings|i feel|struggle|opening up is hard|i feel safe with you)\b'],
    },
    "Intents": {
        "Gathering Information": [r'\?'],
        "Building Comfort": [r'\b(that makes sense|i understand|thank you for sharing)\b'],
        "Testing Boundaries": [r'\b(what are you into|how adventurous|are you open to)\b'],
        "Making Plans": [r'\b(we should|let\'s|are you free|wanna grab)\b'],
        "Expressing Desire": [r'\b(i want you|i need you|can\'t stop thinking about you|i desire you)\b'],
    }
}

# Keywords to help categorize the dynamically extracted topics
SEXUAL_TOPIC_KEYWORDS = ['sex', 'kink', 'fetish', 'bdsm', 'sexting', 'nude', 'orgasm', 'cuddle', 'kiss', 'touch', 'fwb', 'dtf', 'body count', 'onlyfans']
SENSITIVE_TOPIC_KEYWORDS = ['autism', 'ex', 'breakup', 'struggle', 'insecurity', 'anxiety', 'depression', 'ghosting', 'ick']

sentiment_analyzer = SentimentIntensityAnalyzer()

# Initialize the PhraseMatcher
if nlp:
    phrase_matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    for topic, patterns in CUSTOM_VOCAB.items():
        pattern_docs = [nlp.make_doc(text) for text in patterns]
        phrase_matcher.add(topic, pattern_docs)

def _generate_suggestion_prompts(analysis: Dict[str, Any], conversation_history: str) -> Dict[str, str]:
    """Generates specific, high-quality prompts for an external LLM based on the rich analysis."""
    context_summary = f"""- Conversation Phase: {', '.join(analysis['detected_phases']) or 'N/A'}
- Detected Tones: {', '.join(analysis['detected_tones']) or 'N/A'}
- Key Topics: {', '.join(analysis['conversation_state']['topics']['focus']) or 'N/A'}
- Overall Sentiment: {analysis['sentiment_analysis']['overall']}
- Flirtation Score (0-10): {analysis['sentiment_analysis']['flirtation_score']:.1f}"""

    base_prompt = f"""
### CONVERSATION ANALYSIS
{context_summary}

### RECENT CONVERSATION
{conversation_history}

### TASK
Based on the analysis and conversation, generate two creative, concise suggestion phrases (3-5 words max) for the category below. The suggestions must be highly relevant. Output ONLY a JSON array of two strings. Example: ["first suggestion", "second suggestion"]

Category: """

    return {
        "topics": base_prompt + "New Conversation Topics",
        "questions": base_prompt + "Engaging Questions to Ask",
        "intimacy": base_prompt + "Ways to Build Intimacy",
        "sexual": base_prompt + "Flirty or Sexual Escalations"
    }

def run_full_analysis(my_profile: str, their_profile: str, turns: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Performs the entire analysis pipeline using the best available tools:
    1. A high-accuracy Transformer model (en_core_web_trf).
    2. PhraseMatcher with a massive custom vocabulary for key topics.
    3. spaCy's noun chunking for general, dynamic topics.
    4. Regex for conversation structure (Phases, Tones, Intents).
    """
    if not nlp:
        raise RuntimeError("spaCy model is not loaded. Please install and download it.")

    conversation_history_str = "\n".join([f"{t.get('content', '')}" for t in turns])
    full_text_for_rules = f"{my_profile} {their_profile} {conversation_history_str}"
    full_text_lower = full_text_for_rules.lower()

    # --- 1. Dynamic Topic Extraction (Hybrid Approach) ---
    doc = nlp(conversation_history_str)
    
    matches = phrase_matcher(doc)
    custom_topics = {doc.vocab.strings[match_id] for match_id, start, end in matches}

    general_topics = set()
    stopwords = {'i', 'you', 'me', 'my', 'it', 'that', 'a', 'the', 'what', 'wbu', 'hmmmm', 'lol', 'haha', 'the first thing', 'a bit'}
    for chunk in doc.noun_chunks:
        clean_chunk = chunk.text.lower().strip()
        if clean_chunk not in stopwords and len(clean_chunk) > 3 and len(clean_chunk.split()) < 5:
            general_topics.add(clean_chunk)
            
    all_topics = list(custom_topics) + [t for t in general_topics if t not in custom_topics]
    focus_topics = all_topics[:15]

    # --- 2. Schema-Based Tagging (for Structure) ---
    detected_tags = { "detected_phases": set(), "detected_tones": set(), "detected_intents": set() }
    for category, rules in ANALYSIS_SCHEMA.items():
        for tag_name, patterns in rules.items():
            if any(re.search(pattern, full_text_lower) for pattern in patterns):
                if category == "Phases": detected_tags["detected_phases"].add(tag_name)
                elif category == "Tones": detected_tags["detected_tones"].add(tag_name)
                elif category == "Intents": detected_tags["detected_intents"].add(tag_name)

    analysis = {k: list(v) for k, v in detected_tags.items()}

    # --- 3. Categorize Dynamic Topics ---
    sexual_topics = {t for t in focus_topics if any(kw in t.lower() for kw in SEXUAL_TOPIC_KEYWORDS)}
    sensitive_topics = {t for t in focus_topics if any(kw in t.lower() for kw in SENSITIVE_TOPIC_KEYWORDS)}
    neutral_topics = {t for t in focus_topics if t not in sexual_topics and t not in sensitive_topics}

    conversation_state = {
        "topics": {
            "focus": list(focus_topics), "avoid": [], "neutral": list(neutral_topics),
            "sensitive": list(sensitive_topics), "fetish": [], "sexual": list(sexual_topics)
        },
        "recent_topics": list(focus_topics)
    }
    analysis["conversation_state"] = conversation_state

    # --- 4. Quantitative & Sentiment Analysis ---
    sentiment_scores = sentiment_analyzer.polarity_scores(conversation_history_str)
    compound_score = sentiment_scores['compound']
    sentiment = "neutral"
    if compound_score > 0.5: sentiment = "very positive"
    elif compound_score > 0.05: sentiment = "positive"
    elif compound_score < -0.5: sentiment = "very negative"
    elif compound_score < -0.05: sentiment = "negative"

    flirt_keywords = ['flirt', 'teasing', 'sexy', 'hot', 'desire', 'tension', 'imagining', 'irresistible', '😉', '😏', 'cuddle', 'kiss']
    flirt_score = sum(full_text_lower.count(kw) for kw in flirt_keywords)
    flirtation_score = min(10, flirt_score * 2.0)

    analysis["sentiment_analysis"] = {
        "overall": sentiment, "compound_score": compound_score, "flirtation_score": flirtation_score
    }

    # --- 5. Generate Suggestion Prompts ---
    suggestion_prompts = _generate_suggestion_prompts(analysis, conversation_history_str)

    # --- 6. Assemble Final Output ---
    return {
        "analysis_results": analysis,
        "suggestion_prompts": suggestion_prompts
    }