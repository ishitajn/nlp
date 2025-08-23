import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

def _download_nltk_data_if_needed():
    try:
        # Check if the vader_lexicon is available
        nltk.data.find('sentiment/vader_lexicon.zip')
    except (LookupError, nltk.downloader.DownloadError):
        # Download the vader_lexicon if not found
        print("NLTK 'vader_lexicon' not found. Downloading...")
        nltk.download('vader_lexicon', quiet=True)
        print("'vader_lexicon' downloaded.")

# Ensure NLTK data is available before initializing the analyzer
_download_nltk_data_if_needed()
sia = SentimentIntensityAnalyzer()

def tone_mode(text):
    comp = sia.polarity_scores(text)["compound"]
    if comp >= 0.4: return "playful"
    if comp <= -0.2: return "serious"
    return "neutral"

from .transition import transition_score

def rerank(candidates, meta, recent_texts, used_ids, profile_tags, recent_tags):
    # candidates: list[(idx, cosine_sim)]
    mode = tone_mode(" ".join(recent_texts[-3:]))
    scored = []
    for idx, sim in candidates:
        m = meta[idx]
        tags = set(m.get("tags", []))

        novelty = 0.15 if m["id"] not in used_ids else -0.25
        prof = 0.30 if profile_tags.intersection(tags) else 0.0
        tone = 0.15 if (mode == "playful" and m["category"] in ["dating", "sexual", "romance", "playful"]) else 0.0
        trans = transition_score(recent_tags, tags)

        # Adjusted weights to prioritize profile and transition
        final = 0.4 * sim + novelty + prof + tone + trans
        scored.append((final, idx))

    scored.sort(reverse=True)
    return scored
