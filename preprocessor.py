# In preprocessor.py
import re
import spacy
import yake
import requests
from typing import List, Dict
# NEW: Library for expanding contractions like "don't" -> "do not"
import contractions


STOPWORDS = {
    "a", "about", "above", "across", "after", "afterwards", "again", "against", "all", "almost", "alone", "along",
    "already", "also", "although", "always", "am", "among", "amongst", "amoungst", "amount", "an", "and", "another",
    "any", "anyhow", "anyone", "anything", "anyway", "anywhere", "are", "around", "as", "at", "back", "be",
    "became", "because", "become", "becomes", "becoming", "been", "before", "beforehand", "behind", "being",
    "below", "beside", "besides", "between", "beyond", "bill", "both", "bottom", "but", "by", "call", "can",
    "cannot", "cant", "co", "computer", "con", "could", "couldnt", "cry", "de", "describe", "detail", "do",
    "done", "down", "due", "during", "each", "eg", "eight", "either", "eleven", "else", "elsewhere", "empty",
    "enough", "etc", "even", "ever", "every", "everyone", "everything", "everywhere", "except", "few", "fifteen",
    "fify", "fill", "find", "fire", "first", "five", "for", "former", "formerly", "forty", "found", "four",
    "from", "front", "full", "further", "get", "give", "go", "had", "has", "hasnt", "have", "he", "hence", "her",
    "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how",
    "however", "hundred", "i", "ie", "if", "in", "inc", "indeed", "interest", "into", "is", "it", "its", "itself",
    "keep", "last", "latter", "latterly", "least", "less", "ltd", "made", "many", "may", "me", "meanwhile",
    "might", "mill", "mine", "more", "moreover", "most", "mostly", "move", "much", "must", "my", "myself",
    "name", "namely", "neither", "never", "nevertheless", "next", "nine", "no", "nobody", "none", "noone",
    "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on", "once", "one", "only", "onto", "or",
    "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "own", "part", "per", "perhaps",
    "please", "put", "rather", "re", "same", "see", "seem", "seemed", "seeming", "seems", "serious", "several",
    "she", "should", "show", "side", "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone",
    "something", "sometime", "sometimes", "somewhere", "still", "such", "system", "take", "ten", "than",

    "that", "the", "their", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore",
    "therein", "thereupon", "these", "they", "thick", "thin", "third", "this", "those", "though", "three",
    "through", "throughout", "thru", "thus", "to", "together", "too", "top", "toward", "towards", "twelve",
    "twenty", "two", "un", "under", "until", "up", "upon", "us", "very", "via", "was", "we", "well", "were",
    "what", "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein",
    "whereupon", "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "whose",
    "why", "will", "with", "within", "without", "would", "yet", "you", "your", "yours", "yourself", "yourselves",
    # Social media / chat specific stopwords
    'lol', 'wbu', 'hmmm', 'n’t', 'nt', 'im', 'u', 'r', 'ur', 'y', 'tho', 'btw', 'omg', 'idk', 'tbh', 'imo', 'irl', 'fr',
    'ikr', 'smh', 'ily', 'wyd', 'brb', 'gonna', 'wanna', 'gotta', 'kinda', 'hey', 'hi', 'hello', 'sup', 'yo'
}


# --- Service Initialization ---
try:
    nlp = spacy.load("en_core_web_trf")
    # NEW: Upgraded, comprehensive stopword list for social media
    for word in STOPWORDS:
        nlp.Defaults.stop_words.add(word)
    print("spaCy model 'en_core_web_trf' loaded and customized successfully.")
except OSError:
    print("Spacy model 'en_core_web_trf' not found. Please run: python -m spacy download en_core_web_trf")
    nlp = None

# --- SlangHandler for Dynamic Slang Lookup (from previous version) ---
class SlangHandler:
    def __init__(self):
        self.api_url = "https://api.urbandictionary.com/v0/define"
        self.cache: Dict[str, bool] = {}
    def is_known_slang(self, term: str) -> bool:
        term = term.lower()
        if term in self.cache: return self.cache[term]
        try:
            response = requests.get(self.api_url, params={"term": term}, timeout=2)
            response.raise_for_status()
            is_slang = bool(response.json().get("list"))
            self.cache[term] = is_slang
            return is_slang
        except (requests.RequestException, ValueError):
            self.cache[term] = False
            return False
slang_handler = SlangHandler()

# --- Constants for Text Manipulation ---
NOISE_TERMS = {'hmmmm', 'mine', 'mind', 'faves', 'a bit lol'}
VALID_POS = {'NOUN', 'PROPN', 'VERB', 'ADJ'}

def _is_noise(phrase: str, doc: spacy.tokens.Doc) -> bool:
    phrase_lower = phrase.lower()
    if phrase_lower in NOISE_TERMS: return True
    tokens = [token for token in doc if token.text.lower() in phrase_lower]
    if tokens and all(token.pos_ in {'PRON', 'DET', 'AUX', 'PART', 'INTJ'} for token in tokens): return True
    if len(phrase.split()) <= 2 and slang_handler.is_known_slang(phrase): return False
    return False

def _shorten_phrase(phrase: str) -> str:
    if len(phrase.split()) <= 3: return phrase
    kw_extractor = yake.KeywordExtractor(lan="en", n=2, top=1, features=None)
    keywords = kw_extractor.extract_keywords(phrase)
    return keywords[0][0] if keywords else phrase

def extract_canonical_phrases(text: str) -> List[str]:
    if not nlp or not text: return []
    
    # --- NEW: Normalization Pipeline ---
    # 1. Expand contractions ("don't" -> "do not")
    text = contractions.fix(text)
    # 2. Normalize repeated letters ("soooo" -> "soo")
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    
    doc = nlp(text.lower())
    
    candidate_phrases = [chunk.text for chunk in doc.noun_chunks]
    filtered_phrases = [p for p in candidate_phrases if not _is_noise(p, doc) and len(p) > 3]
    canonical_phrases = [_shorten_phrase(p) for p in filtered_phrases]
    
    return list(dict.fromkeys(canonical_phrases))

def clean_text(text: str) -> str:
    if not isinstance(text, str): return ""
    return re.sub(r'\s+', ' ', text).strip()

def clean_and_truncate(conversation_history: list, max_turns: int = 20) -> list:
    if not conversation_history: return []
    truncated_history = conversation_history[-max_turns:]
    cleaned_history = []
    for turn in truncated_history:
        if isinstance(turn, dict) and "content" in turn:
            cleaned_turn = turn.copy()
            cleaned_turn["content"] = clean_text(cleaned_turn["content"])
            cleaned_history.append(cleaned_turn)
    return cleaned_history