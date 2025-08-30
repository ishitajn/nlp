"""
Centralized service container for initializing and managing all external services,
ML models, and configurations.
"""
import json
import re
import spacy
import yake
from typing import Dict, List

from preprocessor import SlangHandler
from embedder import Embedder
from pysentimiento import create_analyzer
from planner import PlannerService

class Services:
    """A container for all shared services and configurations."""
    def __init__(self, config_dir: str = "config"):
        print("Initializing services...")
        # Load configurations
        self.config = {
            "paths": {
                "stopwords": f"{config_dir}/stopwords.json",
                "semantic_concepts": f"{config_dir}/semantic_concepts.json",
                "topic_metadata": "data/topic_metadata.json",
                "slang_cache": "data/slang_cache.json"
            }
        }
        stopwords = self._load_json(self.config['paths']['stopwords'])
        concept_definitions = self._load_json(self.config['paths']['semantic_concepts'])

        # Compile regex patterns for analysis schema
        # This logic was previously in context_engine.py
        analysis_schema_strings = {
            "phases": { "Icebreaker": [r'\b(h(i|e+y+|ello)|how (are |u )?(you|u)( doin)?|your profile|we matched)\b'], "Rapport Building": [r'\b(tell me more|what about you|hobbies|passions|family|career|work|job|hiking|trip|travel)\b'], "Escalation": [r'\b(tension|desire|imagining|in person|what if|chemistry)\b'], "Explicit Banter": [r'\b(fuck|sex|nude|kink|sexting|horny|aroused)\b'], "Logistics": [r'\b(when are you free|let\'s meet|what\'s your number|schedule|date)\b'], },
            "tones": { "Playful": [r'\b(haha|lol|lmao|kidding|teasing|banter|playful|cheeky)\b', r'[😉😜😏]'], "Serious": [r'\b(to be honest|actually|my values|looking for|seriously)\b'], "Romantic": [r'\b(connection|special|beautiful|chemistry|heart|adore|lovely)\b'], "Complimentary": [r'\b(great|amazing|impressive|gorgeous|handsome|hot|sexy|cute)\b'], "Vulnerable": [r'\b(my feelings|i feel|struggle|opening up is hard|i feel safe with you)\b'], },
            "intents": { "Gathering Information": [r'\?'], "Building Comfort": [r'\b(that makes sense|i understand|thank you for sharing)\b'], "Testing Boundaries": [r'\b(what are you into|how adventurous|are you open to)\b'], "Making Plans": [r'\b(we should|let\'s|are you free|wanna grab)\b'], "Expressing Desire": [r'\b(i want you|i need you|can\'t stop thinking about you|i desire you)\b'], }
        }
        self.analysis_schema = {
            category: { tag_name: [re.compile(p, re.IGNORECASE) for p in patterns] for tag_name, patterns in rules.items() }
            for category, rules in analysis_schema_strings.items()
        }
        self.question_starters_regex = re.compile(r'^(who|what|where|when|why|how|is|are|do|does|did|will|can|could|should|would|have|has|had|am|was|were|don\'t|isn\'t|aren\'t)\b', re.IGNORECASE)

        # Initialize NLP models and clients
        self.slang_handler = SlangHandler(cache_path=self.config['paths']['slang_cache'])
        self.nlp = self._init_spacy(stopwords)
        self.kw_extractor = yake.KeywordExtractor(lan="en", n=2, top=1, features=None)
        self.sentiment_analyzer = create_analyzer(task="sentiment", lang="en")
        self.emotion_analyzer = create_analyzer(task="emotion", lang="en")
        self.planner = PlannerService()
        self.embedder = Embedder()

        # Pre-compute concept embeddings
        self.concept_embeddings = {
            name: self.embedder.encode_cached([description])[0]
            for name, description in concept_definitions.items()
        }
        print("Services initialized successfully.")

    def _load_json(self, path: str) -> Dict | List:
        with open(path, 'r') as f:
            return json.load(f)

    def _init_spacy(self, stopwords: List[str]) -> spacy.Language:
        """Initializes the spaCy model with custom stopwords."""
        try:
            nlp = spacy.load("en_core_web_trf")
            for word in stopwords:
                nlp.Defaults.stop_words.add(word)
            return nlp
        except OSError:
            raise RuntimeError("spaCy model 'en_core_web_trf' not found. Please run: python -m spacy download en_core_web_trf")

    def get_preprocessor_args(self) -> Dict:
        """Returns a dictionary of services needed by the preprocessor functions."""
        return {
            "nlp": self.nlp,
            "kw_extractor": self.kw_extractor,
            "slang_handler": self.slang_handler
        }
