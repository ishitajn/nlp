import spacy

try:
    nlp = spacy.load("en_core_web_sm")
    print("spaCy model 'en_core_web_sm' loaded successfully.")
except OSError:
    print("Spacy model 'en_core_web_sm' not found. Please run: python -m spacy download en_core_web_sm")
    nlp = None
