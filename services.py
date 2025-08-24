import spacy

try:
    nlp = spacy.load("en_core_web_sm")
    # Customize the stop word list
    words_to_keep = {"say", "so", "much"}
    for word in words_to_keep:
        nlp.Defaults.stop_words.discard(word) # Use discard to avoid errors if word is not in set
    print("spaCy model 'en_core_web_sm' loaded and customized successfully.")
except OSError:
    print("Spacy model 'en_core_web_sm' not found. Please run: python -m spacy download en_core_web_sm")
    nlp = None
