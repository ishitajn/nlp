# In preprocessor.py
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

def download_nltk_data():
    """Downloads required NLTK data if not already present."""
    try:
        nltk.data.find('tokenizers/punkt')
    except nltk.downloader.DownloadError:
        nltk.download('punkt')
    try:
        nltk.data.find('corpora/stopwords')
    except nltk.downloader.DownloadError:
        nltk.download('stopwords')
    try:
        nltk.data.find('corpora/wordnet')
    except nltk.downloader.DownloadError:
        nltk.download('wordnet')

# Call the download function once when the module is loaded.
download_nltk_data()

# Custom stopwords, including common chat slang and conversational filler
CUSTOM_STOPWORDS = set(stopwords.words('english')) | {
    'lol', 'haha', 'hehe', 'ok', 'okay', 'yeah', 'yes', 'no', 'nah',
    'im', 'u', 'r', 'ur', 'y', 'tho', 'btw', 'omg', 'idk', 'tbh', 'imo',
    'hey', 'hi', 'hello', 'sup', 'yo',
    'like', 'actually', 'basically', 'really', 'gonna', 'wanna'
}

lemmatizer = WordNetLemmatizer()

def preprocess_text(text: str) -> str:
    """
    Cleans, tokenizes, removes stopwords, and lemmatizes a string of text.
    Returns a cleaned string.
    """
    if not isinstance(text, str):
        return ""

    # Remove URLs, mentions, and non-alphanumeric characters (except spaces)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Tokenize and lowercase
    tokens = word_tokenize(text.lower())

    # Remove stopwords and lemmatize
    lemmatized_tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in CUSTOM_STOPWORDS and len(word) > 1
    ]

    return " ".join(lemmatized_tokens)
