# In topic_discoverer.py

import spacy
import numpy as np
from typing import List, Dict
from collections import defaultdict
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity

from embedder import embedder_service

# Load spacy model
try:
    nlp = spacy.load("en_core_web_trf")
    print("Successfully loaded high-accuracy transformer model: en_core_web_trf")
except OSError:
    print("Spacy model 'en_core_web_trf' not found. Please run 'pip install spacy-transformers && python -m spacy download en_core_web_trf'")
    nlp = None


def _clean_phrase(text: str) -> str:
    """Removes leading/trailing whitespace, articles, and possessive pronouns."""
    text = text.lower().strip()
    # Remove articles
    if text.startswith("the ") or text.startswith("a ") or text.startswith("an "):
        text = text.split(" ", 1)[1]
    # Remove possessive pronouns
    possessives = ["my ", "your ", "his ", "her ", "its ", "our ", "their "]
    for p in possessives:
        if text.startswith(p):
            text = text[len(p):]
            break  # A phrase shouldn't start with multiple possessives
    return text


def _extract_keyphrases(doc: spacy.tokens.Doc) -> List[str]:
    """
    Extracts noun chunks and verb phrases from a spaCy Doc.
    """
    keyphrases = set()
    stopwords = {'i', 'you', 'me', 'my', 'it', 'that', 'what', 'wbu', 'hmmmm', 'lol', 'haha', 'the first thing', 'a bit', 'thing', 'hihi', 'im'}

    # Extract noun chunks
    for chunk in doc.noun_chunks:
        clean_chunk = _clean_phrase(chunk.text)
        if chunk.root.pos_ != 'PRON' and clean_chunk not in stopwords and len(clean_chunk.split()) <= 4 and len(clean_chunk) > 3:
            keyphrases.add(clean_chunk)

    # Extract verb phrases (verb + object)
    for token in doc:
        if token.pos_ == 'VERB':
            for child in token.children:
                if child.dep_ == 'dobj' and child.pos_ == 'NOUN': # Direct object
                    phrase = f"{token.lemma_} {child.text}"
                    clean_phrase_text = _clean_phrase(phrase)
                    if clean_phrase_text not in stopwords and len(clean_phrase_text.split()) <= 4:
                        keyphrases.add(clean_phrase_text)

    return list(keyphrases)


def _cluster_keyphrases(keyphrases: List[str], distance_threshold=0.4) -> Dict[int, List[str]]:
    """
    Clusters keyphrases based on their semantic similarity.
    """
    if len(keyphrases) < 2:
        return {0: keyphrases} if keyphrases else {}

    embeddings = embedder_service.encode_cached(keyphrases)

    # Using Agglomerative Clustering
    clustering = AgglomerativeClustering(
        n_clusters=None,
        affinity='cosine',
        linkage='average',
        distance_threshold=distance_threshold
    )
    labels = clustering.fit_predict(embeddings)

    clusters = defaultdict(list)
    for keyphrase, label in zip(keyphrases, labels):
        clusters[label].append(keyphrase)

    return clusters


def _label_clusters(clusters: Dict[int, List[str]]) -> Dict[str, List[str]]:
    """
    Finds a representative label for each cluster.
    Filters out clusters that are too small.
    """
    labeled_clusters = {}
    for cluster_id, keyphrases in clusters.items():
        if len(keyphrases) < 2:  # Skip small clusters
            continue

        embeddings = embedder_service.encode_cached(keyphrases)
        centroid = np.mean(embeddings, axis=0, keepdims=True)

        similarities = cosine_similarity(embeddings, centroid)

        label_index = np.argmax(similarities)
        label = keyphrases[label_index]

        # Ensure the label itself is not in the list of phrases for that topic
        other_phrases = [p for i, p in enumerate(keyphrases) if i != label_index]

        labeled_clusters[label] = other_phrases

    return labeled_clusters


def discover_topics(conversation_text: str) -> Dict:
    """
    Discovers topics in a conversation using keyphrase extraction and clustering.
    Returns a dictionary containing the labeled clusters and all extracted keyphrases.
    """
    if not nlp:
        raise RuntimeError("spaCy model is not loaded.")

    doc = nlp(conversation_text)
    keyphrases = _extract_keyphrases(doc)

    if len(keyphrases) < 2:
        return {"labeled_clusters": {}, "all_keyphrases": keyphrases}

    clusters = _cluster_keyphrases(keyphrases)
    labeled_clusters = _label_clusters(clusters)

    return {
        "labeled_clusters": labeled_clusters,
        "all_keyphrases": keyphrases
    }
