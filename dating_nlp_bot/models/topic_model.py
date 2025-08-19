import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict

from .embeddings import EmbeddingModel
from dating_nlp_bot.utils.keywords import TOPIC_KEYWORDS

class EnhancedTopicModel:
    def __init__(self, num_clusters=5):
        self.num_clusters = num_clusters
        self.embedding_model = EmbeddingModel()
        self.kmeans = KMeans(n_clusters=self.num_clusters, random_state=42, n_init=10)
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=10)

    def get_topics(self, messages: list[str]) -> tuple[dict, list[int]]:
        """
        Gets topics from a list of messages using clustering and TF-IDF.
        Returns a tuple of (topic_map, labels).
        """
        if not messages or len(messages) < self.num_clusters:
            return {}, []

        embeddings = self.embedding_model.get_embeddings(messages)
        labels = self.kmeans.fit_predict(embeddings)

        clusters = defaultdict(list)
        for i, label in enumerate(labels):
            clusters[label].append(messages[i])

        cluster_top_keywords = {}
        for i in range(self.num_clusters):
            cluster_docs = clusters[i]
            if not cluster_docs:
                continue

            tfidf_matrix = self.vectorizer.fit_transform(cluster_docs)
            feature_names = self.vectorizer.get_feature_names_out()
            top_keywords = [feature_names[j] for j in tfidf_matrix.sum(axis=0).argsort()[0, ::-1]][:3]
            cluster_top_keywords[i] = top_keywords

        topic_map = self.map_keywords_to_topics(cluster_top_keywords)
        return topic_map, labels.tolist()

    def map_keywords_to_topics(self, cluster_keywords: dict) -> dict:
        """
        Maps cluster keywords to predefined topic categories.
        """
        topic_map = defaultdict(list)
        topic_map['female_centric'] = defaultdict(list)

        general_topics = ["travel", "food", "sports", "career", "flirt", "sexual", "emotions"]
        female_centric_topics = ["fashion", "wellness", "hobbies", "social", "relationships"]

        for cluster_id, keywords in cluster_keywords.items():
            for keyword in keywords:
                for topic, topic_kws in TOPIC_KEYWORDS.items():
                    if str(keyword) in topic_kws:
                        if topic in general_topics:
                            topic_map[topic].append(str(keyword))
                        elif topic in female_centric_topics:
                            topic_map['female_centric'][topic].append(str(keyword))
                        # Break to avoid adding to multiple categories
                        break

        # Deduplicate keywords
        for key, value in topic_map.items():
            if isinstance(value, list):
                topic_map[key] = list(set(value))
            elif isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    topic_map[key][sub_key] = list(set(sub_value))

        return dict(topic_map)
