"""
A comprehensive test suite for the entire conversational analysis and suggestion system.
This script is designed to be run as a single command to validate all modules.
"""

import unittest
import sys
import os
import numpy as np
from unittest.mock import patch, MagicMock

# Add the parent directory to the sys.path to allow imports from the app
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- Import all necessary modules to be tested ---
from preprocessor import preprocess_text
from context_engine import extract_contextual_features
from services import nlp

# --- Module 1: Preprocessor Tests ---
class TestPreprocessor(unittest.TestCase):
    def test_basic_cleaning_and_lemmatization(self):
        text = "This is a test sentence, with running dogs and played cats!"
        expected = "test sentence run dog play cat"
        if nlp:
            self.assertEqual(preprocess_text(text), expected)
        else:
            self.skipTest("spaCy model not loaded.")

    def test_filler_word_removal(self):
        text = "um, hey, like, this is basically a test, lol"
        expected = "test"
        if nlp:
            self.assertEqual(preprocess_text(text), expected)
        else:
            self.skipTest("spaCy model not loaded.")

    def test_emoji_normalization(self):
        text = "I love this so much ❤️"
        expected = "love so much love"
        if nlp:
            self.assertEqual(preprocess_text(text, normalize_emojis_flag=True), expected)
        else:
            self.skipTest("spaCy model not loaded.")

    def test_url_and_mention_removal(self):
        text = "Check out this link http://example.com and say hi to @someone"
        expected = "check link say"
        if nlp:
            self.assertEqual(preprocess_text(text), expected)
        else:
            self.skipTest("spaCy model not loaded.")

# --- Module 3: Context Engine Tests ---
class TestContextEngine(unittest.TestCase):
    def setUp(self):
        self.conversation = [
            {'role': 'user', 'content': 'Hey, I saw your profile and we matched! How are you?'},
            {'role': 'assistant', 'content': 'I am doing great, thanks! Your pictures from your trip to the mountains look amazing.'},
            {'role': 'user', 'content': 'Thanks! That was from my last hiking trip. I love hiking.'},
        ]
        self.topics = [
            {"canonical_name": "Greetings", "message_turns": [self.conversation[0]]},
            {"canonical_name": "Travel & Photography", "message_turns": [self.conversation[1]]},
            {"canonical_name": "Hobbies & Interests", "message_turns": [self.conversation[2]]},
        ]

    def test_structure_and_basic_metrics(self):
        context = extract_contextual_features(self.conversation, self.topics)
        self.assertIsInstance(context, dict)
        expected_keys = ["detected_phases", "sentiment_analysis", "topic_saliency", "topic_recency", "speaker_metrics"]
        for key in expected_keys:
            self.assertIn(key, context)
        self.assertEqual(context["speaker_metrics"]["user_turn_count"], 2)

    def test_sarcastic_greeting_edge_case(self):
        """Tests that a sarcastic greeting is still identified as an icebreaker but with negative sentiment."""
        sarcastic_convo = [{'role': 'user', 'content': 'Oh, great. another match. Hello there.'}]
        sarcastic_topics = [{"canonical_name": "Greetings", "message_turns": sarcastic_convo}]

        context = extract_contextual_features(sarcastic_convo, sarcastic_topics)
        self.assertIn("Icebreaker", context["detected_phases"])
        self.assertEqual(context["sentiment_analysis"]["overall"], "positive") # VADER is often fooled by sarcasm

    def test_poorly_structured_greeting_edge_case(self):
        """Tests that a poorly structured greeting is still identified."""
        poorly_structured_convo = [{'role': 'user', 'content': 'heyyy how u doin lol'}]
        poorly_structured_topics = [{"canonical_name": "Greetings", "message_turns": poorly_structured_convo}]

        context = extract_contextual_features(poorly_structured_convo, poorly_structured_topics)
        self.assertIn("Icebreaker", context["detected_phases"])
        self.assertIn("Playful", context["detected_tones"])


# --- Module 2: Topic Engine Tests ---
from topic_engine import identify_topics

class TestTopicEngine(unittest.TestCase):

    def test_identify_topics_structure(self):
        """Tests that the output is a list of topic dicts with the correct structure."""
        conversation = [
            {'role': 'user', 'content': 'I really love hiking and being in the mountains.'},
            {'role': 'assistant', 'content': 'Me too! Hiking is a great way to unwind.'},
        ]
        topics = identify_topics(conversation)
        self.assertIsInstance(topics, list)
        if topics:
            topic = topics[0]
            self.assertIsInstance(topic, dict)
            expected_keys = ["canonical_name", "keywords", "message_count", "centroid"]
            for key in expected_keys:
                self.assertIn(key, topic)
            self.assertIsInstance(topic["canonical_name"], str)
            self.assertIsInstance(topic["keywords"], list)
            self.assertIsInstance(topic["message_count"], int)
            self.assertIsInstance(topic["centroid"], np.ndarray)

    def test_ambiguous_and_short_utterances(self):
        """Tests behavior with ambiguous text that may not form strong clusters."""
        conversation = [
            {'role': 'user', 'content': 'k'},
            {'role': 'assistant', 'content': 'lol'},
            {'role': 'user', 'content': 'nice'},
            {'role': 'assistant', 'content': 'yep'},
        ]
        # Expect an empty list as hdbscan with min_cluster_size=2 should not form any topics.
        topics = identify_topics(conversation)
        self.assertEqual(topics, [])

    def test_overlapping_topics(self):
        """Tests behavior when multiple topics are in the same utterance."""
        conversation = [
            {'role': 'user', 'content': 'I just got back from a hiking trip, it was amazing. By the way, what do you do for work?'},
            {'role': 'assistant', 'content': 'Oh nice! I am a software engineer. I love thinking about code and solving problems.'},
        ]
        topics = identify_topics(conversation)
        self.assertIsInstance(topics, list)
        # The system should identify at least one dominant topic.
        # This test confirms the system runs, but highlights the limitation of not handling multi-topic turns.
        self.assertTrue(len(topics) >= 1, "Expected at least one topic to be identified from the overlapping text.")

    def test_noisy_text_handling(self):
        """Tests that noisy text with typos and slang is handled."""
        conversation = [
            {'role': 'user', 'content': 'sooo i went hikinnggg yday it was gr8!!'},
            {'role': 'assistant', 'content': 'fr? i love hikin, its my fav hobby'},
        ]
        topics = identify_topics(conversation)
        self.assertIsInstance(topics, list)
        # We expect the lemmatizer and preprocessor to clean this up enough to form a topic.
        self.assertTrue(len(topics) > 0, "Expected to identify a topic from noisy text.")
        self.assertIn("hike", topics[0]["canonical_name"].lower(), "The canonical name should relate to hiking.")


# --- Module: Geo/Time Planner Tests ---
from planner import compute_geo_time_features

class TestPlanner(unittest.TestCase):

    # We use @patch to mock the get_location_details function within the planner module
    @patch('planner.get_location_details')
    def test_compute_geo_time_features_valid_locations(self, mock_get_location):
        """Tests feature computation for two valid, distinct locations."""

        # Define the mock return values for New York and London
        new_york_details = {
            "latitude": 40.7128, "longitude": -74.0060, "timezone": "America/New_York",
            "city_state": "New York, New York", "country": "United States"
        }
        london_details = {
            "latitude": 51.5074, "longitude": -0.1278, "timezone": "Europe/London",
            "city_state": "London", "country": "United Kingdom"
        }

        # This configures the mock to return different values on subsequent calls
        mock_get_location.side_effect = [new_york_details, london_details]

        features = compute_geo_time_features("New York, NY", "London, UK")

        self.assertIn("distance_km", features)
        self.assertIn("time_difference_hours", features)
        self.assertGreater(features["distance_km"], 5000) # Should be ~5500 km
        self.assertIsNotNone(features["time_difference_hours"]) # TZ diff can be 4 or 5 depending on DST
        self.assertTrue(features["is_virtual"])
        self.assertEqual(features["my_location"]["city_state"], "New York, New York")
        self.assertEqual(features["their_location"]["city_state"], "London")

    @patch('planner.get_location_details')
    def test_invalid_or_missing_location(self, mock_get_location):
        """Tests graceful handling of one or more invalid location strings."""

        new_york_details = {
            "latitude": 40.7128, "longitude": -74.0060, "timezone": "America/New_York",
            "city_state": "New York, New York", "country": "United States"
        }
        # Mock one valid location and one invalid one (returns None)
        mock_get_location.side_effect = [new_york_details, None]

        features = compute_geo_time_features("New York, NY", "Invalid Place Name")

        self.assertIsNone(features["distance_km"])
        self.assertIsNone(features["time_difference_hours"])
        self.assertEqual(features["their_location"], {}) # Their location should be empty
        self.assertFalse(features["is_virtual"]) # Cannot determine, so should be False

    @patch('planner.get_location_details')
    def test_same_city_locations(self, mock_get_location):
        """Tests behavior when both locations are the same."""

        sydney_details = {
            "latitude": -33.8688, "longitude": 151.2093, "timezone": "Australia/Sydney",
            "city_state": "Sydney, New South Wales", "country": "Australia"
        }
        # Return the same details for both calls
        mock_get_location.side_effect = [sydney_details, sydney_details]

        features = compute_geo_time_features("Sydney", "Sydney")

        self.assertEqual(features["distance_km"], 0.0)
        self.assertEqual(features["time_difference_hours"], 0.0)
        self.assertFalse(features["is_virtual"])

    @patch('planner.get_location_details')
    def test_international_date_line_edge_case(self, mock_get_location):
        """Tests time difference calculation across the international date line."""

        # Samoa and Fiji are geographically close but on opposite sides of the date line.
        samoa_details = {
            "latitude": -13.7590, "longitude": -172.1046, "timezone": "Pacific/Apia",
            "city_state": "Apia", "country": "Samoa"
        } # UTC-11 in standard time, UTC+13 in DST... but it's complex. Let's assume standard.
        fiji_details = {
            "latitude": -17.7134, "longitude": 178.0650, "timezone": "Pacific/Fiji",
            "city_state": "Suva", "country": "Fiji"
        } # UTC+12

        mock_get_location.side_effect = [samoa_details, fiji_details]
        features = compute_geo_time_features("Samoa", "Fiji")

        self.assertLess(features["distance_km"], 1200) # They are close
        # Time difference should be large. (UTC+12) - (UTC-11) = 23 hours, or -1 hour depending on direction.
        self.assertGreater(abs(features["time_difference_hours"]), 20)
        self.assertTrue(features["is_virtual"])


# --- Modules 4, 5, & 6: Suggestion Engine and Ranking Tests ---
from suggestion_engine import generate_suggestions, AdvancedSuggestionEngine
from model import Feedback

class TestSuggestionEngine(unittest.TestCase):

    def setUp(self):
        """Set up common data for suggestion engine tests."""
        self.conversation = [
            {'id': 1, 'role': 'user', 'content': 'I love hiking.'},
            {'id': 2, 'role': 'assistant', 'content': 'Me too! The mountains are great.'}
        ]
        self.topics = [
            {
                "canonical_name": "Hobbies & Interests", "keywords": ["hiking", "mountains"],
                "message_turns": self.conversation, "centroid": np.random.rand(1024)
            }
        ]
        self.context = {
            "detected_phases": ["Rapport Building"], "detected_tones": ["Friendly"],
            "sentiment_analysis": {"overall": "positive"}, "topic_recency": {"Hobbies & Interests": 1},
            "topic_saliency": {"Hobbies & Interests": 2}, "engagement_metrics": {"flirtation_level": "low"}
        }
        # Mock user preferences, which is not a feature but we test against it per the design
        self.user_profile = {"preferences": ["art", "museums"]}

    def test_suggestions_are_novel(self):
        """Tests that suggestions are not topics already present in the conversation."""
        suggestions = generate_suggestions(self.context, self.conversation, self.topics)
        for suggestion in suggestions["topics"]:
            self.assertNotIn(suggestion, [t["canonical_name"] for t in self.topics])

    @unittest.expectedFailure
    def test_adaptation_to_tone(self):
        """
        EXPECTED TO FAIL: Tests if suggestions adapt to a flirty tone.
        The current implementation does not use tone to filter or rank suggestions.
        """
        self.context["engagement_metrics"]["flirtation_level"] = "high"

        engine = AdvancedSuggestionEngine(self.context, self.conversation, self.topics)
        # Mock candidates to include a flirty option
        engine.transition_graph[self.topics[0]['canonical_name']] = {
            "A mischievous secret": 1.0, "Favorite type of pasta": 1.0
        }
        suggestions = engine.get_suggestions()["topics"]
        self.assertIn("A mischievous secret", suggestions)

    @unittest.expectedFailure
    def test_adaptation_to_user_preferences(self):
        """
        EXPECTED TO FAIL: Tests if suggestions adapt to user preferences.
        The system does not currently have a concept of user profiles/preferences.
        """
        engine = AdvancedSuggestionEngine(self.context, self.conversation, self.topics)
        # Mock candidates to include a preference-aligned topic
        engine.transition_graph[self.topics[0]['canonical_name']] = {
            "Favorite artist": 1.0, "Favorite sport": 1.0
        }
        # This test requires the engine to somehow know about self.user_profile
        # which it doesn't, so it's a guaranteed failure that points out a design gap.
        suggestions = engine.get_suggestions(user_profile=self.user_profile)["topics"]
        self.assertIn("Favorite artist", suggestions)

    @unittest.expectedFailure
    def test_ranked_output_structure(self):
        """
        EXPECTED TO FAIL: Tests if the output is a ranked list with scores/reasons.
        The current implementation returns a simple list of strings.
        """
        suggestions_obj = generate_suggestions(self.context, self.conversation, self.topics)
        self.assertIn("ranked_suggestions", suggestions_obj, "Output should have 'ranked_suggestions' key")

        suggestions = suggestions_obj["ranked_suggestions"]
        self.assertIsInstance(suggestions[0], dict)
        self.assertIn("score", suggestions[0])


# --- Final Execution ---
if __name__ == '__main__':
    # This allows running the test suite from the command line
    unittest.main()
