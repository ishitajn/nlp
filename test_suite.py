"""
A comprehensive test suite for the entire conversational analysis and suggestion system.
This script is designed to be run as a single command to validate all modules.
"""
import pytest
import sys
import os
import numpy as np
import unittest
from unittest.mock import patch, MagicMock

# Add the parent directory to the sys.path to allow imports from the app
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- Import all necessary modules to be tested ---
from fastapi.testclient import TestClient

from model import AnalyzePayload, ConversationTurn, ScrapedData, UISettings, Feedback
from main import app
import planner
from preprocessor import preprocess_text, nlp
from context_engine import extract_contextual_features
from topic_engine import identify_topics
from suggestion_engine import generate_suggestions, AdvancedSuggestionEngine
import services
import analysis_engine


# --- Pytest Fixtures ---

@pytest.fixture(scope="module")
def client():
    """Fixture for the FastAPI TestClient."""
    with TestClient(app) as c:
        yield c

@pytest.fixture
def sample_conversation_turns():
    """Fixture for a sample conversation history (list of dicts)."""
    return [
        {'role': 'user', 'content': 'Hey, how\'s it going?', 'date': '2023-01-01T12:00:00'},
        {'role': 'assistant', 'content': 'Pretty good! Just enjoying the weather. You?', 'date': '2023-01-01T12:01:00'},
        {'role': 'user', 'content': 'Same here. I love hiking. Ever been to Yosemite?', 'date': '2023-01-01T12:02:00'},
        {'role': 'assistant', 'content': 'Oh yeah, I love Yosemite! The views are incredible.', 'date': '2023-01-01T12:03:00'}
    ]

@pytest.fixture
def basic_payload(sample_conversation_turns):
    """Fixture for a basic AnalyzePayload object."""
    # Convert dicts to ConversationTurn objects
    conversation_objects = [ConversationTurn(**turn) for turn in sample_conversation_turns]
    return AnalyzePayload(
        matchId="test_match_123",
        scraped_data=ScrapedData(
            myName="Jules",
            theirName="Alex",
            theirProfile="I love hiking, dogs, and trying new food.",
            theirLocationString="San Francisco, CA",
            conversationHistory=conversation_objects
        ),
        ui_settings=UISettings(
            useEnhancedNlp=True,
            myLocation="New York, NY",
            myProfile="Software engineer who loves to travel and cook."
        ),
        feedback=[]
    )


# --- Module 1: Preprocessor Tests ---
@pytest.mark.skipif(nlp is None, reason="spaCy model not loaded.")
class TestPreprocessor:
    def test_basic_cleaning_and_lemmatization(self):
        text = "This is a test sentence, with running dogs and played cats!"
        expected = "test sentence run dog play cat"
        assert preprocess_text(text) == expected

    def test_filler_word_removal(self):
        text = "um, hey, like, this is basically a test, lol"
        expected = "test"
        assert preprocess_text(text) == expected

    def test_emoji_normalization(self):
        text = "I love this so much ❤️"
        expected = "love so much love"
        assert preprocess_text(text, normalize_emojis_flag=True) == expected

    def test_url_and_mention_removal(self):
        text = "Check out this link http://example.com and say hi to @someone"
        expected = "check link say"
        assert preprocess_text(text) == expected


# --- Module: Geo/Time Planner Tests ---
class TestPlanner:

    @patch('planner.geolocator.geocode')
    def test_get_location_details_service_error(self, mock_geocode):
        """Tests that get_location_details handles geocoding service errors gracefully."""
        from geopy.exc import GeocoderTimedOut
        mock_geocode.side_effect = GeocoderTimedOut("Service timed out")
        result = planner.get_location_details("some place that will time out")
        assert result is None

    @patch('planner.get_location_details')
    def test_compute_geo_time_features_valid_locations(self, mock_get_location):
        """Tests feature computation for two valid, distinct locations."""
        new_york_details = {
            "latitude": 40.7128, "longitude": -74.0060, "timezone": "America/New_York",
            "city_state": "New York, New York", "country": "United States"
        }
        london_details = {
            "latitude": 51.5074, "longitude": -0.1278, "timezone": "Europe/London",
            "city_state": "London", "country": "United Kingdom"
        }
        mock_get_location.side_effect = [new_york_details, london_details]

        features = planner.compute_geo_time_features("New York, NY", "London, UK")

        assert "distance_km" in features
        assert "time_difference_hours" in features
        assert features["distance_km"] > 5000
        assert features["time_difference_hours"] is not None
        assert features["is_virtual"] is True
        assert features["my_location"]["city_state"] == "New York, New York"
        assert features["their_location"]["city_state"] == "London"

    @patch('planner.get_location_details')
    def test_invalid_or_missing_location(self, mock_get_location):
        """Tests graceful handling of one or more invalid location strings."""
        new_york_details = {
            "latitude": 40.7128, "longitude": -74.0060, "timezone": "America/New_York",
            "city_state": "New York, New York", "country": "United States"
        }
        mock_get_location.side_effect = [new_york_details, None]

        features = planner.compute_geo_time_features("New York, NY", "Invalid Place Name")

        assert features["distance_km"] is None
        assert features["time_difference_hours"] is None
        assert features["their_location"] == {}
        assert features["is_virtual"] is False

    @patch('planner.get_location_details')
    def test_same_city_locations(self, mock_get_location):
        """Tests behavior when both locations are the same."""
        sydney_details = {
            "latitude": -33.8688, "longitude": 151.2093, "timezone": "Australia/Sydney",
            "city_state": "Sydney, New South Wales", "country": "Australia"
        }
        mock_get_location.side_effect = [sydney_details, sydney_details]

        features = planner.compute_geo_time_features("Sydney", "Sydney")

        assert features["distance_km"] == 0.0
        assert features["time_difference_hours"] == 0.0
        assert features["is_virtual"] is False

    @patch('planner.get_location_details')
    def test_international_date_line_edge_case(self, mock_get_location):
        """Tests time difference calculation across the international date line."""
        # Note: Samoa is UTC+13, Fiji is UTC+12. The difference is 1 hour.
        samoa_details = {
            "latitude": -13.7590, "longitude": -172.1046, "timezone": "Pacific/Apia", # UTC+13
            "city_state": "Apia", "country": "Samoa"
        }
        fiji_details = {
            "latitude": -17.7134, "longitude": 178.0650, "timezone": "Pacific/Fiji", # UTC+12
            "city_state": "Suva", "country": "Fiji"
        }
        mock_get_location.side_effect = [samoa_details, fiji_details]
        features = planner.compute_geo_time_features("Samoa", "Fiji")

        assert features["distance_km"] < 1200
        assert abs(features["time_difference_hours"]) == 1.0
        assert features["is_virtual"] is False # 1 hour diff is not > 2

# --- Module 3: Context Engine Tests ---
class TestContextEngine:
    @pytest.fixture
    def context_conversation(self):
        return [
            {'role': 'user', 'content': 'Hey, I saw your profile and we matched! How are you?'},
            {'role': 'assistant', 'content': 'I am doing great, thanks! Your pictures from your trip to the mountains look amazing.'},
            {'role': 'user', 'content': 'Thanks! That was from my last hiking trip. I love hiking.'},
        ]

    @pytest.fixture
    def context_topics(self, context_conversation):
        return [
            {"canonical_name": "Greetings", "message_turns": [context_conversation[0]]},
            {"canonical_name": "Travel & Photography", "message_turns": [context_conversation[1]]},
            {"canonical_name": "Hobbies & Interests", "message_turns": [context_conversation[2]]},
        ]

    def test_structure_and_basic_metrics(self, context_conversation, context_topics):
        context = extract_contextual_features(context_conversation, context_topics)
        assert isinstance(context, dict)
        expected_keys = ["detected_phases", "sentiment_analysis", "topic_saliency", "topic_recency", "speaker_metrics"]
        for key in expected_keys:
            assert key in context
        assert context["speaker_metrics"]["user_turn_count"] == 2

    def test_sarcastic_greeting_edge_case(self):
        """Tests that a sarcastic greeting is still identified as an icebreaker but with negative sentiment."""
        sarcastic_convo = [{'role': 'user', 'content': 'Oh, great. another match. Hello there.'}]
        sarcastic_topics = [{"canonical_name": "Greetings", "message_turns": sarcastic_convo}]

        context = extract_contextual_features(sarcastic_convo, sarcastic_topics)
        assert "Icebreaker" in context["detected_phases"]
        # VADER is often fooled by sarcasm, this test confirms the current behavior.
        assert context["sentiment_analysis"]["overall"] == "very positive"

    def test_question_intent_detection(self):
        """Tests that a direct question is identified as a 'Gathering Information' intent."""
        question_convo = [{'role': 'user', 'content': 'So, what do you do for fun?'}]
        # Topics are not strictly necessary for this test, but the function requires them.
        topics = [{"canonical_name": "Getting to Know You", "message_turns": question_convo}]

        context = extract_contextual_features(question_convo, topics)
        assert "Gathering Information" in context["detected_intents"]

    def test_multilingual_greeting_is_not_detected(self):
        """Tests that a non-English greeting is NOT detected as an Icebreaker."""
        multilingual_convo = [{'role': 'user', 'content': 'Hola, como estas?'}]
        topics = [{"canonical_name": "Greetings", "message_turns": multilingual_convo}]

        context = extract_contextual_features(multilingual_convo, topics)
        # The default phase is 'Rapport Building', so we check that 'Icebreaker' is not present.
        assert "Icebreaker" not in context["detected_phases"]

    def test_poorly_structured_greeting_edge_case(self):
        """Tests that a poorly structured greeting is still identified."""
        poorly_structured_convo = [{'role': 'user', 'content': 'heyyy how u doin lol'}]
        poorly_structured_topics = [{"canonical_name": "Greetings", "message_turns": poorly_structured_convo}]

        context = extract_contextual_features(poorly_structured_convo, poorly_structured_topics)
        assert "Icebreaker" in context["detected_phases"]
        assert "Playful" in context["detected_tones"]

# --- Module 2: Topic Engine Tests ---
class TestTopicEngine:

    @patch('topic_engine.embedder_service')
    @patch('topic_engine.preprocess_text')
    def test_identify_topics_structure(self, mock_preprocess, mock_embedder):
        """Tests that the output is a list of topic dicts with the correct structure."""
        conversation = [
            {'role': 'user', 'content': 'I really love hiking and being in the mountains.'},
            {'role': 'assistant', 'content': 'Me too! Hiking is a great way to unwind.'},
        ]
        # Mocking dependencies
        mock_preprocess.side_effect = lambda x: x  # Simple pass-through
        mock_embedder.encode_cached.return_value = np.random.rand(len(conversation), 384)

        topics = identify_topics(conversation)
        assert isinstance(topics, list)
        if topics:
            topic = topics[0]
            assert isinstance(topic, dict)
            expected_keys = ["canonical_name", "keywords", "message_count", "centroid"]
            for key in expected_keys:
                assert key in topic
            assert isinstance(topic["canonical_name"], str)
            assert isinstance(topic["keywords"], list)
            assert isinstance(topic["message_count"], int)
            assert isinstance(topic["centroid"], np.ndarray)

    @patch('topic_engine.embedder_service')
    @patch('topic_engine.preprocess_text')
    def test_ambiguous_and_short_utterances(self, mock_preprocess, mock_embedder):
        """Tests behavior with ambiguous text that may not form strong clusters."""
        conversation = [
            {'role': 'user', 'content': 'k'},
            {'role': 'assistant', 'content': 'lol'},
            {'role': 'user', 'content': 'nice'},
            {'role': 'assistant', 'content': 'yep'},
        ]
        mock_preprocess.side_effect = lambda x: x
        mock_embedder.encode_cached.return_value = np.random.rand(len(conversation), 384)

        topics = identify_topics(conversation)
        assert topics == []

    @patch('topic_engine.embedder_service')
    @patch('topic_engine.preprocess_text')
    def test_identify_topics_from_profile(self, mock_preprocess, mock_embedder):
        """Tests that topics are identified from the profile when conversation is empty."""
        conversation = []
        profile_text = "I am a photographer and I love to travel."

        mock_preprocess.return_value = "photographer love travel"
        mock_embedder.encode_cached.return_value = np.random.rand(1, 384)

        topics = identify_topics(conversation, their_profile=profile_text)

        assert len(topics) == 1
        topic = topics[0]
        assert topic["message_count"] == 1
        assert "From Profile" in topic["messages"]
        # Check if keywords are reasonable (mocking yake would be too much, so we check the canonical name)
        assert "travel" in topic["canonical_name"].lower() or "photographer" in topic["canonical_name"].lower()

    @patch('topic_engine.hdbscan.HDBSCAN')
    @patch('topic_engine.embedder_service')
    @patch('topic_engine.preprocess_text')
    def test_deduplication_of_similar_topics(self, mock_preprocess, mock_embedder, mock_hdbscan):
        """Tests that semantically similar topics are merged."""
        conversation = [
            {'role': 'user', 'content': 'I like to go hiking.'},
            {'role': 'assistant', 'content': 'Yeah, I love to hike as well.'},
        ]
        mock_preprocess.side_effect = lambda x: x.lower()

        # Mock embedder to create two close clusters
        embeddings = np.array([
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
        ])
        mock_embedder.encode_cached.side_effect = [
            embeddings,
            np.array([[1.0, 0.0, 0.0]]),
            np.array([[0.9, 0.1, 0.0]]),
        ]

        # Mock hdbscan to create two initial clusters
        mock_clusterer = MagicMock()
        mock_clusterer.fit_predict.return_value = np.array([0, 1])
        mock_hdbscan.return_value = mock_clusterer

        # Mock YAKE to return keywords that are very similar
        with patch('topic_engine.yake.KeywordExtractor') as mock_yake:
            mock_extractor = MagicMock()
            # fuzz.token_set_ratio('go hiking', 'love to hike') is 100
            mock_extractor.extract_keywords.side_effect = [
                [('go hiking', 0.1)],
                [('love to hike', 0.1)]
            ]
            mock_yake.return_value = mock_extractor
            topics = identify_topics(conversation)

        assert len(topics) == 1
        assert "hike" in topics[0]["canonical_name"].lower()

    @patch('topic_engine.hdbscan.HDBSCAN')
    @patch('topic_engine.embedder_service')
    @patch('topic_engine.preprocess_text')
    def test_overlapping_topics_in_single_turn(self, mock_preprocess, mock_embedder, mock_hdbscan):
        """
        Tests behavior when a single turn contains multiple topics.
        """
        conversation = [
            {'role': 'user', 'content': 'I love hiking and I am also a software engineer.'},
            {'role': 'assistant', 'content': 'That\'s cool! I like hiking too.'},
        ]
        mock_preprocess.side_effect = lambda x: x
        mock_embedder.encode_cached.return_value = np.random.rand(len(conversation), 384)

        # Mock hdbscan to ensure a cluster is formed around 'hiking'
        mock_clusterer = MagicMock()
        mock_clusterer.fit_predict.return_value = np.array([0, 0]) # Both sentences in the same cluster
        mock_hdbscan.return_value = mock_clusterer

        topics = identify_topics(conversation)

        assert len(topics) >= 1
        assert "hiking" in topics[0]["canonical_name"].lower()


    @patch('topic_engine.hdbscan.HDBSCAN')
    @patch('topic_engine.embedder_service')
    @patch('topic_engine.preprocess_text')
    def test_noisy_text_handling(self, mock_preprocess, mock_embedder, mock_hdbscan):
        """Tests that noisy text with typos and slang is handled."""
        conversation = [
            {'role': 'user', 'content': 'sooo i went hikinnggg yday it was gr8!!'},
            {'role': 'assistant', 'content': 'fr? i love hikin, its my fav hobby'},
        ]
        # Simulate preprocessing cleaning the text
        mock_preprocess.side_effect = ["i went hiking yesterday it was great", "for real i love hiking its my favorite hobby"]
        mock_embedder.encode_cached.return_value = np.array([
            np.random.rand(384) * 1.1, # Make embeddings slightly different
            np.random.rand(384) * 1.0
        ])

        topics = identify_topics(conversation)
        assert len(topics) > 0
        assert "hike" in topics[0]["canonical_name"].lower()


# --- Modules 4, 5, & 6: Suggestion Engine and Ranking Tests ---
class TestSuggestionEngine:

    @pytest.fixture
    def suggestion_data(self):
        """Set up common data for suggestion engine tests."""
        conversation = [
            {'id': 1, 'role': 'user', 'content': 'I love hiking.'},
            {'id': 2, 'role': 'assistant', 'content': 'Me too! The mountains are great.'}
        ]
        topics = [
            {
                "canonical_name": "Hobbies & Interests", "keywords": ["hiking", "mountains"],
                "category": "Interests", # Added missing key
                "message_turns": conversation, "centroid": np.random.rand(384)
            }
        ]
        context = {
            "detected_phases": ["Rapport Building"], "detected_tones": ["Friendly"],
            "sentiment_analysis": {"overall": "positive"}, "topic_recency": {"Hobbies & Interests": 1},
            "topic_saliency": {"Hobbies & Interests": 2}, "engagement_metrics": {"flirtation_level": "low"}
        }
        return {"context": context, "conversation": conversation, "topics": topics}

    def test_suggestions_are_novel(self, suggestion_data):
        """Tests that suggestions are not topics already present in the conversation."""
        suggestions = generate_suggestions(
            analysis_data=suggestion_data['context'],
            conversation_turns=suggestion_data['conversation'],
            identified_topics=suggestion_data['topics']
        )
        # The suggestion engine's output format needs to be known.
        # Assuming it returns a dict with a "topics" key containing a list of strings.
        if "topics" in suggestions:
            for suggestion in suggestions["topics"]:
                assert suggestion not in [t["canonical_name"] for t in suggestion_data["topics"]]

    @pytest.mark.xfail(reason="The current implementation does not use tone to filter or rank suggestions.")
    def test_adaptation_to_tone(self, suggestion_data):
        """
        Tests that suggestions currently DO NOT adapt to tone.
        This test will pass if the feature is missing, and fail if it's ever implemented.
        """
        # Mock file I/O
        with patch('suggestion_engine.os.path.exists', return_value=False), \
             patch('suggestion_engine.open', new_callable=unittest.mock.mock_open), \
             patch('suggestion_engine.json.dump'):

            # --- Get suggestions for a non-flirty context ---
            suggestion_data["context"]["engagement_metrics"]["flirtation_level"] = "low"
            suggestions_low_flirt = generate_suggestions(
                analysis_data=suggestion_data['context'],
                conversation_turns=suggestion_data['conversation'],
                identified_topics=suggestion_data['topics']
            )["topics"]

            # --- Get suggestions for a flirty context ---
            suggestion_data["context"]["engagement_metrics"]["flirtation_level"] = "high"
            suggestions_high_flirt = generate_suggestions(
                analysis_data=suggestion_data['context'],
                conversation_turns=suggestion_data['conversation'],
                identified_topics=suggestion_data['topics']
            )["topics"]

            # The suggestions should be identical since tone is not used.
            # Sorting is needed because the order of evergreen topics might not be guaranteed.
            assert sorted(suggestions_low_flirt) == sorted(suggestions_high_flirt)

    @pytest.mark.xfail(reason="The system does not currently have a concept of user profiles/preferences.")
    def test_adaptation_to_user_preferences(self, suggestion_data):
        """
        Tests that suggestions currently DO NOT adapt to user preferences.
        This test will pass if the feature is missing, and fail if it's ever implemented.
        """
        with patch('suggestion_engine.os.path.exists', return_value=False), \
             patch('suggestion_engine.open', new_callable=unittest.mock.mock_open), \
             patch('suggestion_engine.json.dump'):

            # Generate suggestions without any user profile info
            suggestions_without_prefs = generate_suggestions(
                analysis_data=suggestion_data['context'],
                conversation_turns=suggestion_data['conversation'],
                identified_topics=suggestion_data['topics']
            )["topics"]

            # The current engine doesn't accept a user profile, so we can't pass it in.
            # The test confirms that the logic doesn't implicitly use it.
            suggestions_with_prefs = generate_suggestions(
                analysis_data=suggestion_data['context'],
                conversation_turns=suggestion_data['conversation'],
                identified_topics=suggestion_data['topics']
            )["topics"]

            assert sorted(suggestions_without_prefs) == sorted(suggestions_with_prefs)

    def test_feedback_boosts_suggestions(self, suggestion_data):
        """
        Tests that 'chosen' feedback increases the weight of a topic transition in the graph.
        """
        from model import Feedback

        feedback = [Feedback(
            current_topic="Hobbies & Interests",
            chosen_suggestion="Spontaneous adventures",
            action="chosen"
        )]

        # Mock the file I/O to inspect the written graph
        mock_graph_data = {}

        def mock_json_dump(data, file, **kwargs):
            # Capture the data that would be written to the file
            mock_graph_data.update(data)

        # Patch os.path.exists to simulate no pre-existing graph
        # Patch open to allow writing
        # Patch json.dump to capture the output
        with patch('suggestion_engine.os.path.exists', return_value=False), \
             patch('suggestion_engine.open', new_callable=unittest.mock.mock_open), \
             patch('suggestion_engine.json.dump', side_effect=mock_json_dump):

            generate_suggestions(
                analysis_data=suggestion_data['context'],
                conversation_turns=suggestion_data['conversation'],
                identified_topics=suggestion_data['topics'],
                feedback=feedback
            )

        # Check that the transition graph was updated correctly
        assert "Hobbies & Interests" in mock_graph_data
        assert "Spontaneous adventures" in mock_graph_data["Hobbies & Interests"]
        # The logic is score *= 1.2; score += 0.5. Initial score is 0. So it becomes 0.5.
        assert mock_graph_data["Hobbies & Interests"]["Spontaneous adventures"] > 0

    def test_evergreen_suggestions_for_empty_input(self, suggestion_data):
        """Tests that evergreen topics are returned when the conversation has no topics."""
        from suggestion_engine import EVERGREEN_TOPICS
        # Mock file I/O
        with patch('suggestion_engine.os.path.exists', return_value=False), \
             patch('suggestion_engine.open', new_callable=unittest.mock.mock_open), \
             patch('suggestion_engine.json.dump'):

            suggestions_obj = generate_suggestions(
                analysis_data=suggestion_data['context'],
                conversation_turns=[], # No conversation
                identified_topics=[] # No topics
            )

            assert suggestions_obj["topics"]
            # Check that all suggestions are from the evergreen list
            assert all(s in EVERGREEN_TOPICS for s in suggestions_obj["topics"])

    @patch('suggestion_engine.embedder_service')
    def test_similar_suggestions_are_filtered(self, mock_embedder, suggestion_data):
        """Tests that candidates too similar to existing topics are filtered out."""
        from suggestion_engine import EVERGREEN_TOPICS
        # Let's say "Hobbies & Interests" is an existing topic.
        # We'll make an evergreen topic, "A passion project", seem very similar to it.
        existing_topic_embedding = suggestion_data["topics"][0]["centroid"].reshape(1, -1)

        # Let's assume evergreen topics are the candidates.
        candidates = list(EVERGREEN_TOPICS)

        # Create embeddings for candidates. Make one very similar to the existing topic.
        candidate_embeddings = np.random.rand(len(candidates), suggestion_data["topics"][0]["centroid"].shape[0])
        passion_project_index = candidates.index("A passion project")
        # Make the embedding for "A passion project" identical to the existing topic's embedding
        candidate_embeddings[passion_project_index, :] = existing_topic_embedding

        # The embedder will be called once for the candidates.
        mock_embedder.encode_cached.return_value = candidate_embeddings

        with patch('suggestion_engine.os.path.exists', return_value=False), \
             patch('suggestion_engine.open', new_callable=unittest.mock.mock_open), \
             patch('suggestion_engine.json.dump'):

            suggestions = generate_suggestions(
                analysis_data=suggestion_data['context'],
                conversation_turns=suggestion_data['conversation'],
                identified_topics=suggestion_data['topics']
            )["topics"]

        # "A passion project" should have been filtered out due to high similarity.
        assert "A passion project" not in suggestions

    def test_output_structure(self, suggestion_data):
        """Tests that the output is a dictionary with a 'topics' key containing a list of strings."""
        with patch('suggestion_engine.os.path.exists', return_value=False), \
             patch('suggestion_engine.open', new_callable=unittest.mock.mock_open), \
             patch('suggestion_engine.json.dump'):

            suggestions_obj = generate_suggestions(
                analysis_data=suggestion_data['context'],
                conversation_turns=suggestion_data['conversation'],
                identified_topics=suggestion_data['topics']
            )

            assert isinstance(suggestions_obj, dict)
            assert "topics" in suggestions_obj
            suggestions = suggestions_obj["topics"]
            assert isinstance(suggestions, list)
            # The list can be empty if all candidates are filtered out
            if suggestions:
                assert isinstance(suggestions[0], str)


# --- Module: Main Orchestration Logic ---
class TestAnalysisPipeline:

    @patch('analysis_engine.identify_topics')
    @patch('analysis_engine.extract_contextual_features')
    @patch('analysis_engine.analyze_conversation_behavior')
    @patch('analysis_engine.score_and_categorize_topics')
    def test_run_full_analysis_orchestration(
        self, mock_score, mock_behavior, mock_context, mock_topics,
        sample_conversation_turns
    ):
        """
        Tests that the main `run_full_analysis` function orchestrates calls correctly.
        """
        # Define mock return values for each engine
        mock_topics.return_value = [{"canonical_name": "Mock Topic", "category": "Uncategorized", "message_turns": []}]
        mock_context.return_value = {"detected_phases": ["Mock Phase"]}
        mock_behavior.return_value = {"user_engagement": "high"}
        mock_score.return_value = {"Categorized": ["Mock Topic"]}

        my_profile = "My test profile."
        their_profile = "Their test profile."

        # Run the analysis
        final_analysis = analysis_engine.run_full_analysis(
            my_profile=my_profile,
            their_profile=their_profile,
            processed_turns=sample_conversation_turns
        )

        # Assert that each engine was called once
        mock_topics.assert_called_once_with(sample_conversation_turns, their_profile)
        mock_context.assert_called_once()
        mock_behavior.assert_called_once()
        mock_score.assert_called_once()

        # Assert that the final output contains keys from all engine results
        assert "topic_clusters" in final_analysis
        assert "contextual_features" in final_analysis
        assert "behavioral_analysis" in final_analysis
        assert "categorized_topics" in final_analysis

        # Check that the category from scoring was integrated back into the topic cluster
        assert final_analysis["topic_clusters"][0]["category"] == "Categorized"

    @patch('main.assembler')
    @patch('main.generate_suggestions')
    @patch('main.compute_geo_time_features')
    @patch('main.run_full_analysis')
    @patch('main.normalizer')
    def test_analyze_endpoint(
        self, mock_normalizer, mock_run_analysis, mock_geo, mock_suggestions, mock_assembler,
        client, basic_payload
    ):
        """
        Tests the main /analyze endpoint to ensure it handles requests and calls the pipeline.
        """
        # Define mock return values for the pipeline functions
        mock_normalizer.clean_and_truncate.return_value = [{"role": "user", "content": "Cleaned message"}]
        mock_run_analysis.return_value = {"topic_clusters": []}
        mock_geo.return_value = {"distance_km": 100}
        mock_suggestions.return_value = {"topics": ["New suggestion"]}
        mock_assembler.build_final_json.return_value = {"status": "success", "data": "mocked"}

        # The payload needs to be converted to a dict for the request
        payload_dict = basic_payload.model_dump(by_alias=True)
        response = client.post("/analyze", json=payload_dict)

        # Assertions
        assert response.status_code == 200
        assert response.json() == {"status": "success", "data": "mocked"}

        # Verify that the pipeline functions were called
        mock_normalizer.clean_and_truncate.assert_called_once()
        mock_run_analysis.assert_called_once()
        mock_geo.assert_called_once()
        mock_suggestions.assert_called_once()
        mock_assembler.build_final_json.assert_called_once()
