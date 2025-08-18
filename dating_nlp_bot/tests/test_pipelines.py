import unittest
import json
from ..main import process_payload

class TestPipelines(unittest.TestCase):
    def setUp(self):
        self.sample_payload = {
            "matchId": "12345",
            "scraped_data": {
                "myName": "Jules",
                "theirName": "Alex",
                "theirProfile": "Loves hiking and dogs. Looking for a connection.",
                "theirLocationString": "San Francisco, USA",
                "conversationHistory": [
                    {"role": "assistant", "content": "Hey, how's it going?", "date": "2024-01-01T10:00:00"},
                    {"role": "user", "content": "Hi! I'm doing great, thanks for asking. I saw you like hiking, me too!", "date": "2024-01-01T10:05:00"},
                    {"role": "assistant", "content": "Oh nice! We should go sometime. What's your favorite trail?", "date": "2024-01-01T10:10:00"}
                ]
            },
            "ui_settings": {
                "useEnhancedNlp": False,
                "myLocation": "New York, USA",
                "myProfile": "Software engineer, loves coffee and coding.",
                "local_model_name": None
            }
        }
        self.expected_keys = [
            "sentiment", "topics", "suggested_topics", "conversation_dynamics",
            "geoContext", "response_analysis", "recommended_actions", "conversation_brain",
            "llm_prompt_context"
        ]

    def test_fast_pipeline(self):
        """Test the fast pipeline runs and returns the correct structure."""
        payload = self.sample_payload
        payload["ui_settings"]["useEnhancedNlp"] = False
        result = process_payload(payload)
        self.assertIsInstance(result, dict)
        for key in self.expected_keys:
            self.assertIn(key, result)

        # Check for some nested keys
        self.assertIn("overall", result["sentiment"])
        self.assertIn("map", result["topics"])
        self.assertIn("predictive_actions", result["conversation_brain"])

    def test_enhanced_pipeline(self):
        """Test the enhanced pipeline runs and returns the correct structure."""
        payload = self.sample_payload
        payload["ui_settings"]["useEnhancedNlp"] = True
        result = process_payload(payload)
        self.assertIsInstance(result, dict)
        for key in self.expected_keys:
            self.assertIn(key, result)

        # Check for some nested keys
        self.assertIn("overall", result["sentiment"])
        self.assertIn("map", result["topics"])
        self.assertIn("predictive_actions", result["conversation_brain"])


if __name__ == '__main__':
    unittest.main()
