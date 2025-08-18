import unittest
from fastapi.testclient import TestClient
from ..api import app

class TestApi(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
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
            "sentiment", "topics", "conversation_dynamics", "geoContext",
            "response_analysis", "suggested_topics", "recommended_actions"
        ]

    def test_analyze_endpoint_fast(self):
        """Test the /analyze endpoint with the fast pipeline."""
        payload = self.sample_payload
        payload["ui_settings"]["useEnhancedNlp"] = False
        response = self.client.post("/analyze", json=payload)
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertIsInstance(result, dict)
        for key in self.expected_keys:
            self.assertIn(key, result)

    def test_analyze_endpoint_enhanced(self):
        """Test the /analyze endpoint with the enhanced pipeline."""
        payload = self.sample_payload
        payload["ui_settings"]["useEnhancedNlp"] = True
        response = self.client.post("/analyze", json=payload)
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertIsInstance(result, dict)
        for key in self.expected_keys:
            self.assertIn(key, result)

if __name__ == '__main__':
    unittest.main()
