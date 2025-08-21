import sys
import os
import json
import asyncio

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from app.main import run_analysis_pipeline
from app.schemas import AnalyzePayload

async def run_test():
    """
    Runs an end-to-end test of the analysis pipeline.
    """
    # Sample payload based on README.md
    payload_data = {
      "matchId": "unique-match-id-123",
      "scraped_data": {
        "myName": "Jules",
        "theirName": "Alex",
        "theirProfile": "Loves dogs, hiking, and trying new food spots. Big fan of classic rock music.",
        "theirLocationString": "Los Angeles, USA",
        "conversationHistory": [
          {
            "role": "assistant",
            "content": "Hey! Your profile caught my eye. You have a great smile.",
            "date": "2024-08-18T10:00:00"
          },
          {
            "role": "user",
            "content": "Thanks! You too. I see you like dogs. I have a golden retriever named Max!",
            "date": "2024-08-18T10:05:00"
          },
          {
              "role": "assistant",
              "content": "Oh wow, I love golden retrievers. They are so playful. What's your favorite park to take him to?",
              "date": "2024-08-18T10:10:00"
          },
          {
              "role": "user",
              "content": "We usually go to the big one by the lake. It's great for hiking too. I'm always looking for new trails.",
              "date": "2024-08-18T10:15:00"
          }
        ]
      },
      "ui_settings": {
        "useEnhancedNlp": True,
        "myLocation": "San Francisco, USA",
        "myProfile": "I'm a software engineer who loves being outdoors. My hobbies include hiking, photography, and finding the best tacos in the city.",
        "local_model_name": None
      }
    }

    print("--- Running Pipeline Test ---")
    print("Input Payload:")
    print(json.dumps(payload_data, indent=2))

    # Create a Pydantic model instance
    try:
        payload = AnalyzePayload(**payload_data)
    except Exception as e:
        print(f"\n--- Pydantic Model Validation Error ---")
        print(e)
        return

    # Run the pipeline
    try:
        result = await run_analysis_pipeline(payload)
        print("\n--- Pipeline Execution Successful ---")
        print("Final JSON Output:")
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"\n--- Pipeline Execution Failed ---")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(run_test())
