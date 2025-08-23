# In test_engine_v2.py
import json
from topic_engine_v2 import run_topic_engine

# A sample conversation with varied topics
sample_conversation = [
    {"role": "user", "content": "Hey! Your profile is interesting. I see you're a doctor and you like to travel.", "date": "2023-01-01T12:00:00"},
    {"role": "assistant", "content": "Thanks! Yeah, I love my job, but traveling is my real passion. Just got back from a hiking trip in Peru.", "date": "2023-01-01T12:01:00"},
    {"role": "user", "content": "Wow, Peru! That's amazing. I've always wanted to go. I'm more of a city-explorer myself, I love finding new restaurants and art galleries.", "date": "2023-01-01T12:02:00"},
    {"role": "assistant", "content": "That's cool too! Any favorite food spots I should know about? Maybe we could check one out sometime.", "date": "2023-01-01T12:03:00"},
    {"role": "user", "content": "Definitely! There's this great Italian place downtown. Are you free next week? We could grab dinner.", "date": "2023-01-01T12:04:00"},
    {"role": "assistant", "content": "I'd love that. I'm free on Thursday evening. Does that work for you? Your smile is really cute btw.", "date": "2023-01-01T12:05:00"},
]

def test_new_engine():
    """
    Tests the full pipeline of the new topic engine and prints the results.
    """
    print("--- Running New Topic Engine Test ---")

    # Run the analysis
    analysis_result = run_topic_engine(sample_conversation)

    # Pretty-print the output
    print("\n--- Analysis Result ---")
    print(json.dumps(analysis_result, indent=2))

    # Some basic checks
    if "identified_topics" in analysis_result:
        print("\n--- Basic Checks ---")
        topics = analysis_result["identified_topics"]
        print(f"Successfully identified {len(topics)} topics.")
        if topics:
            first_topic = topics[0]
            assert "topic_id" in first_topic
            assert "keywords" in first_topic
            assert "messages" in first_topic
            assert "category" in first_topic
            print("First topic has the correct structure.")
            print(f"First topic category: {first_topic['category']}")
            print(f"First topic keywords: {first_topic['keywords']}")
    else:
        print("Error: 'identified_topics' key not found in result.")
        if "error" in analysis_result:
            print(f"Engine returned error: {analysis_result['error']}")

if __name__ == "__main__":
    test_new_engine()
