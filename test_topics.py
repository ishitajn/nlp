# In test_topics.py
import asyncio
from analysis_engine import run_full_analysis

# Sample data mimicking the structure of the application's input
sample_my_profile = "I am a software engineer who loves hiking and trying new restaurants."
sample_their_profile = "I'm a doctor. I enjoy playing piano and traveling."
sample_conversation = [
    {"role": "user", "content": "Hey! Your profile is interesting. I see you're a doctor.", "date": "2023-01-01T12:00:00"},
    {"role": "assistant", "content": "Thanks! Yeah, I love my job. What do you do?", "date": "2023-01-01T12:01:00"},
    {"role": "user", "content": "I'm a software engineer. It's pretty different from being a doctor I imagine!", "date": "2023-01-01T12:02:00"},
    {"role": "assistant", "content": "Haha, for sure. So you like hiking? I've been wanting to go more.", "date": "2023-01-01T12:03:00"},
    {"role": "user", "content": "You should! We could go hiking sometime. I also love exploring new food spots.", "date": "2023-01-01T12:04:00"},
    {"role": "assistant", "content": "I'd like that. I'm a big foodie too. We should definitely talk about our favorite restaurants.", "date": "2023-01-01T12:05:00"},
]

async def main():
    print("Running test analysis...")
    analysis_result = run_full_analysis(
        my_profile=sample_my_profile,
        their_profile=sample_their_profile,
        turns=sample_conversation
    )

    print("\n--- Analysis Complete ---")

    # Print the discovered topics
    if "conversation_state" in analysis_result and "discovered_topics" in analysis_result["conversation_state"]:
        discovered_topics = analysis_result["conversation_state"]["discovered_topics"]
        print("\nDiscovered Topics:")
        if discovered_topics:
            for topic, phrases in discovered_topics.items():
                print(f"  - {topic}: {phrases}")
        else:
            print("  No topics discovered.")
    else:
        print("Could not find discovered_topics in the analysis result.")

    # Print canonical topics for comparison
    if "conversation_state" in analysis_result and "topic_mapping" in analysis_result["conversation_state"]:
        canonical_topics = analysis_result["conversation_state"]["topic_mapping"]
        print("\nCanonical Topics (for comparison):")
        if canonical_topics:
            for topic, phrases in canonical_topics.items():
                print(f"  - {topic}: {phrases}")
        else:
            print("  No canonical topics found.")


if __name__ == "__main__":
    # Since run_full_analysis is synchronous, we don't need asyncio.run
    # However, if any part of the chain becomes async, this will be needed.
    # For now, calling main directly is fine.
    # To be safe and future-proof, let's use asyncio.
    asyncio.run(main())
