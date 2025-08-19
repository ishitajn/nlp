# Dating Conversation Analyzer

This project provides a comprehensive NLP-driven analysis of dating conversations. It's built as a Python-based API using FastAPI and features two distinct modes for analysis: a lightweight `fast` mode and a powerful `enhanced` mode that leverages transformer models, including a light LLM for generating conversational insights.

## Features

- **Dual Analysis Modes**: Choose between a fast, rule-based analysis or a slower, more accurate analysis using transformer models.
- **Comprehensive Analysis**: The API returns a rich JSON object containing analysis on:
    - **Sentiment**: The overall mood of the conversation.
    - **Topics**: Liked, disliked, and sensitive topics, with a detailed map of discussed keywords.
    - **Conversation Dynamics**: Pace, stage, reciprocity, and flirtation level.
    - **Geo & Time Context**: Location-based analysis, including distance and time of day.
    - **Response Analysis**: Insights into the last messages exchanged.
    - **Recommended Actions**: UI-focused suggestions for the user's next move.
    - **Conversation Brain**: LLM-powered suggestions for questions, goals, and topic switches.
- **API Interface**: The entire project is exposed via a FastAPI endpoint, with automatic interactive documentation.

## Setup

1.  **Clone the repository.**

2.  **Create and activate a virtual environment** (recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: This will download several machine learning models, which may take some time and disk space.*

## How to Run

1.  **Start the API server** using Uvicorn:
    ```bash
    uvicorn dating_nlp_bot.main:app --reload
    ```
    The `--reload` flag enables hot-reloading, which is useful for development.

2.  **Access the API documentation**:
    Once the server is running, navigate to `http://127.0.0.1:8000/docs` in your web browser. This will open the interactive Swagger UI documentation.

3.  **Use the `/analyze` endpoint**:
    From the documentation, you can try out the `/analyze` endpoint directly. Click on it, then "Try it out", and paste your JSON payload into the request body.

## API Payload

The API expects a `POST` request to `/analyze` with a JSON body in the following format. To switch between the fast and enhanced modes, set the `useEnhancedNlp` flag to `false` or `true` respectively.

```json
{
  "matchId": "unique-match-id",
  "scraped_data": {
    "myName": "User",
    "theirName": "Match",
    "theirProfile": "Loves dogs and long walks on the beach.",
    "theirLocationString": "Los Angeles, USA",
    "conversationHistory": [
      {
        "role": "assistant",
        "content": "Hey! Your profile caught my eye. You have a great smile.",
        "date": "2024-08-18T10:00:00"
      },
      {
        "role": "user",
        "content": "Thanks! You too. I see you like dogs. I have a golden retriever!",
        "date": "2024-08-18T10:_05:00"
      }
    ]
  },
  "ui_settings": {
    "useEnhancedNlp": true,
    "myLocation": "San Francisco, USA",
    "myProfile": "I love hiking and trying new food spots.",
    "local_model_name": null
  }
}
```
