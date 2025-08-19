import uvicorn
import logging
from fastapi import FastAPI, HTTPException
from dating_nlp_bot.schemas import AnalysisPayload, AnalysisOutput
from dating_nlp_bot.pipelines import process_payload

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- FastAPI App Definition ---
app = FastAPI(
    title="Dating Conversation Analyzer API",
    description="An API for NLP-driven analysis of dating conversations.",
    version="3.0.0",
)

# --- API Endpoint ---
@app.post("/analyze", response_model=AnalysisOutput)
def analyze_conversation(payload: AnalysisPayload):
    try:
        logger.info(f"Analyzing matchId: {payload.matchId}")
        payload_dict = payload.model_dump()
        result = process_payload(payload_dict)
        logger.info(f"Successfully analyzed matchId: {payload.matchId}")
        return result
    except Exception as e:
        logger.error(f"Error analyzing matchId: {payload.matchId} - {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An unexpected error occurred during analysis.")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
