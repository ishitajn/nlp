import asyncio
import json
from main import run_analysis_pipeline
from model import AnalyzePayload

async def main():
    with open('load.json', 'r') as f:
        # The load.json is a list of concatenated json objects
        # I need to parse it correctly
        data = f.read()
        # a bit of a hack to make it a valid json array
        json_data = json.loads(f"[{data.strip().rstrip(',')}]")

    # The user is interested in the first conversation
    payload_dict = json_data[0]

    # Create an AnalyzePayload object from the dictionary
    payload = AnalyzePayload(**payload_dict)

    # Run the analysis pipeline
    analysis_result = await run_analysis_pipeline(payload)

    # Overwrite analysis.json with the new result
    with open('analysis.json', 'w') as f:
        json.dump(analysis_result, f, indent=4)

if __name__ == "__main__":
    asyncio.run(main())
