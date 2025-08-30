"""
This script handles the one-time download and conversion of the SentenceTransformer
model to the optimized ONNX format.

This should be run as part of a build/deployment process, not at application runtime.
"""
import logging
from pathlib import Path
from sentence_transformers import SentenceTransformer
from optimum.onnxruntime import ORTModelForFeatureExtraction

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def build_onnx_model(model_name: str = 'BAAI/bge-large-en-v1.5'):
    """
    Downloads a SentenceTransformer model and exports it to ONNX format.
    """
    model_dir = Path("models")
    onnx_path = model_dir / f"{model_name.split('/')[-1]}-onnx"

    if onnx_path.exists() and list(onnx_path.glob("*.onnx")):
        logging.info(f"ONNX model already exists at {onnx_path}. Skipping build.")
        return

    logging.info(f"ONNX model not found. Starting one-time export for '{model_name}'...")

    model_dir.mkdir(exist_ok=True, parents=True)

    # Load the original SentenceTransformer model
    original_model = SentenceTransformer(model_name)

    # Save the model in a format that optimum can convert
    # This saves tokenizer_config.json, etc.
    original_model.save(str(onnx_path))

    # Use optimum to load the saved model and export it to ONNX
    # This creates the model.onnx and model_optimized.onnx files
    ort_model = ORTModelForFeatureExtraction.from_pretrained(onnx_path, export=True)

    # Save the final configuration files needed by the ORTModel loader
    ort_model.save_pretrained(save_directory=str(onnx_path))

    logging.info(f"Export complete. ONNX model saved to {onnx_path}")

if __name__ == "__main__":
    build_onnx_model()
