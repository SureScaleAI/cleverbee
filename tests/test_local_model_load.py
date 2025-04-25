#!/usr/bin/env python3
"""Minimal script to test loading the configured local model via LlamaCpp."""

import sys
import os
import logging
from pathlib import Path

# Add project root to the Python path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("test_local_model_load")

try:
    from dotenv import load_dotenv
    load_dotenv() # Load any potential env vars needed for settings or keys
    from config.settings import LOCAL_MODELS_DIR, LOCAL_MODEL_NAME, N_GPU_LAYERS
    from langchain_community.llms import LlamaCpp
except ImportError as e:
    logger.error(f"Failed to import necessary modules: {e}")
    logger.error("Please ensure you have run setup.sh and activated the virtual environment.")
    sys.exit(1)

def run_test():
    logger.info("--- Local Model Load Test --- ")

    if not LOCAL_MODEL_NAME:
        logger.error("LOCAL_MODEL_NAME is not set in config.yaml or default_config.yaml")
        return

    logger.info(f"Attempting to load model: {LOCAL_MODEL_NAME}")
    logger.info(f"Model Directory: {LOCAL_MODELS_DIR}")
    logger.info(f"GPU Layers (N_GPU_LAYERS from config): {N_GPU_LAYERS}")

    model_path = Path(LOCAL_MODELS_DIR) / LOCAL_MODEL_NAME
    if not model_path.exists():
        logger.error(f"Model file not found at calculated path: {model_path}")
        logger.error("Please ensure the LOCAL_MODEL_NAME in your config.yaml is correct and the file exists in the LOCAL_MODELS_DIR.")
        return

    if not model_path.is_file():
        logger.error(f"Path exists but is not a file: {model_path}")
        return

    try:
        logger.info(f"Initializing LlamaCpp with n_gpu_layers={N_GPU_LAYERS}...")
        llm = LlamaCpp(
            model_path=str(model_path),
            n_gpu_layers=N_GPU_LAYERS,
            n_ctx=4096,  # Use a smaller context for testing
            n_batch=512,
            temperature=0.1,
            max_tokens=50, # Generate only a few tokens
            verbose=True, # Log llama.cpp details
        )
        logger.info("LlamaCpp client initialized successfully.")

        test_prompt = "Hello! Can you hear me?"
        logger.info(f"Invoking model with prompt: '{test_prompt}'")

        response = llm.invoke(test_prompt)

        logger.info("--- Test Result --- ")
        logger.info(f"Model Response: {response}")
        logger.info("--- Test Successful --- ")

    except Exception as e:
        logger.error(f"--- Test Failed --- ", exc_info=True) # Log traceback on error
        logger.error(f"Error during LlamaCpp initialization or invocation: {e}")
        if "Insufficient Memory" in str(e) or "kIOGPUCommandBufferCallbackErrorOutOfMemory" in str(e) or "llama_decode returned -3" in str(e):
             logger.error("This error often indicates insufficient GPU memory for the number of layers specified by N_GPU_LAYERS in config.yaml.")
             logger.error(f"Current N_GPU_LAYERS setting: {N_GPU_LAYERS}. Try reducing this value in config.yaml.")
        elif "cublas" in str(e).lower() or "metal" in str(e).lower() or "blas" in str(e).lower():
             logger.error("This might indicate an issue with the llama-cpp-python build or GPU driver setup.")
             logger.error("Try reinstalling with: CMAKE_ARGS='-DLLAMA_METAL=on' pip install llama-cpp-python --force-reinstall --no-cache-dir --verbose")


if __name__ == "__main__":
    run_test() 