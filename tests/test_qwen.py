# tests/test_qwen.py

import unittest
import pytest
import os
from pathlib import Path
import logging

# Configure logging for the test
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Langchain and Transformers imports (ensure installed)
try:
    from langchain_community.llms import HuggingFacePipeline
    from langchain.chains.summarize import load_summarize_chain
    from langchain_core.documents import Document
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    from accelerate import init_empty_weights, infer_auto_device_map # For device mapping
except ImportError as e:
    logger.error(f"Missing required libraries for Qwen test: {e}. Please install langchain, langchain-community, torch, transformers, accelerate, sentencepiece, bitsandbytes.")
    # Skip tests if libraries missing
    pytestmark = pytest.mark.skip(reason=f"Missing dependencies: {e}")


@pytest.mark.heavy # Mark as heavy due to model loading
@pytest.mark.asyncio
class TestQwenSummarization(unittest.TestCase): # Renamed class

    @classmethod
    def setUpClass(cls):
        """Load the model once for the test class."""
        cls.model_id = "qwen-2.5-32b" # Use the actual directory name found
        cls.model_dir = Path("models") / cls.model_id # Construct expected path
        
        if not cls.model_dir.exists():
             pytest.skip(f"Qwen model not found at {cls.model_dir}. Run setup.sh. Skipping test.", allow_module_level=True)
             
        logger.info(f"Loading Qwen model from: {cls.model_dir}")
        try:
            # Determine device 
            if torch.backends.mps.is_available():
                device = torch.device("mps")
                dtype = torch.float16 
                logger.info("Using MPS device (Mac ARM GPU)")
            elif torch.cuda.is_available():
                device = torch.device("cuda")
                dtype = torch.bfloat16 
                logger.info("Using CUDA device")
            else:
                device = torch.device("cpu")
                dtype = torch.float32
                logger.info("Using CPU device")

            # Load tokenizer
            cls.tokenizer = AutoTokenizer.from_pretrained(cls.model_dir)

            # Load model
            cls.model = AutoModelForCausalLM.from_pretrained(
                cls.model_dir,
                torch_dtype=dtype,
                device_map="auto",
                trust_remote_code=True
            )
            
            # Create HF Pipeline
            cls.pipe = pipeline(
                "text-generation",
                model=cls.model,
                tokenizer=cls.tokenizer,
                max_new_tokens=512, 
            )
            
            # Wrap in Langchain HF Pipeline
            cls.llm_pipeline = HuggingFacePipeline(pipeline=cls.pipe)
            
            # Create the 'stuff' summarization chain (for error test)
            cls.stuff_chain = load_summarize_chain(cls.llm_pipeline, chain_type="stuff")
            
            # Create the 'map_reduce' summarization chain (for content test)
            cls.map_reduce_chain = load_summarize_chain(cls.llm_pipeline, chain_type="map_reduce")
            
            logger.info("Qwen model and summarization chains loaded successfully.")

        except Exception as e:
             logger.error(f"Failed to load Qwen model or setup chain: {e}", exc_info=True)
             pytest.fail(f"Model/Chain setup failed: {e}")

    @classmethod
    def tearDownClass(cls):
        """Clean up resources if needed."""
        if hasattr(cls, 'model'): del cls.model
        if hasattr(cls, 'tokenizer'): del cls.tokenizer
        if hasattr(cls, 'pipe'): del cls.pipe
        if hasattr(cls, 'llm_pipeline'): del cls.llm_pipeline
        if hasattr(cls, 'stuff_chain'): del cls.stuff_chain
        if hasattr(cls, 'map_reduce_chain'): del cls.map_reduce_chain # Clean up new chain
        
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
             torch.cuda.empty_cache()
        logger.info("Cleaned up Qwen test resources.")

    async def test_qwen_stuff_summarization_large_input_error(self):
        """
        Test that the stuff chain raises RuntimeError with large input for Qwen.
        """
        logger.info("Running test: test_qwen_stuff_summarization_large_input_error")
        # --- Arrange ---
        char_count_target = 78000 # Target slightly above the logged failure point
        large_content = ("This is a sentence repeated many times to simulate large input. " * (char_count_target // 60))[:char_count_target]
        mock_docs = [
            Document(page_content=large_content)
        ]
        logger.info(f"Simulating input with {len(mock_docs)} documents, total ~{len(large_content)} chars.")

        # --- Act & Assert ---
        with pytest.raises(RuntimeError, match=r"(Invalid buffer size|CUDA out of memory|MPS backend out of memory)"):
            logger.info("Invoking stuff_chain.ainvoke...")
            await self.stuff_chain.ainvoke({"input_documents": mock_docs})
            logger.info("ainvoke completed (THIS SHOULD NOT BE REACHED if error occurs)") 

        logger.info("Successfully caught expected RuntimeError for stuff chain.")

    async def test_qwen_map_reduce_summary_content(self):
        """
        Test map_reduce summarization includes key content from multiple chunks.
        """
        logger.info("Running test: test_qwen_map_reduce_summary_content")
        # --- Arrange ---
        key_phrase1 = "The philosophical implications of artificial intelligence are profound."
        key_phrase2 = "Ethical considerations must guide AI development."
        key_phrase3 = "Consciousness remains a challenging topic in AI research."
        
        # Create multiple documents with distinct key phrases
        doc1_content = f"Introduction to AI. {key_phrase1} We explore various models. " * 10 # Repeat for substance
        doc2_content = f"Exploring AI Ethics. {key_phrase2} Bias in algorithms is a major concern. " * 10
        doc3_content = f"Advanced AI Concepts. {key_phrase3} The path to AGI is unclear. " * 10

        mock_docs = [
            Document(page_content=doc1_content),
            Document(page_content=doc2_content),
            Document(page_content=doc3_content),
        ]
        logger.info(f"Simulating map_reduce input with {len(mock_docs)} documents.")

        # --- Act --- 
        try:
            logger.info("Invoking map_reduce_chain.ainvoke...")
            # Use invoke for map_reduce as it might handle async differently internally sometimes
            # result = await self.map_reduce_chain.ainvoke({"input_documents": mock_docs})
            # Using run directly as invoke might have issues with map_reduce in some langchain versions
            result = await self.map_reduce_chain.arun(mock_docs)
            logger.info("map_reduce_chain.arun completed.")
        except Exception as e:
            logger.error(f"map_reduce_chain failed during test: {e}", exc_info=True)
            pytest.fail(f"map_reduce chain execution failed unexpectedly: {e}")
        
        # --- Assert ---
        self.assertIsNotNone(result, "Summarization result should not be None")
        self.assertIsInstance(result, str, "Summarization result should be a string")
        self.assertGreater(len(result), 0, "Summarization result should not be empty")
        logger.info(f"Generated Summary (map_reduce): {result}")
        
        # Check for presence of key phrases (allow for slight rephrasing by model)
        # Making checks case-insensitive and looking for core parts
        self.assertIn("philosophical implications", result.lower(), f"Key phrase 1 core part not found in summary: {result}")
        self.assertIn("ethical considerations", result.lower(), f"Key phrase 2 core part not found in summary: {result}")
        self.assertIn("consciousness remains", result.lower(), f"Key phrase 3 core part not found in summary: {result}")
        
        logger.info("Successfully verified key phrases in map_reduce summary.")


# --- Entry point for running tests ---
if __name__ == '__main__':
    pytest.main([__file__]) 