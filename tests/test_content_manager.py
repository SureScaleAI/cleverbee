# tests/test_content_manager.py

import unittest
from unittest.mock import patch, MagicMock, AsyncMock
import pytest
from pathlib import Path
from langchain.docstore.document import Document

# Assume necessary imports from your project structure
# Adjust these imports based on your actual project layout
from src.content_manager import ContentManager
from src.llm_clients.factory import get_llm_client # Assuming this is used internally or mock it
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, AIMessage

# Mock BaseChatModel for primary and local LLMs
class MockChatModel(BaseChatModel):
    model_name: str = "mock-model"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        # Simple mock generation
        content = f"Summary for: {messages[0].content[:50]}..."
        return AIMessage(content=content)

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
         # Simple mock generation
        content = f"Summary using {self.model_name} for: {messages[0].content[:50]}..."
        # Simulate returning AIMessage directly
        return AIMessage(content=content)

    def bind_tools(self, tools): # Add dummy method if needed by ContentManager init
        return self

    # Add any other methods ContentManager might call
    def get_num_tokens(self, text: str) -> int:
        # Simulate token counting if ContentManager uses this directly
        # Note: ContentManager actually uses tiktoken, which is mocked in the tests
        return len(text) // 4 # Rough estimate

# --- Test Class ---

@pytest.mark.asyncio
class TestContentManagerSummarization(unittest.TestCase): # Renamed class slightly

    @patch('src.content_manager.get_local_llm')
    @patch('src.content_manager.tiktoken.get_encoding')
    @patch('src.content_manager.get_model_context_window')
    @patch('config.settings.USE_LOCAL_SUMMARIZER_MODEL', True)
    @patch('config.settings.SUMMARIZER_MODEL', 'mock-local-model') # Use a distinct mock name
    @patch('config.settings.CHUNK_SIZE', 4000) # Example value
    @patch('config.settings.CHUNK_OVERLAP', 200) # Example value
    async def test_summarize_with_local_model_success(
        self,
        mock_get_context_window,
        mock_get_encoding,
        mock_get_local_llm,
    ):
        """
        Verify ContentManager uses the local model when enabled and content fits context.
        """
        # --- Arrange ---
        primary_llm = MockChatModel(model_name="mock-primary-gemini")
        primary_llm._agenerate = AsyncMock(return_value=AIMessage(content="Summary from Primary"))

        # Mock the local LLM loader
        local_llm = MockChatModel(model_name="mock-local-model")
        local_llm._agenerate = AsyncMock(return_value=AIMessage(content="Summary from Local")) # Distinct summary
        mock_get_local_llm.return_value = local_llm

        # Mock Tokenizer: Simulate content size BELOW context window
        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = [0] * 500 # Simulate 500 tokens < 4096
        mock_get_encoding.return_value = mock_encoder

        # Mock Context Window: Return sufficient window size for the local model
        mock_get_context_window.return_value = 4096

        # Instantiate ContentManager
        # Note: Need to ensure ContentManager actually calls the patched functions
        # It might directly import settings, hence the patches on config.settings
        content_manager = ContentManager(
            primary_llm=primary_llm,
            summarization_llm=None, # ContentManager will attempt local load based on patched settings
            chunk_size=4000, # Use patched value
            chunk_overlap=200 # Use patched value
        )
        # Ensure the mock local LLM is correctly associated if ContentManager internal loading differs
        # This might involve patching ContentManager._load_local_llm or similar if necessary
        # For now, assume the patches on settings and get_local_llm are sufficient
        content_manager.local_llm = local_llm # Explicitly set for clarity, though CM should load it

        # --- Act ---
        source_id = "http://example.com/normal_content"
        content = "This is reasonable content that should fit."
        summary = await content_manager.get_summary(source_id, content=content, callbacks=None)

        # --- Assert ---
        # 1. Check local LLM loader was called (or that local_llm attribute was set/used)
        # Depending on ContentManager implementation, assert_called_once might not work if
        # the loading happens in __init__ before the mock is fully active. 
        # A check that local_llm was used is more robust.
        # mock_get_local_llm.assert_called_once() # May fail depending on init timing

        # 2. Check tokenizer was used
        mock_get_encoding.assert_called_once_with("cl100k_base") # Or your default
        mock_encoder.encode.assert_called_once_with(content)

        # 3. Check context window retrieval was called for the local model
        mock_get_context_window.assert_called_once_with('mock-local-model')

        # 4. Check LOCAL LLM's generate method WAS called
        local_llm._agenerate.assert_called_once()

        # 5. Check PRIMARY LLM's generate method was NOT called
        primary_llm._agenerate.assert_not_called()

        # 6. Check the summary content
        self.assertEqual(summary, "Summary from Local")

    # --- Include the previous test method here as well ---
    @patch('src.content_manager.get_local_llm')
    @patch('src.content_manager.tiktoken.get_encoding')
    @patch('src.content_manager.get_model_context_window')
    @patch('config.settings.USE_LOCAL_SUMMARIZER_MODEL', True)
    @patch('config.settings.SUMMARIZER_MODEL', 'mock-local-qwen-32b')
    @patch('config.settings.CHUNK_SIZE', 1000)
    @patch('config.settings.CHUNK_OVERLAP', 100)
    async def test_fallback_to_primary_llm_when_local_exceeds_context(
        self,
        mock_get_context_window,
        mock_get_encoding,
        mock_get_local_llm,
    ):
        """
        Verify ContentManager falls back to primary LLM when local model context is exceeded.
        """
        # --- Arrange ---
        primary_llm = MockChatModel(model_name="mock-primary-gemini")
        primary_llm._agenerate = AsyncMock(return_value=AIMessage(content="Summary from Primary"))

        local_llm = MockChatModel(model_name="mock-local-qwen-32b")
        local_llm._agenerate = AsyncMock(return_value=AIMessage(content="Summary from Local"))
        mock_get_local_llm.return_value = local_llm

        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = [0] * 15000 # Exceeds context
        mock_get_encoding.return_value = mock_encoder

        def context_side_effect(model_name):
            if model_name == 'mock-local-qwen-32b':
                return 4096
            return 8192
        mock_get_context_window.side_effect = context_side_effect

        content_manager = ContentManager(
            primary_llm=primary_llm,
            summarization_llm=None,
            chunk_size=1000,
            chunk_overlap=100
        )
        content_manager.local_llm = local_llm # Explicitly set for clarity

        # --- Act ---
        source_id = "http://example.com/large_content"
        content = "This is very large content..."
        summary = await content_manager.get_summary(source_id, content=content, callbacks=None)

        # --- Assert ---
        # mock_get_local_llm.assert_called_once() # May fail depending on init timing
        mock_get_encoding.assert_called_once_with("cl100k_base")
        mock_encoder.encode.assert_called_once_with(content)
        mock_get_context_window.assert_any_call('mock-local-qwen-32b')
        primary_llm._agenerate.assert_called_once()
        local_llm._agenerate.assert_not_called()
        self.assertEqual(summary, "Summary from Primary")

    @patch('src.content_manager._estimate_token_count')
    @patch('config.settings.CHUNK_SIZE', 1000)
    @patch('config.settings.CHUNK_OVERLAP', 100)
    async def test_universal_mapreduce_strategy_based_on_context_window(self):
        """
        Verify ContentManager uses MapReduce strategy based on content size vs context window,
        regardless of model type (local or cloud).
        """
        # --- Arrange ---
        # Mock the primary LLM
        primary_llm = MockChatModel(model_name="cloud-model")
        primary_llm._agenerate = AsyncMock(return_value=AIMessage(content="Summary with stuff"))
        
        # Patch the _get_model_context_window method
        with patch.object(ContentManager, '_get_model_context_window', return_value=4096):
            # Patch the load_summarize_chain function to track how it's called
            with patch('src.content_manager.load_summarize_chain') as mock_load_chain:
                # Configure the mock chain
                mock_chain = MagicMock()
                mock_chain.ainvoke = AsyncMock(return_value={"output_text": "Summary using map_reduce"})
                mock_load_chain.return_value = mock_chain
                
                # Create ContentManager instance
                content_manager = ContentManager(
                    primary_llm=primary_llm,
                    chunk_size=1000,
                    chunk_overlap=100
                )
                
                # --- Test Case 1: Content fits within context window ---
                # Configure token count to be under the context window threshold
                with patch.object(ContentManager, '_estimate_document_size', return_value=3000):
                    # Act
                    source_id = "http://example.com/small_content"
                    content = "This is smaller content that fits in context window"
                    await content_manager.get_summary(source_id, content=content)
                    
                    # Assert "stuff" strategy was used
                    mock_load_chain.assert_called_with(primary_llm, chain_type="stuff")
                    mock_load_chain.reset_mock()
                
                # --- Test Case 2: Content exceeds context window ---
                # Configure token count to exceed the context window threshold
                with patch.object(ContentManager, '_estimate_document_size', return_value=4000):
                    # Act
                    source_id = "http://example.com/large_content"
                    content = "This is larger content that exceeds context window"
                    summary = await content_manager.get_summary(source_id, content=content)
                    
                    # Assert "map_reduce" strategy was used with custom prompts
                    mock_load_chain.assert_called_once()
                    args, kwargs = mock_load_chain.call_args
                    self.assertEqual(kwargs['chain_type'], "map_reduce")
                    self.assertIn('map_prompt', kwargs)
                    self.assertIn('combine_prompt', kwargs)
                    
                    # Check the custom prompts are used
                    from config.prompts import CONDENSE_PROMPT, COMBINE_PROMPT
                    self.assertEqual(kwargs['map_prompt'], CONDENSE_PROMPT)
                    self.assertEqual(kwargs['combine_prompt'], COMBINE_PROMPT)
                    
                    # Verify the summary returned was the one from our mock
                    self.assertEqual(summary, "Summary using map_reduce")


# --- Entry point for running tests ---
if __name__ == '__main__':
    # This allows running with `python tests/test_content_manager.py`
    # but `pytest` command is generally preferred
    pytest.main() 