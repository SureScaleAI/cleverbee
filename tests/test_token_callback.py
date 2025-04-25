import pytest
import logging
from unittest.mock import patch, MagicMock, AsyncMock
from typing import Dict, Any, List, Optional

from langchain_core.outputs import LLMResult
from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.messages import AIMessage

from src.token_callback import TokenUsageCallbackHandler

@pytest.fixture
def mock_llm_result():
    """Create a mock LLM result with token usage information."""
    result = MagicMock(spec=LLMResult)
    result.llm_output = {
        "token_usage": {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150
        },
        "model_name": "claude-3-7-sonnet-20250219"
    }
    return result

@pytest.fixture
def token_handler():
    """Create a token tracking callback handler for testing."""
    # Create a token handler with mock settings
    with patch('src.token_callback.TRACK_TOKEN_USAGE', True), \
         patch('src.token_callback.LOG_COST_SUMMARY', True), \
         patch('src.token_callback.CLAUDE_COST_PER_1K_INPUT_TOKENS', 0.008), \
         patch('src.token_callback.CLAUDE_COST_PER_1K_OUTPUT_TOKENS', 0.024), \
         patch('src.token_callback.GEMINI_COST_PER_1K_INPUT_TOKENS', 0.00035), \
         patch('src.token_callback.GEMINI_COST_PER_1K_OUTPUT_TOKENS', 0.0007), \
         patch('src.token_callback.GEMINI_SUMMARY_COST_PER_1K_INPUT', 0.0001), \
         patch('src.token_callback.GEMINI_SUMMARY_COST_PER_1K_OUTPUT', 0.0004), \
         patch('src.token_callback.MODEL_PRICING', {
            # Claude models
            "claude": {
                "input_cost_per_1k": 0.008,
                "output_cost_per_1k": 0.024
            },
            # Main Gemini model
            "gemini_main": {
                "input_cost_per_1k": 0.00035,
                "output_cost_per_1k": 0.0007
            },
            # Summarization Gemini model
            "gemini_summary": {
                "input_cost_per_1k": 0.0001,
                "output_cost_per_1k": 0.0004
            }
        }):
        handler = TokenUsageCallbackHandler()
        yield handler

class TestTokenCallback:
    """Tests for the token tracking callback handler."""
    
    @pytest.mark.asyncio
    async def test_on_llm_end_claude(self, token_handler, mock_llm_result):
        """Test that token usage is tracked correctly for Claude models."""
        # Set up the Claude model name
        mock_llm_result.llm_output["model_name"] = "claude-3-7-sonnet-20250219"
        
        # Set up token usage
        mock_llm_result.llm_output["token_usage"] = {
            "prompt_tokens": 1000,
            "completion_tokens": 500,
            "total_tokens": 1500
        }
        
        # Reset the token handler
        token_handler.reset()
        
        # Call on_llm_end with the Claude result
        await token_handler.on_llm_end(mock_llm_result)
        
        # Verify the token usage was tracked correctly
        assert token_handler.prompt_tokens == 1000
        assert token_handler.completion_tokens == 500
        assert token_handler.total_tokens == 1500
        
        # Verify the model usage was tracked correctly
        assert token_handler.model_usage["claude"]["input_tokens"] == 1000
        assert token_handler.model_usage["claude"]["output_tokens"] == 500
        
        # Verify the cost was calculated correctly
        # Claude cost: 1000 * 0.008/1000 + 500 * 0.024/1000 = 0.008 + 0.012 = 0.02
        assert token_handler.model_usage["claude"]["total_cost"] == pytest.approx(0.02)
    
    @pytest.mark.asyncio
    async def test_on_llm_end_gemini(self, token_handler, mock_llm_result):
        """Test that token usage is tracked correctly for Gemini models."""
        # Set up the Gemini model name
        mock_llm_result.llm_output["model_name"] = "gemini-1.5-pro-latest"
        
        # Set up token usage
        mock_llm_result.llm_output["token_usage"] = {
            "prompt_tokens": 2000,
            "completion_tokens": 1000,
            "total_tokens": 3000
        }
        
        # Reset the token handler
        token_handler.reset()
        
        # Call on_llm_end with the Gemini result
        await token_handler.on_llm_end(mock_llm_result)
        
        # Verify the token usage was tracked correctly
        assert token_handler.prompt_tokens == 2000
        assert token_handler.completion_tokens == 1000
        assert token_handler.total_tokens == 3000
        
        # Verify the model usage was tracked correctly
        assert token_handler.model_usage["gemini_main"]["input_tokens"] == 2000
        assert token_handler.model_usage["gemini_main"]["output_tokens"] == 1000
        
        # Calculate the expected cost
        # Gemini cost: 2000 * 0.00035/1000 + 1000 * 0.0007/1000 = 0.0007 + 0.0007 = 0.0014
        expected_cost = (2000 * 0.00035/1000) + (1000 * 0.0007/1000)
        assert token_handler.model_usage["gemini_main"]["total_cost"] == pytest.approx(expected_cost)
    
    @pytest.mark.asyncio
    async def test_on_llm_end_gemini_summary(self, token_handler, mock_llm_result):
        """Test that token usage is tracked correctly for Gemini Flash models used for summarization."""
        # Set up the Gemini summary model name
        mock_llm_result.llm_output["model_name"] = "gemini-2.0-flash"
        
        # Set up token usage
        mock_llm_result.llm_output["token_usage"] = {
            "prompt_tokens": 5000,
            "completion_tokens": 1000,
            "total_tokens": 6000
        }
        
        # Reset the token handler
        token_handler.reset()
        
        # Call on_llm_end with the Gemini Flash result
        await token_handler.on_llm_end(mock_llm_result)
        
        # Verify the token usage was tracked correctly
        assert token_handler.prompt_tokens == 5000
        assert token_handler.completion_tokens == 1000
        assert token_handler.total_tokens == 6000
        
        # Verify the model usage was tracked correctly
        assert token_handler.model_usage["gemini_summary"]["input_tokens"] == 5000
        assert token_handler.model_usage["gemini_summary"]["output_tokens"] == 1000
        
        # Verify the cost was calculated correctly
        # Gemini Flash cost: 5000 * 0.0001/1000 + 1000 * 0.0004/1000 = 0.0005 + 0.0004 = 0.0009
        assert token_handler.model_usage["gemini_summary"]["total_cost"] == pytest.approx(0.0009)
    
    def test_log_summary(self, token_handler, caplog):
        """Test that log_summary outputs the correct information."""
        # Set token usage for different models
        token_handler.model_usage["claude"]["input_tokens"] = 1000
        token_handler.model_usage["claude"]["output_tokens"] = 500
        token_handler.model_usage["claude"]["total_cost"] = 0.02
        
        token_handler.model_usage["gemini_main"]["input_tokens"] = 2000
        token_handler.model_usage["gemini_main"]["output_tokens"] = 1000
        token_handler.model_usage["gemini_main"]["total_cost"] = 0.0014
        
        token_handler.model_usage["gemini_summary"]["input_tokens"] = 5000
        token_handler.model_usage["gemini_summary"]["output_tokens"] = 1000
        token_handler.model_usage["gemini_summary"]["total_cost"] = 0.0009
        
        token_handler.total_tokens = 10500
        token_handler.prompt_tokens = 8000
        token_handler.completion_tokens = 2500
        token_handler.successful_requests = 3
        
        # Configure logging to capture output
        caplog.set_level(logging.INFO)
        
        # Call log_summary
        token_handler.log_summary()
        
        # Verify that the summary was logged correctly
        assert "Token Usage and Cost Summary" in caplog.text
        assert "Claude Usage" in caplog.text
        assert "Gemini Main Usage" in caplog.text
        assert "Gemini Summary Usage" in caplog.text
        assert "Total usage across all models" in caplog.text
        # More flexible assertion that doesn't depend on exact formatting
        assert "10" in caplog.text and "500 tokens" in caplog.text
    
    def test_get_total_cost(self, token_handler):
        """Test that get_total_cost returns the correct total cost."""
        # Set costs for different models
        token_handler.model_usage["claude"]["total_cost"] = 0.02
        token_handler.model_usage["gemini_main"]["total_cost"] = 0.0014
        token_handler.model_usage["gemini_summary"]["total_cost"] = 0.0009
        
        # Calculate the total cost
        total_cost = token_handler.get_total_cost()
        
        # Verify the total cost is correct (0.02 + 0.0014 + 0.0009 = 0.0223)
        assert total_cost == pytest.approx(0.0223)
    
    def test_reset(self, token_handler):
        """Test that reset properly clears all token usage statistics."""
        # Set some values
        token_handler.total_tokens = 10000
        token_handler.prompt_tokens = 7000
        token_handler.completion_tokens = 3000
        token_handler.successful_requests = 5
        token_handler.model_usage["claude"]["input_tokens"] = 1000
        token_handler.model_usage["claude"]["output_tokens"] = 500
        token_handler.model_usage["claude"]["total_cost"] = 0.02
        
        # Reset the token handler
        token_handler.reset()
        
        # Verify all values were reset to their initial state
        assert token_handler.total_tokens == 0
        assert token_handler.prompt_tokens == 0
        assert token_handler.completion_tokens == 0
        assert token_handler.successful_requests == 0
        assert token_handler.model_usage["claude"]["input_tokens"] == 0
        assert token_handler.model_usage["claude"]["output_tokens"] == 0
        assert token_handler.model_usage["claude"]["total_cost"] == 0.0
    
    @pytest.mark.asyncio
    async def test_extract_gemini_token_counts_from_usage_metadata(self, token_handler):
        """Test extracting token counts from Gemini AIMessage with usage_metadata."""
        # Create a mock AIMessage with usage_metadata
        mock_message = MagicMock(spec=AIMessage)
        mock_message.usage_metadata = {
            "input_tokens": 200,
            "output_tokens": 100,
            "total_tokens": 300,
            "input_token_details": {"cache_read": 0}
        }
        
        # Create kwargs with the mock message
        kwargs = {"messages": [mock_message]}
        
        # Call the extract method
        token_counts = token_handler._extract_gemini_token_counts(kwargs)
        
        # Verify the token counts were extracted correctly
        assert token_counts is not None
        assert token_counts["prompt_tokens"] == 200
        assert token_counts["completion_tokens"] == 100
        assert token_counts["total_tokens"] == 300
    
    @pytest.mark.asyncio
    async def test_on_llm_end_gemini_with_usage_metadata(self, token_handler):
        """Test on_llm_end with Gemini AIMessage containing usage_metadata."""
        # Create a mock LLM result
        mock_result = MagicMock(spec=LLMResult)
        mock_result.llm_output = {}
        
        # Create a mock AIMessage with usage_metadata
        mock_message = MagicMock(spec=AIMessage)
        mock_message.usage_metadata = {
            "input_tokens": 300,
            "output_tokens": 150,
            "total_tokens": 450,
            "input_token_details": {"cache_read": 0}
        }
        
        # Set up the kwargs with the message and model info
        kwargs = {
            "messages": [mock_message],
            "invocation_params": {"model": "gemini-1.5-pro-latest"}
        }
        
        # Reset the token handler
        token_handler.reset()
        
        # Set the current model
        token_handler.current_model = "gemini_main"
        
        # Call on_llm_end with the mock result and kwargs
        await token_handler.on_llm_end(mock_result, **kwargs)
        
        # Verify the token usage was tracked correctly
        assert token_handler.prompt_tokens == 300
        assert token_handler.completion_tokens == 150
        assert token_handler.total_tokens == 450
        
        # Verify the model usage was tracked correctly
        assert token_handler.model_usage["gemini_main"]["input_tokens"] == 300
        assert token_handler.model_usage["gemini_main"]["output_tokens"] == 150
        
        # Calculate the expected cost
        # Gemini cost: 300 * 0.00035/1000 + 150 * 0.0007/1000
        expected_cost = (300 * 0.00035/1000) + (150 * 0.0007/1000)
        assert token_handler.model_usage["gemini_main"]["total_cost"] == pytest.approx(expected_cost)
    
    @pytest.mark.asyncio
    async def test_gemini_real_world_scenario(self, token_handler):
        """Test a real-world scenario with Gemini 1.5 Pro model.
        
        This test simulates all the different ways token usage data might be structured
        with the Gemini 1.5 Pro model in production.
        """
        # Mock LLMResult
        mock_result = MagicMock(spec=LLMResult)
        mock_result.llm_output = {}
        
        # Create a more complex AIMessage with nested usage data
        mock_message = MagicMock(spec=AIMessage)
        # Primary usage_metadata (method 1)
        mock_message.usage_metadata = {
            "input_tokens": 350,
            "output_tokens": 200,
            "total_tokens": 550,
            "input_token_details": {"cache_read": 0}
        }
        
        # Add secondary _values structure (method 2)
        mock_message._values = {
            "usage_metadata": {
                "input_tokens": 350,
                "output_tokens": 200,
                "total_tokens": 550
            },
            "content": {
                "usage": {
                    "input_tokens": 350,
                    "output_tokens": 200,
                    "total_tokens": 550
                }
            }
        }
        
        # Add additional_kwargs (method 3)
        mock_message.additional_kwargs = {
            "usage": {
                "input_tokens": 350,
                "output_tokens": 200,
                "total_tokens": 550
            }
        }
        
        # Set up response_metadata (method 4)
        mock_message.response_metadata = {
            "input_tokens": 350,
            "output_tokens": 200,
            "total_tokens": 550
        }
        
        # Test cases with different kwargs structures
        test_cases = [
            # Test case 1: Simple message in messages
            {
                "description": "Simple message in messages",
                "kwargs": {
                    "messages": [mock_message],
                    "invocation_params": {"model": "gemini-1.5-pro-latest"}
                }
            },
            # Test case 2: With run_info
            {
                "description": "With run_info",
                "kwargs": {
                    "messages": [mock_message],
                    "invocation_params": {"model": "gemini-1.5-pro-latest"},
                    "run_info": MagicMock(usage={
                        "input_tokens": 350,
                        "output_tokens": 200,
                        "total_tokens": 550
                    })
                }
            },
            # Test case 3: With raw_response
            {
                "description": "With raw_response",
                "kwargs": {
                    "messages": [mock_message],
                    "invocation_params": {"model": "gemini-1.5-pro-latest"},
                    "raw_response": MagicMock(usage={
                        "input_tokens": 350,
                        "output_tokens": 200,
                        "total_tokens": 550
                    })
                }
            },
            # Test case 4: With direct usage
            {
                "description": "With direct usage",
                "kwargs": {
                    "messages": [mock_message],
                    "invocation_params": {"model": "gemini-1.5-pro-latest"},
                    "usage": {
                        "input_tokens": 350,
                        "output_tokens": 200,
                        "total_tokens": 550
                    }
                }
            }
        ]
        
        # Additional test cases with response object
        response_test_cases = [
            # Test case 5: With response object having usage_metadata
            {
                "description": "With response object having usage_metadata",
                "mock_response": MagicMock(
                    usage_metadata={
                        "input_tokens": 350,
                        "output_tokens": 200,
                        "total_tokens": 550
                    }
                )
            },
            # Test case 6: With response object having usage
            {
                "description": "With response object having usage",
                "mock_response": MagicMock(
                    usage={
                        "prompt_tokens": 350,
                        "completion_tokens": 200,
                        "total_tokens": 550
                    }
                )
            }
        ]
        
        # Test each case without the 'response' keyword argument
        for i, test_case in enumerate(test_cases):
            # Reset the token handler
            token_handler.reset()
            
            # Set the current model
            token_handler.current_model = "gemini_main"
            
            # Call on_llm_end with the mock result and kwargs
            await token_handler.on_llm_end(mock_result, **test_case["kwargs"])
            
            # Verify the token usage was tracked correctly
            assert token_handler.prompt_tokens == 350, f"Case {i+1} failed: prompt tokens"
            assert token_handler.completion_tokens == 200, f"Case {i+1} failed: completion tokens"
            assert token_handler.total_tokens == 550, f"Case {i+1} failed: total tokens"
            
            # Verify the model usage was tracked correctly
            assert token_handler.model_usage["gemini_main"]["input_tokens"] == 350, f"Case {i+1} failed: input tokens"
            assert token_handler.model_usage["gemini_main"]["output_tokens"] == 200, f"Case {i+1} failed: output tokens"
            
            # Calculate the expected cost
            # Gemini cost: 350 * 0.00035/1000 + 200 * 0.0007/1000
            expected_cost = (350 * 0.00035/1000) + (200 * 0.0007/1000)
            assert token_handler.model_usage["gemini_main"]["total_cost"] == pytest.approx(expected_cost), f"Case {i+1} failed: cost"
        
        # Test response object cases separately
        for i, test_case in enumerate(response_test_cases):
            # Reset the token handler
            token_handler.reset()
            
            # Set the current model
            token_handler.current_model = "gemini_main"
            
            # Create specific kwargs for this test - use google_response instead of response
            # to avoid conflict with the positional parameter to on_llm_end
            test_kwargs = {
                "messages": [mock_message],
                "invocation_params": {"model": "gemini-1.5-pro-latest"},
                "google_response": test_case["mock_response"]  # Changed from "response" to "google_response"
            }
            
            # Call on_llm_end with the mock result and kwargs
            await token_handler.on_llm_end(mock_result, **test_kwargs)
            
            # Verify the token usage was tracked correctly
            assert token_handler.prompt_tokens == 350, f"Response case {i+1} failed: prompt tokens"
            assert token_handler.completion_tokens == 200, f"Response case {i+1} failed: completion tokens"
            assert token_handler.total_tokens == 550, f"Response case {i+1} failed: total tokens"
            
            # Verify the model usage was tracked correctly
            assert token_handler.model_usage["gemini_main"]["input_tokens"] == 350, f"Response case {i+1} failed: input tokens"
            assert token_handler.model_usage["gemini_main"]["output_tokens"] == 200, f"Response case {i+1} failed: output tokens"
            
            # Calculate the expected cost
            expected_cost = (350 * 0.00035/1000) + (200 * 0.0007/1000)
            assert token_handler.model_usage["gemini_main"]["total_cost"] == pytest.approx(expected_cost), f"Response case {i+1} failed: cost"
    
    @pytest.mark.asyncio
    async def test_gemini_no_token_usage(self, token_handler, caplog):
        """Test the handler gracefully handles messages without token usage metadata."""
        # Set up logging to capture debug messages
        caplog.set_level(logging.DEBUG)
        
        # Mock LLMResult without token usage
        mock_result = MagicMock(spec=LLMResult)
        mock_result.llm_output = {}
        
        # Create a message without any token usage metadata
        mock_message = MagicMock(spec=AIMessage)
        mock_message.content = "This is a test message with no token information"
        
        # Set up the kwargs
        kwargs = {
            "messages": [mock_message],
            "invocation_params": {"model": "gemini-1.5-pro-latest"}
        }
        
        # Reset the token handler
        token_handler.reset()
        
        # Set the current model
        token_handler.current_model = "gemini_main"
        
        # Call on_llm_end with the mock result and kwargs
        await token_handler.on_llm_end(mock_result, **kwargs)
        
        # Verify that the token usage wasn't tracked
        assert token_handler.prompt_tokens == 0
        assert token_handler.completion_tokens == 0
        assert token_handler.total_tokens == 0
        assert token_handler.model_usage["gemini_main"]["input_tokens"] == 0
        assert token_handler.model_usage["gemini_main"]["output_tokens"] == 0
        assert token_handler.model_usage["gemini_main"]["total_cost"] == 0.0
        
        # Verify the warning was logged
        assert "No token usage data found in LLM response" in caplog.text
        assert "No token counts found in response" in caplog.text 