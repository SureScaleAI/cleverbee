import asyncio
import logging
import traceback
from typing import Dict, Any

import pytest

# Use the adapter's client
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.tools import BaseTool

# Keep the config loading function for convenience
from src.mcp_client import load_mcp_server_configs

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@pytest.mark.asyncio
async def test_mcp_adapter_integration():
    """
    Tests MCP tools loaded via langchain-mcp-adapters using pytest.
    """
    logger.info("--- Starting MCP Adapter Integration Test (pytest) ---")

    # 1. Load MCP Server Configs from mcp.json
    adapter_configs = load_mcp_server_configs()
    if not adapter_configs:
        pytest.fail("Failed to load MCP server configs from mcp.json.")

    if not adapter_configs:
        pytest.fail("No valid MCP server configurations found to pass to the adapter.")

    logger.info(f"Adapter configs prepared for servers: {list(adapter_configs.keys())}")

    try:
        # 2. Initialize MultiServerMCPClient and get tools
        logger.info("Initializing MultiServerMCPClient...")
        async with MultiServerMCPClient(adapter_configs) as client:
            logger.info("Client initialized. Getting tools...")
            tools: list[BaseTool] = client.get_tools()
            logger.info(f"Received {len(tools)} LangChain tool(s) from the adapter.")

            tools_dict: Dict[str, BaseTool] = {tool.name: tool for tool in tools}
            logger.info(f"Tool names: {list(tools_dict.keys())}")

            # --- Test get_transcripts (YouTube) ---
            yt_tool_name = "get_transcripts"
            transcript_tool = tools_dict.get(yt_tool_name)
            assert transcript_tool is not None, f"Tool '{yt_tool_name}' not found via adapter."

            logger.info(f"Testing '{yt_tool_name}' tool...")
            try:
                args = {"lang": "en", "url": "https://www.youtube.com/watch?v=1RR7SAcqr5M"}
                result = await transcript_tool.ainvoke(input=args)
                logger.info(f"[{yt_tool_name} Result] (Type: {type(result)})")
                logger.info(f"[{yt_tool_name} Result] (first 300 chars): {str(result)[:300]}...")

                # Basic check: result should be a string containing expected text
                assert isinstance(result, str), f"Expected string result, got {type(result)}"
                assert "data surfer" in result.lower(), f"'data surfer' not found in transcript"
                logger.info(f"[PASS] '{yt_tool_name}' test passed.")

            except Exception as e:
                logger.error(f"[FAIL] '{yt_tool_name}' test failed with exception: {e}")
                logger.error(traceback.format_exc())
                pytest.fail(f"Exception during '{yt_tool_name}' tool execution: {e}")

            # --- Test search_abstracts (PubMed) ---
            pm_tool_name = "search_abstracts"
            pubmed_tool = tools_dict.get(pm_tool_name)
            assert pubmed_tool is not None, f"Tool '{pm_tool_name}' not found via adapter."

            logger.info(f"Testing '{pm_tool_name}' tool...")
            try:
                # 1. Log and call with nested 'request' argument (matches main app)
                args_nested = {"term": "covid vaccine efficacy", "retmax": 2}
                input_data_nested = {"request": args_nested}
                logger.info(f"[TEST] Calling '{pm_tool_name}' with nested argument: {input_data_nested}")
                result_nested = await pubmed_tool.ainvoke(input=input_data_nested)
                logger.info(f"[{pm_tool_name} Nested Result] (Type: {type(result_nested)})")
                logger.info(f"[{pm_tool_name} Nested Result] (first 300 chars): {str(result_nested)[:300]}...")
                assert isinstance(result_nested, str), f"Expected string result, got {type(result_nested)}"
                assert result_nested.strip(), "Received empty string result"
                logger.info(f"[PASS] '{pm_tool_name}' test passed for nested argument.")

                # 2. Log and call with flat argument (no 'request' wrapper)
                args_flat = {"term": "covid vaccine efficacy", "retmax": 2}
                logger.info(f"[TEST] Calling '{pm_tool_name}' with flat argument: {args_flat}")
                result_flat = await pubmed_tool.ainvoke(input=args_flat)
                logger.info(f"[{pm_tool_name} Flat Result] (Type: {type(result_flat)})")
                logger.info(f"[{pm_tool_name} Flat Result] (first 300 chars): {str(result_flat)[:300]}...")
                assert isinstance(result_flat, str), f"Expected string result, got {type(result_flat)}"
                assert result_flat.strip(), "Received empty string result"
                logger.info(f"[PASS] '{pm_tool_name}' test passed for flat argument.")

            except Exception as e:
                logger.error(f"[FAIL] '{pm_tool_name}' test failed with exception: {e}")
                logger.error(traceback.format_exc())
                pytest.fail(f"Exception during '{pm_tool_name}' tool execution: {e}")

    except Exception as e:
        logger.error(f"An error occurred during MultiServerMCPClient initialization or usage: {e}")
        logger.error(traceback.format_exc())
        pytest.fail(f"Error setting up MultiServerMCPClient: {e}")

    logger.info("--- MCP Adapter Integration Test (pytest) Complete ---")