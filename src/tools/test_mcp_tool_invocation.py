#!/usr/bin/env python3
"""Test script to invoke an MCP tool directly, useful for testing tool configuration."""

import sys
import os
import asyncio
import logging
from pathlib import Path

# Add the project root to the path to allow imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config.settings import TOOLS_CONFIG, PRIMARY_MODEL_TYPE, GEMINI_MODEL_NAME, GEMINI_API_KEY, CLAUDE_MODEL_NAME, ANTHROPIC_API_KEY
from src.llm_clients.factory import get_llm_client
from src.tools.tool_registry import load_mcp_server_configs
from src.tools.mcp_tools import load_mcp_tools
from src.content_manager import ContentManager
from langchain_mcp_adapters.client import MultiServerMCPClient

logger = logging.getLogger("test_mcp_tool_invocation")
logging.basicConfig(level=logging.INFO)

def get_appropriate_llm():
    """Get the right LLM based on the current configuration."""
    if PRIMARY_MODEL_TYPE == "gemini":
        llm = get_llm_client(
            provider="gemini",
            model_name=GEMINI_MODEL_NAME,
            api_key=GEMINI_API_KEY
        )
    else:
        llm = get_llm_client(
            provider="claude",
            model_name=CLAUDE_MODEL_NAME,
            api_key=ANTHROPIC_API_KEY
        )
    return llm

# Example arguments for search_abstracts (adjust as needed)
EXAMPLE_ARGS = {
    "request": {
        "term": "deep learning"
        # Add other required arguments here if needed
    }
}

async def test_search_abstracts(mcp_server_configs):
    # Use the configured LLM provider and model
    llm = get_appropriate_llm()
    content_manager = ContentManager(primary_llm=llm)
    async with MultiServerMCPClient(connections=mcp_server_configs) as mcp_client:
        mcp_tools = await load_mcp_tools(mcp_client=mcp_client, content_manager=content_manager)
        logger.info(f"Loaded MCP tools: {[tool.name for tool in mcp_tools]}")
        search_abstracts_tool = next((t for t in mcp_tools if t.name == "search_abstracts"), None)
        if not search_abstracts_tool:
            logger.error("search_abstracts tool not found among loaded MCP tools!")
            return
        logger.info(f"Invoking search_abstracts with args: {EXAMPLE_ARGS}")
        try:
            # Try both sync and async invocation
            if hasattr(search_abstracts_tool, 'arun'):
                result = await search_abstracts_tool.arun(EXAMPLE_ARGS)
            else:
                result = search_abstracts_tool.run(EXAMPLE_ARGS)
            logger.info(f"search_abstracts result: {result}")
        except Exception as e:
            logger.error(f"Error invoking search_abstracts: {e}", exc_info=True)

if __name__ == "__main__":
    mcp_server_configs = load_mcp_server_configs()
    if not mcp_server_configs:
        logger.error("No MCP server configs found in mcp.json! Exiting.")
        sys.exit(1)
    asyncio.run(test_search_abstracts(mcp_server_configs))
