import importlib
import logging
import json
import os
from typing import List, Dict, Any, Optional, Type, Union
import inspect
import asyncio
import uuid
import time
import shutil

from langchain_core.tools import BaseTool
from langchain_core.callbacks import BaseCallbackHandler
from pydantic import BaseModel, Field # Keep BaseModel, Field for standard tools

# <<< Import MultiServerMCPClient from the adapter >>>
from langchain_mcp_adapters.client import MultiServerMCPClient

# Import ContentManager only for tool initialization
from src.content_manager import ContentManager # Needed for browser tool dependency

# Import tool classes for model_rebuild
from src.browser import PlaywrightBrowserTool
from src.tools.reddit_search_tool import RedditSearchTool, RedditExtractPostTool

# Ensure all forward references are resolved for Pydantic models
PlaywrightBrowserTool.model_rebuild()
RedditSearchTool.model_rebuild()
RedditExtractPostTool.model_rebuild()

# <<< Kept load_mcp_server_configs import >>>
from src.mcp_client import load_mcp_server_configs

from src.chainlit_callbacks import ChainlitCallbackHandler # Ensure this is imported

# <<< Import AVAILABLE_TOOLS from settings >>>
from config.settings import AVAILABLE_TOOLS

logger = logging.getLogger(__name__)

async def load_tools(
    configs: Dict[str, Dict[str, Any]],
    content_manager: Optional[ContentManager] = None,
    callbacks: Optional[List[BaseCallbackHandler]] = None
) -> List[BaseTool]:
    """
    Loads standard tools based on the provided configuration dictionary.
    MCP tools should be loaded separately by the component managing the MCP client lifecycle.

    Args:
        configs: Dictionary where keys are tool names (must match AVAILABLE_TOOLS) 
                   and values are config dicts for the tool.
        content_manager: An instance of ContentManager, required for tools like PlaywrightBrowserTool.
        callbacks: List of LangChain callback handlers to attach to tools.

    Returns:
        A list of instantiated standard BaseTool objects.
    """
    loaded_tools: List[BaseTool] = []
    logger.info(f"Loading standard tools based on configuration: {list(configs.keys())}")
    
    # Find the ChainlitCallbackHandler if present
    chainlit_handler: Optional[ChainlitCallbackHandler] = None
    if callbacks:
        for handler in callbacks:
            if isinstance(handler, ChainlitCallbackHandler):
                chainlit_handler = handler
                break
    else:
        logger.warning("No callbacks provided to load_tools. CAPTCHA prompts may not work correctly.")

    if not chainlit_handler:
        logger.warning("ChainlitCallbackHandler not found in callbacks. CAPTCHA prompts will fall back to console.")

    # --- Load Standard Tools (non-MCP) --- 
    for name, config in configs.items():
        # Ensure only tools defined in AVAILABLE_TOOLS are processed here
        tool_class_path = AVAILABLE_TOOLS.get(name)
        
        # >>> ADDED CHECK: Skip browser tool here, it's handled in Agent __init__ <<<
        if name == "web_browser":
            logger.debug(f"Skipping tool '{name}' in load_tools (handled by Agent constructor).")
            continue
        # >>> END ADDED CHECK <<<
        
        if tool_class_path:
            try:
                # Dynamically import the tool class from the string path
                if isinstance(tool_class_path, str):
                    module_path, class_name = tool_class_path.rsplit('.', 1)
                    try:
                        module = importlib.import_module(module_path)
                        tool_class = getattr(module, class_name)
                        logger.debug(f"Successfully imported {tool_class_path}")
                    except (ImportError, AttributeError) as e:
                        logger.error(f"Failed to import tool class {tool_class_path}: {e}")
                        continue
                else:
                    # For backward compatibility if AVAILABLE_TOOLS still has direct class references
                    tool_class = tool_class_path
                
                # Check if the tool needs the content_manager
                sig = inspect.signature(tool_class.__init__)
                init_params = sig.parameters
                tool_kwargs = config.copy() # Start with config provided

                # Inject content_manager if required
                if 'content_manager' in init_params and content_manager:
                    tool_kwargs['content_manager'] = content_manager
                elif 'content_manager' in init_params and not content_manager:
                    logger.warning(f"Tool '{name}' requires ContentManager, but none was provided. Skipping.")
                    continue

                # Inject callbacks if accepted
                if 'callbacks' in init_params:
                    tool_kwargs['callbacks'] = callbacks
                elif 'chainlit_handler' in init_params and chainlit_handler:
                    tool_kwargs['chainlit_handler'] = chainlit_handler
                elif 'chainlit_handler' in init_params and not chainlit_handler:
                    logger.warning(f"Tool '{name}' has 'chainlit_handler' param but handler not found in callbacks.")

                # Remove standard keys that might not be init params before instantiation
                tool_kwargs.pop('enabled', None) # Assuming 'enabled' is not an init param

                # Instantiate the tool with potentially modified kwargs
                tool_instance = tool_class(**tool_kwargs)
                loaded_tools.append(tool_instance)
                logger.info(f"Loaded standard tool: {name}")
            except Exception as e:
                logger.error(f"Failed to load standard tool '{name}': {e}", exc_info=True)
        else:
            # Log if a tool name from config is not in AVAILABLE_TOOLS
            logger.warning(f"Tool name '{name}' from config is not in AVAILABLE_TOOLS dictionary. Skipping.")

    logger.info(f"Standard tools loaded: {len(loaded_tools)}. Names: {[tool.name for tool in loaded_tools]}")
    return loaded_tools 

def get_all_tool_metadata(standard_tools: list, mcp_server_configs: dict) -> str:
    """
    Returns a markdown-formatted string listing all available tools, their descriptions, and min/max call limits.
    Args:
        standard_tools: List of loaded BaseTool objects (internal tools)
        mcp_server_configs: Dict of MCP tool configs loaded from mcp.json
    Returns:
        Markdown string for prompt injection
    """
    lines = []
    lines.append('| Tool Name | Description | Min Calls | Max Calls |')
    lines.append('|-----------|-------------|-----------|-----------|')
    # Internal/standard tools
    for tool in standard_tools:
        desc = getattr(tool, 'description', 'No description')
        min_calls = getattr(tool, 'min_calls', None)
        max_calls = getattr(tool, 'max_calls', None)
        # Fallback to config if not set on tool
        if min_calls is None or max_calls is None:
            # Try to get from tool_configs if present
            min_calls = getattr(tool, 'min_tool_calls', 0)
            max_calls = getattr(tool, 'max_tool_calls', 3)
        lines.append(f"| {tool.name} | {desc} | {min_calls} | {max_calls} |")
    # MCP tools from mcp.json
    for name, cfg in mcp_server_configs.items():
        desc = cfg.get('description', 'No description')
        min_calls = cfg.get('minToolCalls', 0)
        max_calls = cfg.get('maxToolCalls', 3)
        lines.append(f"| {name} | {desc} | {min_calls} | {max_calls} |")
    return '\n'.join(lines)

# <<< Removed Example Usage section (main function) >>> 