import logging
from typing import List, Optional

from langchain_core.tools import BaseTool
from langchain_core.callbacks import BaseCallbackHandler
from langchain_mcp_adapters import tools as mcp_adapter_tools

# Import ContentManager for type hints
from src.content_manager import ContentManager

logger = logging.getLogger(__name__)

async def load_mcp_tools(
    mcp_client,
    callbacks: Optional[List[BaseCallbackHandler]] = None,
    content_manager: Optional["ContentManager"] = None
) -> List[BaseTool]:
    """
    Dynamically loads tools from active MCP client connection using the langchain_mcp_adapters package.
    This function should be called with an active MultiServerMCPClient 
    that will remain open throughout the tool usage.
    
    Args:
        mcp_client: An initialized and active MultiServerMCPClient instance from langchain_mcp_adapters
        callbacks: Optional list of LangChain callback handlers to attach to tools
        content_manager: Optional ContentManager instance to pass to tools that require it
        
    Returns:
        A list of LangChain tool objects wrapped around MCP functions
    """
    if not mcp_client:
        logger.warning("No MCP client provided, cannot load MCP tools")
        return []
    
    logger.info("Loading MCP tools from active client")
    try:
        # Use the MCP client to get tools (per documentation)
        mcp_tools = mcp_client.get_tools()
        
        # If content_manager is provided, attach it to tools that might need it
        if content_manager and mcp_tools:
            for tool in mcp_tools:
                if hasattr(tool, 'content_manager') and tool.content_manager is None:
                    tool.content_manager = content_manager
                    logger.debug(f"Attached ContentManager to MCP tool: {tool.name}")
        
        if mcp_tools:
            tool_names = [t.name for t in mcp_tools]
            logger.info(f"Successfully loaded {len(mcp_tools)} MCP tools: {tool_names}")
        else:
            logger.warning("No MCP tools were loaded from the client")
            
        return mcp_tools
    except Exception as e:
        logger.error(f"Error loading MCP tools: {e}")
        return [] 