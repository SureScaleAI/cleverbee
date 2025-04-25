import logging
from typing import Any, Dict
import json
import os

logger = logging.getLogger(__name__)

# --- MCP SERVER REQUEST LOGGING ---
def log_mcp_request(request_type, tool_name, args):
    logger.info(f"[MCP SERVER] Received request: type={request_type}, tool={tool_name}, args={args}")


# Helper to load MCP server configs from mcp.json (reuse/adapt from tool_registry)
def load_mcp_server_configs(mcp_json_path: str = "mcp.json") -> Dict[str, Dict[str, Any]]:
    configs = {}
    full_path = os.path.join(os.getcwd(), mcp_json_path)
    if os.path.exists(full_path):
        try:
            with open(full_path, 'r') as f:
                data = json.load(f)
            raw_configs = data.get("mcpServers", {})
            for name, cfg in raw_configs.items():
                configs[name] = cfg
        except Exception as e:
            logger.error(f"Failed to load or parse MCP server configs from {mcp_json_path}: {e}")
    else:
        logger.warning(f"{mcp_json_path} not found. No MCP server configurations loaded.")
    return configs 