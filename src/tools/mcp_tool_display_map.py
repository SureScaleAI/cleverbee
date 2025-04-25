# This module provides a mapping from MCP internal tool function names to MCP config keys (user-facing names)
# and descriptions, for use in display logic.

import os
import json
from typing import Dict, Any

MCP_JSON_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "mcp.json")

def load_mcp_display_map(mcp_json_path: str = MCP_JSON_PATH) -> Dict[str, dict]:
    """
    Loads a mapping from internal MCP tool function name to config key (user-facing name) and description.
    Returns: {internal_func_name: {'config_key': ..., 'description': ...}}
    """
    if not os.path.exists(mcp_json_path):
        return {}
    with open(mcp_json_path, 'r') as f:
        data = json.load(f)
    mapping = {}
    for config_key, cfg in data.get("mcpServers", {}).items():
        # Heuristic: function name is often in 'function', 'tool', or similar fields, or fallback to config_key
        internal_func_name = cfg.get("function") or cfg.get("tool") or config_key
        mapping[internal_func_name] = {
            "config_key": config_key,
            "description": cfg.get("description", "")
        }
    return mapping
