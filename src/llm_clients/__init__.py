# This file makes llm_clients a package.

# Removed import: from .base import LLMClient, LLMResponse

# Exports can be defined here if needed.

# Make this a package and potentially expose clients
# Removed import: from .claude import ClaudeClient
# Removed import: from .gemini import GeminiClient

# LangChain integration is now used for all clients
# Removed: __all__ = ["ClaudeClient", "GeminiClient"] 