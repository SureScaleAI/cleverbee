import asyncio
import logging
from typing import Dict, Any, List, Tuple, Optional, Set, Union, Callable, Awaitable
from datetime import datetime
import json
import re
import lxml.html
import lxml.etree
import os
import time
import copy
import inspect
import uuid
import functools
import urllib.parse
import logging.handlers
import ast
import aiohttp

# LangChain core components
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain_core.messages import (
    BaseMessage, 
    HumanMessage, 
    AIMessage, 
    SystemMessage, 
    ToolMessage,
    ChatMessage,
    FunctionMessage
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate, PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableConfig, RunnableParallel
from langchain_core.output_parsers.openai_tools import PydanticToolsParser # Or relevant tool parser
from pydantic import BaseModel, Field # Import directly from pydantic
from langchain_core.exceptions import LangChainException
from langchain_core.callbacks import BaseCallbackHandler
from langchain.callbacks.manager import AsyncCallbackManagerForLLMRun
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.memory import BaseMemory
from langchain.memory import ConversationBufferMemory, ConversationSummaryBufferMemory
from langchain_core.output_parsers import StrOutputParser

# MCP specific import for error handling
try:
    from mcp.shared.exceptions import McpError
except ImportError:
    # Define a fallback for testing if MCP is not available
    class McpError(Exception):
        pass

# Project specific imports
from config.settings import (
    LLM_PROVIDER, GEMINI_MODEL, CLAUDE_MODEL,
    MIN_REGULAR_WEB_PAGES, MAX_REGULAR_WEB_PAGES,
    MIN_POSTS_PER_SEARCH, MAX_POSTS_PER_SEARCH,
    MAX_RESULTS_PER_SEARCH_PAGE, 
    USE_PROGRESSIVE_LOADING,
    GEMINI_API_KEY,
    CONVERSATION_MEMORY_MAX_TOKENS,
    TOOLS_CONFIG,
    MEMORY_KEY,
    PRIMARY_MODEL_TYPE,
    SUMMARY_MAX_TOKENS,
    USE_LOCAL_SUMMARIZER_MODEL,
    SUMMARIZER_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    PRIMARY_MODEL_NAME,
    ENABLE_THINKING,
    THINKING_BUDGET,
    CONDENSE_FREQUENCY,
    FINAL_SUMMARY_MAX_TOKENS,
    NEXT_STEP_MODEL,
)
# Define MAX_ITERATIONS as a constant based on maximum expected pages
MAX_ITERATIONS = 30  # Allow sufficient iterations for all planned tool calls

from config.prompts import (
    INITIAL_RESEARCH_PROMPT,
    NEXT_ACTION_TEMPLATE,
    SUMMARY_PROMPT,
    ACTION_SYSTEM_PROMPT,
    CONDENSE_PROMPT,
    TOOL_CORRECTION_PROMPT
)
# Tool loading
from src.tools.tool_registry import load_mcp_server_configs, load_tools
try:
    from mcp import StdioServerParameters
except ImportError:
    # Define a simple fallback class for testing
    class StdioServerParameters:
        pass

from src.content_manager import ContentManager, extract_mcp_content_universal
from src.llm_clients.factory import get_llm_client # To get the summarization LLM
from src.token_callback import TokenCallbackManager, TokenUsageCallbackHandler # Re-evaluate if needed directly or just via callbacks

# Define internal vs MCP tools
# Internal tools require special handling, all others are treated as generic MCP tools
INTERNAL_TOOLS = {'web_browser', 'reddit_search', 'reddit_extract_post'}

logger = logging.getLogger(__name__)
# Cache root logger level check for efficiency
_log_traceback = logger.isEnabledFor(logging.DEBUG)

# Logging setup
LOG_DIR = '.logs'
# Add date/time to log file name for per-run logs
log_time_str = datetime.now().strftime('%Y%m%d_%H%M%S')
LOG_FILE = os.path.join(LOG_DIR, f'researcher_agent_{log_time_str}.log')
os.makedirs(LOG_DIR, exist_ok=True)
file_handler = logging.handlers.RotatingFileHandler(LOG_FILE, maxBytes=5*1024*1024, backupCount=5)
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
file_handler.setFormatter(file_formatter)
# Only add file handler if not already present
if not any(isinstance(h, logging.handlers.RotatingFileHandler) for h in logger.handlers):
    logger.addHandler(file_handler)
# Set console handler to INFO (or WARNING for less noise)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(levelname)s %(message)s')
console_handler.setFormatter(console_formatter)
if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
    logger.addHandler(console_handler)

def coerce_str_to_dict(val):
    """
    Recursively convert stringified dicts/lists to real dicts/lists.
    """
    if isinstance(val, str):
        val_strip = val.strip()
        # Only try to parse if it looks like a dict or list
        if (val_strip.startswith("{") and val_strip.endswith("}")) or \
           (val_strip.startswith("[") and val_strip.endswith("]")):
            try:
                # Try JSON first
                return json.loads(val_strip)
            except Exception:
                try:
                    # Fallback to ast.literal_eval for single quotes
                    return ast.literal_eval(val_strip)
                except Exception:
                    logging.warning(f"Failed to parse string as dict/list: {val_strip[:100]}")
                    return val
        else:
            return val
    elif isinstance(val, dict):
        return {k: coerce_str_to_dict(v) for k, v in val.items()}
    elif isinstance(val, list):
        return [coerce_str_to_dict(item) for item in val]
    else:
        return val

def normalize_tool_args(tool_args, tool):
    """
    Normalize argument keys in tool_args to match the tool's args_schema, mapping common synonyms.
    This is applied recursively for nested dicts. Logs a warning if a mapping is applied.
    """
    # Only operate on dicts
    if not isinstance(tool_args, dict):
        return tool_args
    
    # Try to get the expected keys from the tool's args_schema (if available)
    expected_keys = set()
    nested_paths = set()  # New: Track nested paths like "request.term"
    schema = getattr(tool, 'args_schema', None)
    
    if schema and isinstance(schema, dict):
        expected_keys = set(schema.keys())
    elif schema and hasattr(schema, 'model_fields'):  # Pydantic v2
        expected_keys = set(schema.model_fields.keys())
        # Try to find nested field patterns by checking for any nested Pydantic models
        for field_name, field in schema.model_fields.items():
            if hasattr(field, 'annotation') and hasattr(field.annotation, 'model_fields'):
                for nested_field in field.annotation.model_fields:
                    nested_paths.add(f"{field_name}.{nested_field}")
    elif schema and hasattr(schema, '__fields__'):    # Pydantic v1
        expected_keys = set(schema.__fields__.keys())
        # Similar attempt for Pydantic v1
        for field_name, field in schema.__fields__.items():
            if hasattr(field, 'type_') and hasattr(field.type_, '__fields__'):
                for nested_field in field.type_.__fields__:
                    nested_paths.add(f"{field_name}.{nested_field}")
    
    # Common synonym mapping (add more as needed)
    synonym_map = {
        'query': 'term',
        'text': 'content',
        'q': 'term',
        'search': 'term',
        'input': 'content',
        'keywords': 'term',
    }
    
    # Build a reverse map for all expected keys
    reverse_map = {v: k for k, v in synonym_map.items() if v in expected_keys}
    
    # Special handling for nested path errors (e.g., "request.term")
    # Extract parent objects and nested fields from error paths
    for path in nested_paths:
        if '.' in path:
            parent, child = path.split('.', 1)
            if parent not in expected_keys:
                continue
                
            # If we have a simple query key in input but need a nested structure
            if 'query' in tool_args and parent not in tool_args and child == 'term':
                # Create nested structure with the query value
                return {parent: {child: tool_args['query']}}
            
            # Handle similar cases for other synonym mappings
            for synonym, canonical in synonym_map.items():
                if synonym in tool_args and canonical == child:
                    # Create nested structure
                    return {parent: {child: tool_args[synonym]}}
    
    # Apply mapping
    new_args = {}
    for k, v in tool_args.items():
        mapped_key = k
        # If the key is a known synonym and the canonical key is expected, map it
        if k in synonym_map and synonym_map[k] in expected_keys:
            mapped_key = synonym_map[k]
            logging.warning(f"Normalized tool arg key '{k}' to '{mapped_key}' for tool '{getattr(tool, 'name', str(tool))}'")
        # If the key is not expected but its synonym is, map it
        elif k not in expected_keys and k in reverse_map:
            mapped_key = reverse_map[k]
            logging.warning(f"Normalized tool arg key '{k}' to '{mapped_key}' for tool '{getattr(tool, 'name', str(tool))}'")
        # Recursively normalize nested dicts
        if isinstance(v, dict):
            v = normalize_tool_args(v, tool)
        new_args[mapped_key] = v
    
    return new_args

class ResearcherAgent:
    """Research agent implementation using standard LangChain BaseChatModel and LCEL.
    
    This agent is responsible for conducting research on a given topic,
    managing tools, and generating a final summary.
    """

    def __init__(
        self, 
        llm_client: BaseChatModel,
        callbacks: Optional[List[BaseCallbackHandler]] = None,
        mcp_server_configs: Optional[Dict[str, Dict[str, Any]]] = None
    ):
        """Initialize the research agent with LLM client and optional callbacks."""
        self.llm_client = llm_client
        self.callbacks = callbacks or []
        self.mcp_server_configs = mcp_server_configs or {}
        
        # Configuration options from settings
        self.enable_thinking = ENABLE_THINKING
        self.thinking_budget = THINKING_BUDGET
        self.model_name = getattr(llm_client, 'model_name', getattr(llm_client, 'model', "unknown"))
        
        # Tool-related attributes
        self.tools = []
        self.tool_invoker = None
        self.planner = None
        self.action_runner = None
        self.browser_tool_instance = None
        self.tool_configs = {}
        self.planned_tool_limits = {}
        
        # LLM instances
        self.llm_with_tools = None
        self.next_step_llm = self.llm_client  # Default to primary LLM
        
        # --- Initialize Next Step LLM Client ---
        try:
            # Use the NEXT_STEP_MODEL setting from config
            self.next_step_llm = get_llm_client(
                provider="gemini", # Assuming next step is always Gemini for now
                model_name=NEXT_STEP_MODEL, # Use the specific config setting
                is_summary_client=False, # It's not for summarization
                is_next_step_client=True, # Specify this is the next step client
                callbacks=self.callbacks, # Pass callbacks for tracking
                use_retry_wrapper=True,  # Add retry functionality
                max_retries=3  # Set maximum retries to 3
            )
            logger.info(f"Initialized Next Step LLM: {NEXT_STEP_MODEL}")
            if not self.next_step_llm:
                raise RuntimeError("Next Step LLM object is None after initialization.")
            
            # Ensure metadata includes safety settings for empty parts
            if hasattr(self.next_step_llm, 'metadata') and isinstance(self.next_step_llm.metadata, dict):
                self.next_step_llm.metadata["sanitize_history"] = True
                logger.info("Added sanitize_history flag to NextStep LLM metadata")
        except Exception as e:
            # Fallback to primary LLM if next step LLM initialization fails
            logger.error(f"Failed to initialize Next Step LLM: {e}. Falling back to primary LLM.")
            self.next_step_llm = self.llm_client  # Fallback to primary LLM
        # --- End Next Step LLM initialization ---
        
        # Initialize the summarization LLM
        try:
            self.summarization_llm = get_llm_client(
                provider="gemini",
                model_name=SUMMARIZER_MODEL,
                is_summary_client=True
            )
            logger.info(f"Initialized summarization LLM: {SUMMARIZER_MODEL}")
        except Exception as e:
            logger.error(f"Failed to initialize summarization LLM: {e}")
            self.summarization_llm = self.llm_client  # Fallback to primary LLM
        # Set final_summary_llm to primary LLM
        self.final_summary_llm = self.llm_client
        
        # Initialize the content manager
        self.content_manager = ContentManager(
            primary_llm=self.llm_client,
            summarization_llm=self.summarization_llm,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        logger.info("ContentManager initialized")
        
        # Initialize browser tool
        try:
            # Find Chainlit handler if present
            chainlit_handler = None
            if self.callbacks:
                for handler in self.callbacks:
                    if type(handler).__name__ == 'ChainlitCallbackHandler':
                        chainlit_handler = handler
                        break
            
            # Import PlaywrightBrowserTool
            from src.browser import PlaywrightBrowserTool
            
            self.browser_tool_instance = PlaywrightBrowserTool(
                content_manager=self.content_manager,
                callbacks=self.callbacks,
                chainlit_callback=chainlit_handler
            )
            logger.info("PlaywrightBrowserTool instance created successfully")
        except Exception as e:
            logger.error(f"Failed to initialize PlaywrightBrowserTool: {e}")
            self.browser_tool_instance = None
        
        # Chain components
        self.system_research_prompt = None
        self.initial_planning_template = None
        self.next_action_template = None
        self.initial_planner_chain = None
        self.action_iteration_chain = None
        self.summarization_chain = None
        self.condensation_chain = None
        
        # Memory and other components
        self.memory = ConversationSummaryBufferMemory(
            llm=self.llm_client,
            max_token_limit=CONVERSATION_MEMORY_MAX_TOKENS,
            memory_key=MEMORY_KEY,
            return_messages=True
        )
        self.mcp_client = None
        
        # Property to store accumulated content
        self._current_accumulated_content = ""
        
        # Planning stages
        self.planning_stages = {"initial_search", "initial_planning"}
        self.current_stage = ""
    
    @property
    def current_accumulated_content(self) -> str:
        """Get the current accumulated content from the most recent research run."""
        return self._current_accumulated_content

    async def _setup_chains_and_tools(self):
        """Set up the chains and tools (including MCP tools) for the agent."""
        logger.info("Setting up chains and tools for the agent (including MCP tools if configured)")

        # Defensive: Always close previous MCP client if present
        if hasattr(self, 'mcp_client') and self.mcp_client is not None:
            try:
                logger.info("Closing previous MCP client before re-initialization")
                await self.mcp_client.__aexit__(None, None, None)
            except Exception as e:
                logger.error(f"Error closing previous MCP client: {e}")
            self.mcp_client = None
        
        # 1. Ensure ContentManager is initialized
        if self.content_manager is None:
            logger.warning("ContentManager not initialized in __init__, creating now")
            self.content_manager = ContentManager(
                primary_llm=self.llm_client,
                summarization_llm=self.summarization_llm,
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP
            )
            
        content_manager = self.content_manager
        logger.info("Using ContentManager for tool setup")
        
        # 2. Load standard tools (excluding browser tool, which is instantiated in __init__)
        other_standard_tools = []
        try:
            other_standard_tools = await load_tools(
                configs=TOOLS_CONFIG, # load_tools will skip 'web_browser'
                content_manager=content_manager,
                callbacks=self.callbacks
            )
            logger.info(f"Loaded {len(other_standard_tools)} other standard tools (excluding browser)")
        except Exception as e:
            logger.error(f"Error loading other standard tools: {e}", exc_info=_log_traceback)
        
        # Combine pre-initialized browser tool (if available) with other loaded tools
        self.tools = []
        if self.browser_tool_instance:
            self.tools.append(self.browser_tool_instance)
            logger.info(f"Added pre-initialized browser tool ('{self.browser_tool_instance.name}') to agent toolset.")
        else:
            logger.warning("Browser tool instance was not available or failed to initialize. Proceeding without it.")
        
        self.tools.extend(other_standard_tools)
        logger.info(f"Tool names after standard tool load: {[tool.name for tool in self.tools]}")
        
        # 3. Load MCP tools if config present
        if self.mcp_server_configs:
            from langchain_mcp_adapters.client import MultiServerMCPClient
            from src.tools.mcp_tools import load_mcp_tools
            try:
                logger.info("Initializing MultiServerMCPClient for MCP tools in agent setup")
                self.mcp_client = None  # Initialize here inside the try block
                self.mcp_client = MultiServerMCPClient(connections=self.mcp_server_configs)
                await self.mcp_client.__aenter__()
                logger.info("MultiServerMCPClient initialized successfully in agent setup")
                try:
                    mcp_tools = await load_mcp_tools(
                        mcp_client=self.mcp_client,
                        callbacks=self.callbacks,
                        content_manager=content_manager
                    )
                    self.tools.extend(mcp_tools)
                    logger.info(f"Added {len(mcp_tools)} MCP tools to the agent's toolset in agent setup")
                    logger.info(f"Tool names after MCP tool load: {[tool.name for tool in self.tools]}")
                    # Log detailed tool schemas for debugging
                    for tool in self.tools:
                        logger.info(f"Tool '{tool.name}' schema: input_keys={getattr(tool, 'input_keys', None)}, args_schema={getattr(tool, 'args_schema', None)}")
                except Exception as e:
                    logger.error(f"Error loading MCP tools in agent setup: {e}", exc_info=_log_traceback)
            except ImportError as ie:
                logger.error(f"Failed to import MCP client in agent setup: {ie}. Running without MCP tools.")
            except Exception as e:
                logger.error(f"Error initializing MCP client in agent setup: {e}", exc_info=_log_traceback)
                if hasattr(self, 'mcp_client') and self.mcp_client:
                    try:
                        await self.mcp_client.__aexit__(None, None, None)  # Clean up if initialization failed
                        logger.info("Closed MCP client after initialization failure")
                    except Exception as exit_err:
                        logger.error(f"Error closing MCP client after initialization failure: {exit_err}")
                self.mcp_client = None
        
        # 4. Set up initial tool configurations
        self._process_initial_tool_configs()
        
        # 5. Bind tools to LLM if supported by the model
        # Log tool schemas before binding
        for tool in self.tools:
            logger.info(f"[Pre-bind] Tool '{tool.name}' schema: input_keys={getattr(tool, 'input_keys', None)}, args_schema={getattr(tool, 'args_schema', None)}")
            
        # Check if the model supports binding tools
        if hasattr(self.llm_client, 'bind_tools'):
            try:
                logger.info("Model supports tool binding, binding tools to LLM")
                self.llm_with_tools = self.llm_client.bind_tools(self.tools)
                # Log tool schemas after binding (if relevant)
                for tool in self.tools:
                    logger.info(f"[Post-bind] Tool '{tool.name}' schema: input_keys={getattr(tool, 'input_keys', None)}, args_schema={getattr(tool, 'args_schema', None)}")
            except Exception as e:
                logger.error(f"Error binding tools to LLM: {e}", exc_info=_log_traceback)
                # Fall back to using the LLM client directly
                self.llm_with_tools = self.llm_client
                logger.info("Using LLM client directly due to binding error")
        else:
            # For models like LlamaCpp that don't support bind_tools, use the LLM client directly
            logger.warning(f"Model ({type(self.llm_client).__name__}) does not support tool binding - using LLM client directly")
            self.llm_with_tools = self.llm_client
        
        # 6. Set up prompt templates and chains
        self.system_research_prompt = SystemMessage(content=INITIAL_RESEARCH_PROMPT.template)
        self.initial_planning_template = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(INITIAL_RESEARCH_PROMPT.template),
            ("human", "{topic} (current date: {current_date})")
        ])
        self.next_action_template = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(ACTION_SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name=MEMORY_KEY),
            ("human", NEXT_ACTION_TEMPLATE)
        ])
        # Ensure action chains use the correct LLM instances
        self.initial_planner_chain = self.initial_planning_template | self.llm_with_tools # Planner uses primary LLM (with tools)
        # <<< USE self.next_step_llm for the action iteration chain >>>
        self.action_iteration_chain = self.next_action_template | self.next_step_llm # Action iteration uses the dedicated next_step_llm (no tools needed here directly)
        
        # Build summarization chain (uses final_summary_llm)
        self.summarization_chain = self._build_summarization_chain()
        
        # Build condensation chain (uses only summarization_llm)
        condense_prompt_template = PromptTemplate.from_template(CONDENSE_PROMPT.template)
        if not self.summarization_llm:
            raise RuntimeError("Summarization LLM is not available for condensation chain")
        self.condensation_chain = condense_prompt_template | self.summarization_llm
        logger.info(f"Building condensation chain using LLM: {getattr(self.summarization_llm, 'model', 'Unknown')}")
        logger.info("Condensation chain successfully built.")
        
        # Log info about the built chain (or lack thereof)
        if self.condensation_chain:
             logger.info("Condensation chain successfully built.")
        else:
             logger.error("Condensation chain could NOT be built due to missing summarizer LLM.")

        logger.info("Successfully set up chains and all tools (standard + MCP, if configured)")

    def _process_initial_tool_configs(self):
        """Process tool configurations and set default min/max call counts."""
        self.tool_configs = {}
        
        # Process each tool and set its configuration
        for tool in self.tools:
            self.tool_configs[tool.name] = {}
            
            # Set default min/max calls based on tool type
            if tool.name == 'web_browser':
                self.tool_configs[tool.name]['min_calls'] = MIN_REGULAR_WEB_PAGES
                self.tool_configs[tool.name]['max_calls'] = MAX_REGULAR_WEB_PAGES
            elif tool.name in ('reddit_search', 'reddit_extract_post'):
                self.tool_configs[tool.name]['min_calls'] = MIN_POSTS_PER_SEARCH
                self.tool_configs[tool.name]['max_calls'] = MAX_POSTS_PER_SEARCH
            else:
                # Default values for other tools
                self.tool_configs[tool.name]['min_calls'] = 0
                self.tool_configs[tool.name]['max_calls'] = 3
        
        logger.info(f"Processed initial tool configurations for {len(self.tool_configs)} tools")

    def _build_summarization_chain(self):
        summary_prompt_template = ChatPromptTemplate.from_messages([
            SystemMessage(content="You are an expert research summarizer."), 
            ("human", SUMMARY_PROMPT), # Expects {topic} and {accumulated_content}
        ])
        # Use the primary LLM for the final summary
        if not self.final_summary_llm:
            raise RuntimeError("Primary LLM is not available for summarization chain (final summary)")
        logger.info(f"Building final summarization chain using PRIMARY LLM: {getattr(self.final_summary_llm, 'model', 'Unknown')}")
        return summary_prompt_template | self.final_summary_llm

    def _calculate_total_processed(self, processed_counts: Dict[str, Any]) -> int:
        """Helper method to calculate total processed items including MCP tools.
        
        Args:
            processed_counts: Dictionary of processed counts
            
        Returns:
            Total processed content count
        """
        # Filter for integer counts before summing
        regular_total = 0
        try:
            # Sum non-MCP tool counts, ensuring each value is an integer
            regular_total = sum(count for key, count in processed_counts.items() 
                            if key not in ['mcp_tools', 'tool_functions'] and isinstance(count, int) and count > 0)
        except Exception as e:
            logger.warning(f"Error calculating regular total in _calculate_total_processed: {e}")
        
        mcp_total = 0
        try:
            # Process MCP tool counts if they exist
            if 'mcp_tools' in processed_counts and isinstance(processed_counts['mcp_tools'], dict):
                # Ensure MCP tool counts are integers before summing
                mcp_total = sum(count for count in processed_counts['mcp_tools'].values() 
                            if isinstance(count, int) and count > 0)
        except Exception as e:
            logger.warning(f"Error calculating MCP total in _calculate_total_processed: {e}")
        
        # Add function-specific counts (these don't contribute to the total since they're already counted elsewhere)
        functions_detail = ""
        try:
            if 'tool_functions' in processed_counts and isinstance(processed_counts['tool_functions'], dict):
                functions_detail = ", ".join([f"{func}: {count}" for func, count in processed_counts['tool_functions'].items()])
                logger.debug(f"Function-specific counts: {functions_detail}")
        except Exception as e:
            logger.warning(f"Error processing function counts in _calculate_total_processed: {e}")
        
        total = regular_total + mcp_total
        logger.debug(f"Calculated total processed: {total} (regular: {regular_total}, mcp: {mcp_total})")
        logger.debug(f"Function usage breakdown: {functions_detail}")
        return total

    async def _process_generic_mcp_tool_output(
        self, 
        tool_name: str, 
        function_name: Optional[str], 
        tool_args: Dict[str, Any], 
        output_str: str, 
        processed_counts: Dict[str, int],
        accumulated_content: str,
        tool_call_id: str, 
        callbacks: Optional[List[BaseCallbackHandler]] = None
    ) -> Tuple[str, str, bool, Dict[str, int]]:
        """Generic handler for any MCP tool output.
        
        Args:
            tool_name: Name of the MCP tool
            function_name: Optional function name within the tool
            tool_args: Arguments passed to the tool
            output_str: String output from the tool
            processed_counts: Dictionary tracking tool usage counts
            accumulated_content: Current accumulated research content
            tool_call_id: ID of the tool call
            callbacks: Optional list of callbacks for LLM calls
            
        Returns:
            Tuple containing:
                - tool_content_for_history: Content to include in tool message history
                - accumulated_content: Updated accumulated content
                - content_added_this_call: Whether content was added (for condensation triggers)
                - processed_counts: Updated processed counts dict
        """
        content_added_this_call = False # Initialize to False
        tool_content_for_history = ""
        display_name = function_name if function_name else tool_name
        
        if not output_str or output_str.startswith("Error:"):
            logger.warning(f"Empty or error output from MCP tool {tool_name} ({display_name}): {output_str}")
            tool_content_for_history = f"Error/Empty Output from MCP Tool ({display_name}): {output_str[:100]}..."
            return tool_content_for_history, accumulated_content, content_added_this_call, processed_counts
        
        # Check if this is a reddit_search with extract_result_index
        is_reddit_extraction = False
        if tool_name == "reddit_search" and isinstance(tool_args, dict) and tool_args.get("extract_result_index") is not None:
            # This is a reddit_search with extract_result_index, so it's both search and extraction
            try:
                # Try to parse the output as JSON (reddit extraction returns a dict with post & comments)
                json_output = json.loads(output_str) if isinstance(output_str, str) else output_str
                is_reddit_extraction = "post" in json_output or "post_content" in json_output
                if is_reddit_extraction:
                    logger.info(f"Detected Reddit extraction output via search with extract_result_index")
            except (json.JSONDecodeError, TypeError, AttributeError):
                # If not valid JSON or doesn't have post/comments, it's not a reddit extraction
                is_reddit_extraction = False
                logger.info(f"Output isn't JSON Reddit extraction, treating as standard output")
        
        # Check for transcript extraction tools generically
        is_transcript_tool = display_name in ["get_transcripts", "get_transcript"]
        
        # Check if this is a search result list
        is_search_result = False
        search_result_list = None
        search_urls = []  # Added: explicit list to store extracted URLs
        
        # Identify search tools by name
        search_tool_names = ["web_search", "reddit_search"]
        is_search_tool = tool_name in search_tool_names or (
            function_name and any(search_name in function_name for search_name in search_tool_names)
        )
        
        # --- Special handling for web_browser search results --- 
        is_web_browser_search = (tool_name == 'web_browser' and isinstance(tool_args, dict) 
                                and tool_args.get('action') == 'search')
                                
        # If this is a search tool, try to parse it as search results
        if (is_search_tool and not is_reddit_extraction) or is_web_browser_search:
            try:
                if isinstance(output_str, str):
                    try:
                        parsed_json = json.loads(output_str)
                        if isinstance(parsed_json, list):
                            search_result_list = parsed_json
                            is_search_result = True
                            logger.info(f"Parsed search results as JSON list with {len(search_result_list)} items")
                            # Extract URLs from search results
                            for item in search_result_list:
                                if isinstance(item, dict):
                                    url = item.get('url', item.get('link', None))
                                    if url:
                                        search_urls.append(url)
                    except json.JSONDecodeError:
                        if re.search(r"\d+\.\s+.*?(\[Link\]|\(https|\(\/|\n\d+\.\s+", output_str):
                            is_search_result = True
                            search_result_list = output_str
                            logger.info(f"Identified text-based search results format")
                            matches = re.findall(r'\((https?://[^\s)]+)\)', output_str)
                            if matches:
                                search_urls.extend(matches)
                                logger.info(f"Extracted {len(matches)} URLs from text-based search results")
                elif isinstance(output_str, list):
                    if output_str and isinstance(output_str[0], dict) and any(
                        key in output_str[0] for key in ["url", "link", "title", "snippet"]
                    ):
                        search_result_list = output_str
                        is_search_result = True
                        logger.info(f"Identified list of dictionaries as search results with {len(search_result_list)} items")
                        for item in search_result_list:
                            if isinstance(item, dict):
                                url = item.get('url', item.get('link', None))
                                if url:
                                    search_urls.append(url)
            except Exception as e:
                logger.warning(f"Error parsing potential search results: {e}")
                is_search_result = False
        
        # Process based on whether this is a content-extraction tool or search result tool
        if (is_reddit_extraction or is_transcript_tool) and not output_str.startswith("Error executing tool call"):
            # This is a content-extraction operation that succeeded
            try:
                logger.info(f"Generating summary for MCP tool output: {tool_name} ({display_name})")
                source_identifier = f"{tool_name}_{function_name if function_name else ''}"
                
                # Add special handling for transcript tools
                if is_transcript_tool:
                    try:
                        # Get source URL for reference
                        source_url = tool_args.get('url', 'Unknown source URL')
                        header = f"--- Transcript from URL: {source_url} ---"
                        
                        # Try to summarize, but handle potential errors
                        try:
                            summary = await self.content_manager.get_summary(source_identifier, content=output_str, callbacks=callbacks)
                        except Exception as e:
                            logger.error(f"Failed to summarize transcript: {e}", exc_info=True)
                            # Use truncated content as fallback (but still count it as content)
                            truncated_output = output_str[:20000] + "... [Transcript truncated]" if len(output_str) > 20000 else output_str
                            summary = truncated_output
                            
                        # --- CONTENT ACCUMULATION ---
                        # Store both summary and full content for the final report
                        accumulated_content += (
                            f"\n\n{header}\n"
                            f"--- BEGIN FULL TRANSCRIPT ---\n"
                            f"{output_str}\n"
                            f"--- END FULL TRANSCRIPT ---\n"
                            f"--- End Transcript ---\n\n"
                        )
                        
                        # Update tracking variables
                        tool_content_for_history = f"Added transcript ({len(output_str)} chars, summary {len(summary)} chars)"
                        content_added_this_call = True
                        
                        # Increment the appropriate counter
                        if 'mcp_tools' not in processed_counts:
                            processed_counts['mcp_tools'] = {}
                        
                        if tool_name not in processed_counts['mcp_tools']:
                            processed_counts['mcp_tools'][tool_name] = 0
                        
                        processed_counts['mcp_tools'][tool_name] += 1
                        
                        # Also track tool function specifically
                        if 'tool_functions' not in processed_counts:
                            processed_counts['tool_functions'] = {}
                        function_key = display_name
                        if function_key not in processed_counts['tool_functions']:
                            processed_counts['tool_functions'][function_key] = 0
                        processed_counts['tool_functions'][function_key] += 1
                            
                        logger.info(f"Processed transcript for {source_url}")
                        return tool_content_for_history, accumulated_content, content_added_this_call, processed_counts
                    
                    except Exception as e:
                        logger.error(f"Error processing transcript: {e}", exc_info=True)
                        # Continue with general processing as fallback
                
                # General summarization for other tool outputs
                try:
                    summary = await self.content_manager.get_summary(source_identifier, content=output_str, callbacks=callbacks)
                    
                    # Add to accumulated content with descriptive header
                    header = f"--- Output from MCP Tool: {tool_name}"
                    if function_name:
                        header += f" (function: {function_name})"
                    header += " ---"
                    
                    # --- MODIFIED CONTENT ACCUMULATION ---
                    # Store both summary and full content for the final report
                    accumulated_content += (
                        f"\n\n<source>\n"
                        f"Source N\n"
                        f"Tool: {tool_name}\n"
                        f"Function: {function_name}\n"
                        f"Parameters: {tool_args}\n"
                        + (f"URL: {tool_args['url']}\n" if isinstance(tool_args, dict) and 'url' in tool_args else "")
                        + f"--- BEGIN FULL CONTENT ---\n"
                        f"{output_str}\n"
                        f"--- END FULL CONTENT ---\n"
                        f"</source>\n\n"
                    )
                    # --- END MODIFICATION ---
                    
                    # Update tool_content_for_history and tracking variables
                    tool_content_for_history = f"Summarized {len(output_str)}-char output from MCP tool {display_name} into {len(summary)}-char summary."
                    content_added_this_call = True # Set to True ONLY if summarized content is added
                    
                    # Increment the appropriate counter
                    if 'mcp_tools' not in processed_counts:
                        processed_counts['mcp_tools'] = {}
                    
                    if tool_name not in processed_counts['mcp_tools']:
                        processed_counts['mcp_tools'][tool_name] = 0
                    
                    processed_counts['mcp_tools'][tool_name] += 1
                    logger.info(f"Incremented count for MCP tool '{tool_name}' to {processed_counts['mcp_tools'][tool_name]}")
                    
                except Exception as e:
                    logger.error(f"Failed to summarize MCP tool output for {tool_name} ({display_name}): {e}", exc_info=True)
                    # Fallback to including truncated output
                    truncated_output = output_str[:10000] + "... [Content truncated]" if len(output_str) > 10000 else output_str
                    
                    header = f"--- Output from MCP Tool: {tool_name}"
                    if function_name:
                        header += f" (function: {function_name})"
                    header += " (summarization failed) ---"
                    
                    # --- MODIFIED ERROR ACCUMULATION ---
                    # Store truncated raw content as summary fallback, but full content for final summary
                    summary_fallback = truncated_output
                    accumulated_content += (
                        f"\n\n{header}\n"
                        f"--- BEGIN FULL CONTENT ---\n"
                        f"{truncated_output}\n"
                        f"--- END FULL CONTENT ---\n"
                        f"--- End MCP Tool Output ---\n\n"
                    )
                    # --- END MODIFICATION ---
                    tool_content_for_history = f"Warning: Failed to summarize output from MCP tool {display_name}. Included truncated raw output."
                    content_added_this_call = True # Set to True if truncated raw output is added due to summary error
            except Exception as e:
                logger.error(f"Unexpected error in MCP tool output processing: {e}", exc_info=True)
                # Final fallback - add a very truncated version
                truncated_output = output_str[:5000] + "... [Content severely truncated due to error]" if len(output_str) > 5000 else output_str
                header = f"--- Output from MCP Tool (processing error): {tool_name} ---"
                accumulated_content += f"\n\n{header}\n{truncated_output}\n--- End MCP Tool Output ---\n"
                tool_content_for_history = f"Error processing MCP tool output: {e}"
                content_added_this_call = True
        elif is_search_result:
            # --- CRITICAL: Always summarize ALL search results (including Reddit) as a markdown table using the summarizer LLM ---
            # This ensures that URLs and post info are available for subsequent extraction steps.
            # Do NOT use regex/manual formatting for Reddit; always use the LLM-based table for both Google and Reddit search results.
            if isinstance(search_result_list, list) and search_result_list and isinstance(search_result_list[0], dict):
                logger.info("Summarizing search results as markdown table using summarizer model (Google, Reddit, etc).")
                table_md = await self.content_manager.summarize_search_results_as_table(search_result_list, top_n=5, callbacks=callbacks)
                header = f"--- Search Results Table from MCP Tool: {tool_name}"
                if function_name:
                    header += f" (function: {function_name})"
                header += " ---"
                accumulated_content += f"\n\n{header}\n{table_md}\n--- End Search Results Table ---\n"
                tool_content_for_history = table_md
                logger.info("Added markdown table of search results to accumulated_content and scratchpad.")
            else:
                # Fallback: previous formatting for text-based or unknown results
                header = f"--- Search Results from MCP Tool: {tool_name}"
                if function_name:
                    header += f" (function: {function_name})"
                header += " ---"
                display_output = output_str[:20000] + "... [Output truncated]" if len(output_str) > 20000 else output_str
                accumulated_content += f"\n\n{header}\n{display_output}\n--- End Search Results ---\n"
                tool_content_for_history = display_output
                logger.info("Added fallback search results to accumulated_content and scratchpad.")
            # content_added_this_call remains False for search/list results (for condensation purposes)
            result_count = len(search_result_list) if isinstance(search_result_list, list) else "text-based"
            logger.info(f"Formatted {result_count} search results from {tool_name} for agent scratchpad")
            return tool_content_for_history, accumulated_content, content_added_this_call, processed_counts
        else:
            # For other operations that aren't content extractions or search results,
            # include output directly but don't increment the counter
            header = f"--- Output from MCP Tool: {tool_name}"
            if function_name:
                header += f" (function: {function_name})"
            header += " ---"
            
            # Truncate if needed
            display_output = output_str[:20000] + "... [Output truncated]" if len(output_str) > 20000 else output_str
            accumulated_content += f"\n\n{header}\n{display_output}\n--- End Output ---\n"
            
            tool_content_for_history = display_output  # Include the actual output in the history
            # content_added_this_call remains False for non-extractions
            logger.info(f"Added output from MCP tool {tool_name} ({display_name}) to accumulated content (not counting towards condensation).")
        
        return tool_content_for_history, accumulated_content, content_added_this_call, processed_counts

    async def run_research(self, topic: str, current_date: str, callbacks: Optional[List[BaseCallbackHandler]] = None) -> str:
        """Main entry point for running the research workflow.
        
        Args:
            topic: The research topic or query
            current_date: Current date string for prompt context
            callbacks: Optional callbacks for this specific run
            
        Returns:
            Summarized research report
        """
        # When callbacks are passed, they override the instance callbacks for this run
        run_callbacks = callbacks or self.callbacks
        mcp_client_needs_cleanup = False
        
        # Set start time before any operations
        start_time = datetime.now()
        
        try:
            # Validate all tools are properly initialized
            if not hasattr(self, 'tools') or not self.tools:
                logger.warning("Tools not initialized. Calling _setup_chains_and_tools...")
                await self._setup_chains_and_tools()
                
            # Reset or initialize processing counters
            processed_counts = {
                'regular_web_pages': 0,
                'reddit_posts': 0,
                'other': 0,
                'tool_functions': {},
                'base_tool_calls': {} # <<< ADDED: Track base tool totals
            }
            
            # Enforce MCP client lifecycle if not handled externally
            if hasattr(self, 'mcp_client') and self.mcp_client is None and self.mcp_server_configs:
                from langchain_mcp_adapters.client import MultiServerMCPClient
                try:
                    logger.info("Initializing MCP client in run_research")
                    self.mcp_client = MultiServerMCPClient(connections=self.mcp_server_configs)
                    await self.mcp_client.__aenter__()
                    mcp_client_needs_cleanup = True
                    logger.info("MCP client initialized in run_research")
                except Exception as e:
                    logger.error(f"Error initializing MCP client in run_research: {e}")
                    # Continue without MCP tools if initialization fails
            
            # Create a custom run config for this specific research run
            run_config = {
                "callbacks": run_callbacks
            }
            
            # Set initial values for all args
            accumulated_content = ""
            condensed_content_for_prompt = ""
            content_added_since_last_condense = 0
            condense_frequency = CONDENSE_FREQUENCY
            consecutive_errors = 0
            max_iterations = MAX_REGULAR_WEB_PAGES + 5  # Initial estimate, may be updated
            history = []
            warning_count = 0
            dynamic_tool_limits = {}
            
            # Call the core research logic
            final_accumulated_content = await self._run_research_core(
                topic=topic,
                current_date=current_date,
                run_config=run_config,
                accumulated_content=accumulated_content,
                condensed_content_for_prompt=condensed_content_for_prompt,
                processed_counts=processed_counts,
                content_added_since_last_condense=content_added_since_last_condense,
                condense_frequency=condense_frequency,
                consecutive_errors=consecutive_errors,
                max_iterations=max_iterations,
                history=history,
                warning_count=warning_count,
                dynamic_tool_limits=dynamic_tool_limits,
                start_time=start_time  # Pass start_time to the core function
            )
            
            # Generate and return summary report based on research
            report = await self._generate_summary(
                topic=topic,
                accumulated_content=final_accumulated_content, 
                run_config=run_config,
                content_counts=processed_counts
            )
            
            # Calculate and log total time
            elapsed_time = datetime.now() - start_time
            logger.info(f"Research completed in {elapsed_time.total_seconds():.1f} seconds")
            logger.info(f"Sources processed: {processed_counts['regular_web_pages']} web pages, "
                       f"{processed_counts['reddit_posts']} reddit posts, Other: {processed_counts['other']}")
            
            return report
            
        except Exception as e:
            logger.error(f"Error in run_research: {e}", exc_info=True)
            # Ensure accumulated content is preserved even in case of failure
            # If _run_research_core was called and set _current_accumulated_content, it will already be set
            # If the error happened before _run_research_core was called, set it to a default error message
            if not self._current_accumulated_content:
                self._current_accumulated_content = f"Research could not be completed due to an error: {e}"
            
            # Generate summary even if there was an error
            try:
                report = await self._generate_summary(
                    topic=topic,
                    accumulated_content=self._current_accumulated_content, 
                    run_config=run_config,
                    content_counts=processed_counts if 'processed_counts' in locals() else None
                )
                logger.info("Generated summary report despite research errors")
                return report
            except Exception as summary_error:
                logger.error(f"Failed to generate summary after research error: {summary_error}", exc_info=True)
                return f"Research failed: {e}\nCould not generate summary: {summary_error}"
            
        finally:
            # Clean up MCP client if we initialized it
            if mcp_client_needs_cleanup and hasattr(self, 'mcp_client') and self.mcp_client:
                try:
                    logger.info("Cleaning up MCP client in run_research finally block")
                    await self.mcp_client.__aexit__(None, None, None)
                    logger.info("MCP client closed successfully")
                    self.mcp_client = None
                except Exception as e:
                    logger.error(f"Error cleaning up MCP client: {e}")
                    self.mcp_client = None
            
            # Close all open browser pages but keep the browser running
            try:
                from src.browser_manager import browser_manager
                logger.info("Closing all browser pages at the end of the research")
                await browser_manager.close_all_pages()
                logger.info("All browser pages closed successfully")
            except Exception as e:
                logger.error(f"Error closing browser pages: {e}", exc_info=True)

            # --- Final Token Usage Reporting --- 
            active_callbacks = run_callbacks if 'run_callbacks' in locals() else self.callbacks
            token_stats_logged = False
            
            if active_callbacks:
                # First, look for TokenCallbackManager which is the preferred way
                for handler in active_callbacks:
                    if type(handler).__name__ == 'TokenCallbackManager':
                        try:
                            logger.info("Found TokenCallbackManager for final token reporting")
                            if hasattr(handler, 'model_usage') and handler.model_usage:
                                model_count = len(handler.model_usage)
                                total_tokens = sum(
                                    usage.get('input_tokens', 0) + usage.get('output_tokens', 0) 
                                    for usage in handler.model_usage.values()
                                )
                                total_cost = handler.get_total_cost()
                                logger.info(f"TokenCallbackManager stats: {model_count} models, {total_tokens:,} total tokens, ${total_cost:.4f} cost")
                                token_stats_logged = True
                            elif hasattr(handler, 'handler') and hasattr(handler.handler, 'token_cost_processor'):
                                processor = handler.handler.token_cost_processor
                                if hasattr(processor, 'token_usage') and processor.token_usage:
                                    total_tokens = sum(
                                        usage.get('prompt', 0) + usage.get('completion', 0)
                                        for usage in processor.token_usage.values()
                                    )
                                    total_cost = processor.total_cost
                                    model_count = len(processor.token_usage)
                                    logger.info(f"Token processor stats: {model_count} models, {total_tokens:,} total tokens, ${total_cost:.4f} cost")
                                    token_stats_logged = True
                        except Exception as e:
                            logger.error(f"Error extracting token stats from TokenCallbackManager: {e}", exc_info=True)
                        # No break here, continue checking other handlers
                
                # If TokenCallbackManager didn't work or didn't fully capture, try TokenUsageCallbackHandler
                if not token_stats_logged: # Check if we still need stats
                    for handler in active_callbacks:
                        if type(handler).__name__ == 'TokenUsageCallbackHandler':
                            try:
                                if hasattr(handler, 'token_cost_processor'):
                                    processor = handler.token_cost_processor
                                    if hasattr(processor, 'token_usage') and processor.token_usage:
                                        total_tokens = sum(
                                            usage.get('prompt', 0) + usage.get('completion', 0)
                                            for usage in processor.token_usage.values()
                                        )
                                        total_cost = processor.total_cost
                                        model_count = len(processor.token_usage)
                                        logger.info(f"TokenUsageCallbackHandler stats: {model_count} models, {total_tokens:,} total tokens, ${total_cost:.4f} cost")
                                        token_stats_logged = True
                            except Exception as e:
                                logger.error(f"Error extracting token stats from TokenUsageCallbackHandler: {e}", exc_info=True)
                            # No break here
                
                # Last resort: Look for ChainlitCallbackHandler
                if not token_stats_logged: # Check if we still need stats
                    for handler in active_callbacks:
                        if type(handler).__name__ == 'ChainlitCallbackHandler' and hasattr(handler, 'token_manager'):
                            try:
                                logger.info("Found ChainlitCallbackHandler with token_manager")
                                token_manager = handler.token_manager
                                if token_manager and hasattr(token_manager, 'handler') and hasattr(token_manager.handler, 'token_cost_processor'):
                                    processor = token_manager.handler.token_cost_processor
                                    if hasattr(processor, 'token_usage') and processor.token_usage:
                                        total_tokens = sum(
                                            usage.get('prompt', 0) + usage.get('completion', 0)
                                            for usage in processor.token_usage.values()
                                        )
                                        total_cost = processor.total_cost
                                        model_count = len(processor.token_usage)
                                        logger.info(f"ChainlitCallbackHandler token_manager stats: {model_count} models, {total_tokens:,} total tokens, ${total_cost:.4f} cost")
                                        token_stats_logged = True
                            except Exception as e:
                                logger.error(f"Error extracting token stats from ChainlitCallbackHandler: {e}", exc_info=True)
                            # No break here
            
            if not token_stats_logged:
                logger.warning("Could not find TokenUsageCallbackHandler or TokenCallbackManager with appropriate attributes to report final token usage.")
            # --- End Token Usage Reporting ---

    async def _run_research_core(
        self,
        topic: str,
        current_date: str,
        run_config: RunnableConfig,
        accumulated_content: str,
        condensed_content_for_prompt: str,
        processed_counts: Dict[str, int],
        content_added_since_last_condense: int,
        condense_frequency: int,
        consecutive_errors: int,
        max_iterations: int,
        history: List[BaseMessage],
        warning_count: int,
        dynamic_tool_limits: Dict[str, Dict[str, int]],
        start_time: datetime = None  # Add start_time as an optional argument
    ) -> str:
        # Reset base_tool_calls if not already present in processed_counts (e.g., if passed in)
        if 'base_tool_calls' not in processed_counts:
            processed_counts['base_tool_calls'] = {}
            
        if start_time is None:
            start_time = datetime.now()
        logger.info("Phase 1: Initial Planning...")
        self.current_stage = "initial_planning" # For potential thinking logic
        current_response = None # Initialize current_response
        try:
            planner_input = {
                "topic": topic, 
                "current_date": current_date,
                "min_regular_web_pages": MIN_REGULAR_WEB_PAGES,
                "max_regular_web_pages": MAX_REGULAR_WEB_PAGES,
                "min_posts": MIN_POSTS_PER_SEARCH,   # <<< Added
                "max_posts": MAX_POSTS_PER_SEARCH,    # <<< Added
                "tool_metadata": self._get_tool_metadata_string(),  # <<< Added
            }
            # Add thinking budget if applicable
            if self.enable_thinking and "claude-3-7" in self.model_name: # Check model supports it
                 planner_input['max_tokens'] = self.thinking_budget
                 logger.debug(f"Invoking initial planner with thinking budget: {self.thinking_budget}")

            logger.debug(f"Initial Planner Input: {planner_input}")
            planner_response: AIMessage = await self.initial_planner_chain.ainvoke(
                planner_input,
                config=run_config
            )
            # Log raw response only at DEBUG
            logger.debug(f"Initial Planner Raw Response: {planner_response}")

            # === Parse Dynamic Tool Limits from Planner Response ===
            planner_content_str = ""
            if isinstance(planner_response.content, list):
                # Join list elements, assuming the main text is first and JSON block is last
                # This might need adjustment if the format varies
                planner_content_str = "\n".join(str(item) for item in planner_response.content)
                logger.debug("Planner response content was a list, joined for parsing.")
            elif isinstance(planner_response.content, str):
                planner_content_str = planner_response.content
            else:
                 logger.warning(f"Unexpected planner response content type: {type(planner_response.content)}. Skipping dynamic limit parsing.")

            if planner_content_str: # Proceed only if we have a string to parse
                try:
                    # Regex to find the markdown table block under '### Planned Tool Calls'
                    table_match = re.search(r"### Planned Tool Calls\s*\n((?:\|.*\n)+)", planner_content_str)
                    if table_match:
                        table_str = table_match.group(1)
                        table_lines = [line.strip() for line in table_str.strip().split('\n') if line.strip()]
                        # Expect at least 3 lines: header, separator, at least one data row
                        if len(table_lines) >= 3:
                            # Skip header and separator
                            for row in table_lines[2:]:
                                # Split row into columns
                                cols = [col.strip() for col in row.strip('|').split('|')]
                                if len(cols) >= 3:
                                    tool_name = cols[0]
                                    try:
                                        min_val = int(cols[1])
                                        max_val = int(cols[2])
                                    except ValueError:
                                        logger.warning(f"Could not parse min/max as int for tool '{tool_name}': min='{cols[1]}', max='{cols[2]}'")
                                        continue
                                    # Validate against absolute limits from initial config
                                    if tool_name in self.tool_configs:
                                        abs_min = self.tool_configs[tool_name].get('min_calls', 0)
                                        abs_max = self.tool_configs[tool_name].get('max_calls', float('inf'))
                                        # Clamp planned values to absolute limits
                                        parsed_min = max(abs_min, min_val)
                                        parsed_max = min(abs_max, max_val)
                                        if parsed_min > parsed_max: # Ensure min <= max after clamping
                                            parsed_min = parsed_max
                                            logger.warning(f"Planned min ({min_val}) for {tool_name} exceeded planned max ({max_val}) after clamping to absolute limits ({abs_min}-{abs_max}). Setting min = max = {parsed_max}.")
                                        dynamic_tool_limits[tool_name] = {'min': parsed_min, 'max': parsed_max}
                                        logger.info(f"Parsed dynamic limits for {tool_name}: min {parsed_min}, max {parsed_max} (Original plan: min {min_val}, max {max_val})")
                                    else:
                                        logger.warning(f"Planner specified limits for tool '{tool_name}' which is not in the configured tools. Ignoring.")
                                else:
                                    logger.warning(f"Could not parse planned tool call row: '{row}' (cols: {cols})")
                        else:
                            logger.info("'Planned Tool Calls' table found but not enough rows to parse.")
                    else:
                        logger.info("No 'Planned Tool Calls' table found in planner response.")

                    # --- Update self.tool_configs with dynamic limits --- 
                    if dynamic_tool_limits:
                        logger.info("Updating tool configurations with dynamically planned limits.")
                        for tool_name, limits in dynamic_tool_limits.items():
                            if tool_name in self.tool_configs:
                                self.tool_configs[tool_name]['min_calls'] = limits['min']
                                self.tool_configs[tool_name]['max_calls'] = limits['max']
                                logger.debug(f"Updated config for {tool_name}: {self.tool_configs[tool_name]}")
                        logger.info(f"Final effective tool configurations: {self.tool_configs}")

                except Exception as parse_err:
                    logger.error(f"Failed to parse dynamic tool limits from planner response: {parse_err}", exc_info=_log_traceback)
                    # Continue without dynamic limits if parsing fails
            # === End Parsing ===

            # --- Calculate Effective Max Results based on Plan ---
            content_producing_tools = {'web_browser', 'reddit_extract_post'}
            effective_max_results = 0
            for tool_name, config in self.tool_configs.items():
                if tool_name in content_producing_tools:
                    effective_max_results += config.get('max_calls', 0)
            
            # Fallback if no content tools planned or max_calls are zero
            if effective_max_results == 0:
                logger.warning(f"Calculated effective_max_results is 0. Falling back to static MAX_REGULAR_WEB_PAGES: {MAX_REGULAR_WEB_PAGES}")
                effective_max_results = MAX_REGULAR_WEB_PAGES
            else:
                 logger.info(f"Calculated effective_max_results based on planned tool limits: {effective_max_results}")
            # --- End Calculate ---

            # --- NEW: Calculate max_iterations as sum of all max planned tool uses + 20% buffer ---
            if hasattr(self, 'planned_tool_limits') and self.planned_tool_limits:
                total_max_tool_uses = sum(limits.get('max', 0) for limits in self.planned_tool_limits.values())
            else:
                total_max_tool_uses = sum(config.get('max_calls', 0) for config in self.tool_configs.values())
            max_iterations = int(total_max_tool_uses * 1.2 + 0.5)  # Add 20% buffer, round up
            if max_iterations < 20:
                max_iterations = 20  # Set a reasonable minimum
            logger.info(f"Agent max_iterations set to: {max_iterations} (sum of all max tool uses + 20% buffer)")

            # --- Corrected Initial Human Message Formatting ---
            initial_prompt_content_for_history = INITIAL_RESEARCH_PROMPT.template.format(
                topic=topic,
                current_date=current_date,
                min_regular_web_pages=MIN_REGULAR_WEB_PAGES,
                max_regular_web_pages=MAX_REGULAR_WEB_PAGES,
                min_posts=MIN_POSTS_PER_SEARCH, # <<< Added
                max_posts=MAX_POSTS_PER_SEARCH,  # <<< Added
                tool_metadata=self._get_tool_metadata_string() # <<< FIXED: Add tool_metadata
            )
            initial_human_message = HumanMessage(content=initial_prompt_content_for_history)
            # -----------------------------------------------\

            # Start history with the human message and the planner's response
            history.append(initial_human_message)
            history.append(planner_response) # Add AI response right after Human
            logger.debug(f"Initial history after planning (length {len(history)}): {[msg.pretty_repr() for msg in history]}")

            # Add initial plan content to accumulation
            if planner_response.content:
                accumulated_content += f"\n\n--- Initial Plan ---\n{planner_response.content}\n"
                condensed_content_for_prompt = f"--- Initial Plan ---\n{planner_response.content}\n"
                logger.info(f"Initial Planner Response Content (preview): {planner_response.content[:200]}...") # Log preview at INFO
                logger.debug(f"Full Initial Planner Response Content: {planner_response.content}") # Log full content at DEBUG
            else:
                logger.info("Initial planner did not return text content.")

            current_response = planner_response # Assign the planner response to start the loop

        except Exception as e:
            logger.error(f"Error during initial planning: {e}", exc_info=_log_traceback) # Use cached bool for traceback
            return f"Research failed during planning: {e}"

        # --- Phase 2 & 3: Iterative Action Loop ---
        logger.info(f"Phase 3: Iterative Research and Content Processing (max iterations: {effective_max_results + 5})")
        self.current_stage = "research"

        consecutive_errors = 0
        last_failed_tool_info = None # Store info about the last failed tool call

        # Chain is defined in _setup_chains_and_tools

        # Define max_iterations based on MAX_REGULAR_WEB_PAGES
        max_iterations = effective_max_results + 5 # Allow extra steps based on dynamic max
        logger.info(f"Agent max_iterations set to: {max_iterations} (based on effective_max_results: {effective_max_results})")

        # History is already initialized with Human and AI (planner) messages

        tool_map = {tool.name: tool for tool in self.tools} # Define tool_map once before the loop

        # Add tracking for invalid extraction URLs
        invalid_extraction_urls = set()

        for iteration in range(max_iterations):
            logger.info(f"--- Iteration: {iteration + 1}/{max_iterations} ---")
            # --- Iteration logging (fix for TypeError) ---
            total_processed = sum(v for v in processed_counts.values() if isinstance(v, (int, float)))
            logger.info(f"State: Total Processed={total_processed}, Counts={processed_counts}, Consecutive Problems={consecutive_errors}")

            # Ensure we have a response to process from the previous step (or initial plan)
            if not current_response:
                 logger.error("Loop started without a current_response. This should not happen.")
                 return "Research failed due to internal error (missing response)."

            # [LLM_RAW_FUNCTION_CALL_DEBUG] Log the raw LLM output before tool routing
            logger.info(f"[LLM_RAW_FUNCTION_CALL_DEBUG] Iteration {iteration+1}: AIMessage.tool_calls={getattr(current_response, 'tool_calls', None)}, content type={type(current_response.content)}, content preview={str(current_response.content)[:300]}")

            tool_messages: List[ToolMessage] = []
            executed_tool_call_this_iter = False # Flag for this iteration

            needs_condensation = False # Flag to trigger condensation *after* tool processing
            tool_calls_to_execute = [] # Store tool calls to be executed

            # --- 1a. Check for Explicit Tool Calls Attribute ---
            if current_response.tool_calls:
                tool_calls_to_execute = current_response.tool_calls
                executed_tool_call_this_iter = True # Mark that we are processing tools
                logger.info(f"Processing {len(tool_calls_to_execute)} tool calls from AIMessage.tool_calls attribute.")
            # --- 1b. Check for Tool Calls Embedded in Content (Manual Parsing) ---
            elif isinstance(current_response.content, (list, str)):
                content_to_parse = current_response.content
                is_list_content = isinstance(content_to_parse, list)
                log_content_type = "list" if is_list_content else "string"
                logger.debug(f"AIMessage.tool_calls is empty, checking content ({log_content_type}) for tool call block...")
                
                json_block_str = None
                other_content_parts = []
                extracted_from_json = False
                extracted_from_text = False # Flag for new text pattern
                
                # --- NEW: Check for ACTION CONFIRMATION block first ---
                # This handles cases where the LLM correctly formats the ACTION CONFIRMATION block
                # but fails to provide the tool_calls attribute
                confirmation_block_found = False
                confirmation_tool_name = None
                confirmation_args = {}
                
                content_str = content_to_parse
                if is_list_content:
                    content_str = " ".join(str(item) for item in content_to_parse)
                
                if isinstance(content_str, str):
                    # Look for ACTION CONFIRMATION block in the reasoning content
                    confirmation_match = re.search(r"ACTION CONFIRMATION:\s*Tool:\s*(\w+)\s*Parameters:\s*([\s\S]+?)END CONFIRMATION", content_str)
                    if confirmation_match:
                        confirmation_block_found = True
                        confirmation_tool_name = confirmation_match.group(1).strip()
                        params_text = confirmation_match.group(2).strip()
                        
                        # Parse parameters from bulleted list
                        param_lines = re.findall(r'-\s*(\w+):\s*(.+?)(?:\n|$)', params_text)
                        for key, value in param_lines:
                            confirmation_args[key.strip()] = value.strip()
                        
                        logger.info(f"Found ACTION CONFIRMATION block for tool: {confirmation_tool_name} with args: {confirmation_args}")
                        
                        # Create a tool call from the confirmation block
                        if confirmation_tool_name and confirmation_args:
                            tool_call_id = f"confirmation_parsed_{iteration}_{0}"
                            formatted_call = {
                                'name': confirmation_tool_name,
                                'args': confirmation_args,
                                'id': tool_call_id
                            }
                            tool_calls_to_execute = [formatted_call]
                            executed_tool_call_this_iter = True
                            extracted_from_text = True  # Mark extraction success
                            logger.info(f"Created tool call from ACTION CONFIRMATION block: {formatted_call['name']} with args {formatted_call['args']}")
                            
                            # Update current_response for history consistency
                            content_before_confirmation = content_str[:confirmation_match.start()].strip()
                            # Only add a reasoning block if there is non-empty reasoning text
                            if content_before_confirmation:
                                accumulated_content += f"\n\n--- Step {iteration+1} Reasoning/Action ---\n{content_before_confirmation}\n"
                                logger.info(f"LLM Reasoning/Action (preview): {content_before_confirmation[:200]}...")
                                logger.debug(f"Full LLM Reasoning/Action Content: {content_before_confirmation}")
                            current_response = AIMessage(
                                content=content_before_confirmation,
                                tool_calls=[formatted_call],
                                response_metadata=current_response.response_metadata,
                                id=current_response.id
                            )
                            logger.debug("Replaced current_response with AIMessage containing confirmation-parsed tool call.")
                # --- END NEW SECTION ---

                # Only continue with other parsing methods if we haven't extracted a tool call from confirmation block
                if not confirmation_block_found:
                    # Original logic for list content
                    for item in content_to_parse:
                        if isinstance(item, str) and item.strip().startswith('```json'):
                            json_block_str = item.strip().lstrip('```json').rstrip('```').strip()
                            logger.debug(f"Found potential JSON block in list item: {json_block_str}")
                            extracted_from_json = True
                            break 
                        else:
                            other_content_parts.append(str(item))
                    else: # Content is a string
                        # 1. Check for JSON block first
                        json_match = re.search(r"```json\s*({.*?})\s*```", content_to_parse, re.DOTALL)
                        if json_match:
                            json_block_str = json_match.group(1).strip()
                            other_content_parts.append(content_to_parse[:json_match.start()].strip())
                            logger.debug(f"Found potential JSON block in string content: {json_block_str}")
                            extracted_from_json = True
                        else:
                            # 2. If no JSON, check for tool_name key="value" format
                            logger.debug("No JSON block found, checking for tool_name key=value format...")
                            # Regex explanation:
                            # ^                   - Start of a line (due to re.MULTILINE)
                            # ([a-zA-Z_][a-zA-Z0-9_]+) - Capture tool name (group 1)
                            # \\s+                - One or more spaces
                            # ((?:\\w+=\\".*?\\"\\s*)+) - Capture one or more key="value" pairs (group 2)
                            # $                   - End of the line
                            tool_pattern_match = re.search(
                                r"^([a-zA-Z_][a-zA-Z0-9_]+)\s+((?:\w+=\".*?\"\s*)+)$", 
                                content_to_parse.strip(), 
                                re.MULTILINE
                            )
                            
                            if tool_pattern_match:
                                tool_name = tool_pattern_match.group(1)
                                args_str = tool_pattern_match.group(2).strip()
                                logger.info(f"Found potential text-based tool call: name='{tool_name}', args_str='{args_str}'")
                                
                                args_dict = {}
                                try:
                                    for key, value in re.findall(r'(\w+)="(.*?)"', args_str):
                                        args_dict[key] = value
                                    
                                    if args_dict:
                                        tool_call_id = f"text_parsed_{iteration}_{0}"
                                        formatted_call = {
                                            'name': tool_name,
                                            'args': args_dict,
                                            'id': tool_call_id
                                        }
                                        tool_calls_to_execute = [formatted_call]
                                        executed_tool_call_this_iter = True
                                        extracted_from_text = True # Mark extraction success
                                        logger.info(f"Manually parsed 1 text-based tool call: {formatted_call['name']} with args {formatted_call['args']}")
                                        
                                        # Preserve reasoning text before the tool call line
                                        reasoning_text = content_to_parse[:tool_pattern_match.start()].strip()
                                        other_content_parts.append(reasoning_text)

                                        # Update current_response for history consistency
                                        current_response = AIMessage(
                                            content=reasoning_text,
                                            tool_calls=[formatted_call], 
                                            response_metadata=current_response.response_metadata,
                                            id=current_response.id
                                        )
                                        logger.debug("Replaced current_response with AIMessage containing text-parsed tool call.")
                                    else:
                                        logger.warning(f"Failed to parse arguments from text-based tool call string: '{args_str}'")
                                except Exception as text_parse_err:
                                    logger.error(f"Error parsing text-based tool call arguments '{args_str}': {text_parse_err}")
                            
                            # 3. If neither JSON nor text pattern found, treat whole string as content
                            if not extracted_from_json and not extracted_from_text:
                                 logger.debug("No known tool call format (JSON or text pattern) found in string content.")
                                 other_content_parts.append(content_to_parse) # Use original string

                # --- Process extracted JSON block (if found earlier) ---
                if extracted_from_json and json_block_str:
                    try:
                        parsed_tool_call = json.loads(json_block_str)
                        parsed_name = None
                        parsed_args = None

                        # Try standard 'name'/'args' first
                        if isinstance(parsed_tool_call, dict) and 'name' in parsed_tool_call and 'args' in parsed_tool_call:
                            parsed_name = parsed_tool_call['name']
                            parsed_args = parsed_tool_call['args']
                            logger.debug("Parsed tool call using 'name'/'args' keys.")
                        # Fallback to 'tool_name'/'parameters'
                        elif isinstance(parsed_tool_call, dict) and 'tool_name' in parsed_tool_call and 'parameters' in parsed_tool_call:
                            parsed_name = parsed_tool_call['tool_name']
                            parsed_args = parsed_tool_call['parameters']
                            logger.debug("Parsed tool call using 'tool_name'/'parameters' keys.")
                        # Fallback to 'tool_name'/'tool_params' (seen in logs)
                        elif isinstance(parsed_tool_call, dict) and 'tool_name' in parsed_tool_call and 'tool_params' in parsed_tool_call:
                            parsed_name = parsed_tool_call['tool_name']
                            parsed_args = parsed_tool_call['tool_params'] # Use tool_params here
                            logger.debug("Parsed tool call using 'tool_name'/'tool_params' keys.")

                        # If any format was successfully parsed:
                        if parsed_name is not None and parsed_args is not None:
                            tool_call_id = f"json_parsed_{iteration}_{0}" # Use different prefix
                            # Standardize to 'name' and 'args' for downstream use
                            formatted_call = {
                                'name': parsed_name,
                                'args': parsed_args,
                                'id': tool_call_id
                            }
                            tool_calls_to_execute = [formatted_call]
                            executed_tool_call_this_iter = True
                            logger.info(f"Manually parsed 1 JSON tool call from content: {formatted_call['name']} with args {formatted_call['args']}")

                            current_response = AIMessage(
                                content=" ".join(other_content_parts).strip(), # Join potential list parts
                                tool_calls=[formatted_call],
                                response_metadata=current_response.response_metadata,
                                id=current_response.id
                            )
                            logger.debug("Replaced current_response with AIMessage containing JSON-parsed tool call.")
                        else:
                             # Log warning if no format matched
                             logger.warning(f"Parsed JSON block, but doesn't match expected keys ('name'/'args' or 'tool_name'/'parameters' or 'tool_name'/'tool_params'): {parsed_tool_call}")
                    except json.JSONDecodeError as json_err:
                        logger.warning(f"Found JSON block in content, but failed to parse: {json_err}")
                
                # If NO tool call was extracted by any method, ensure other_content_parts has the original content
                if not executed_tool_call_this_iter and not other_content_parts:
                    if isinstance(content_to_parse, list):
                        other_content_parts.extend(str(item) for item in content_to_parse)
                    else:
                        other_content_parts.append(content_to_parse)
                    logger.debug("No tool call extracted, preserving original content.")

            # --- 1c. Process Tool Calls if Found (either from attribute or manual parse) --- 
            if tool_calls_to_execute:
                logger.debug(f"Tool calls to execute: {tool_calls_to_execute}")

                # Add reasoning to accumulated content (ensure it uses the updated current_response.content)
                # <<< MODIFICATION START: Safely extract text content >>>
                reasoning_text_parts = []
                raw_content = current_response.content
                if isinstance(raw_content, str):
                    # If already a string (e.g., after manual parsing), use it directly
                    reasoning_text_parts.append(raw_content)
                elif isinstance(raw_content, list):
                    # If it's a list, extract text from known structures
                    for item in raw_content:
                        if isinstance(item, str):
                            reasoning_text_parts.append(item)
                        elif isinstance(item, dict) and item.get('type') == 'text':
                            reasoning_text_parts.append(item.get('text', ''))
                        # Add elif for other potential dict structures if needed
                
                reasoning_content = " ".join(reasoning_text_parts).strip()
                # <<< MODIFICATION END >>>

                # --- NEW: Extract structured ACTION CONFIRMATION block for URL extraction ---
                # Look for ACTION CONFIRMATION block in the reasoning content
                url_from_confirmation = None
                if reasoning_content:
                    confirmation_match = re.search(r"ACTION CONFIRMATION:\s*Tool:\s*(\w+)\s*Parameters:\s*([\s\S]+?)END CONFIRMATION", reasoning_content)
                    if confirmation_match:
                        tool_name_from_confirmation = confirmation_match.group(1).strip()
                        params_text = confirmation_match.group(2).strip()
                        
                        # Check if this is a web_browser navigate_and_extract action
                        if tool_name_from_confirmation == "web_browser":
                            # Look for URL parameter in the confirmation block
                            url_match = re.search(r'-\s*url:\s*([^\n]+)', params_text)
                            if url_match:
                                url_from_confirmation = url_match.group(1).strip()
                                logger.info(f"Found URL in ACTION CONFIRMATION block: {url_from_confirmation}")
                            
                            # Check if action is navigate_and_extract
                            action_match = re.search(r'-\s*action:\s*([^\n]+)', params_text)
                            if action_match and action_match.group(1).strip() == "navigate_and_extract":
                                # If we have both navigate_and_extract action and a URL, ensure the tool call uses it
                                if url_from_confirmation and tool_calls_to_execute:
                                    for tool_call in tool_calls_to_execute:
                                        if tool_call.get("name") == "web_browser" and isinstance(tool_call.get("args"), dict):
                                            if tool_call["args"].get("action") == "navigate_and_extract" and not tool_call["args"].get("url"):
                                                # URL is missing in the tool call, add it from confirmation block
                                                tool_call["args"]["url"] = url_from_confirmation
                                                logger.info(f"Added missing URL to tool call from ACTION CONFIRMATION block: {url_from_confirmation}")
                # --- END NEW SECTION ---
                
                # Execute the tools 
                for tool_call in tool_calls_to_execute:
                    tool_name = tool_call.get("name")
                    tool_args = tool_call.get("args")
                    tool_call_id = tool_call.get("id")
                    base_tool_name = tool_name

                    # === IMPORTANT: Argument normalization steps ===
                    # 1. coerce_str_to_dict: Ensures all arguments are real dicts/lists (not stringified),
                    #    handling cases where LLMs or integrations output stringified JSON/Python objects.
                    # 2. normalize_tool_args: Ensures all argument keys match the tool's schema, mapping common
                    #    synonyms (like 'query' -> 'term'). This makes the agent robust to LLMs using generic keys.
                    #    Both are required for robust, schema-compliant tool invocation. Do not remove either.
                    tool_args = coerce_str_to_dict(tool_args)

                    tool_to_call = tool_map.get(tool_name)
                    if tool_to_call:
                        tool_args = normalize_tool_args(tool_args, tool_to_call)

                    # --- NEW: Additional URL validation for web_browser navigate_and_extract ---
                    if tool_name == "web_browser" and isinstance(tool_args, dict) and (tool_args.get("action") == "navigate_and_extract" or tool_args.get("action") == "extract"):
                        if not tool_args.get("url") and url_from_confirmation:
                            # We have a URL from confirmation but not in the arguments
                            tool_args["url"] = url_from_confirmation
                            logger.info(f"Added missing URL to web_browser navigate_and_extract from confirmation block: {url_from_confirmation}")
                        elif not tool_args.get("url"):
                            # No URL in args or confirmation, check if we can find one in the recent search results
                            logger.warning("No URL provided for web_browser navigate_and_extract. Looking for URL in recent search results.")
                            
                            # Look for a URL in the latest tool message history that might contain search results
                            for msg in reversed(history):
                                if isinstance(msg, ToolMessage) and "URL:" in msg.content:
                                    # Try to find a URL in the message using regex
                                    url_match = re.search(r'URL:\s*(https?://[^\s]+)', msg.content)
                                    if url_match:
                                        found_url = url_match.group(1).strip()
                                        logger.info(f"Found URL in recent search results: {found_url}")
                                        tool_args["url"] = found_url
                                        break
                                    
                                    # If regex failed, try to find URL in a table format
                                    if "| URL |" in msg.content or "| url |" in msg.content:
                                        table_url_match = re.search(r'\|\s*\d+\s*\|[^|]+\|\s*(https?://[^|\s]+)\s*\|', msg.content)
                                        if table_url_match:
                                            found_url = table_url_match.group(1).strip()
                                            logger.info(f"Found URL in table format from recent search results: {found_url}")
                                            tool_args["url"] = found_url
                                            break
                    # --- END NEW SECTION ---

                    # Prevent Immediate Retry
                    is_immediate_retry = False
                    if (last_failed_tool_info and
                        last_failed_tool_info["name"] == tool_name and
                        last_failed_tool_info["args"] == tool_args and
                        last_failed_tool_info["id"] == tool_call_id):
                        logger.warning(f"Immediate retry detected and PREVENTED for tool call ID: {tool_call_id}, Name: {tool_name}, Args: {tool_args}")
                        is_immediate_retry = True
                        tool_messages.append(
                            ToolMessage(content=last_failed_tool_info["error"], tool_call_id=tool_call_id)
                        )
                        # Don't increment consecutive_errors here, let the main check handle it
                    # End Retry Check

                    if not is_immediate_retry and tool_name in tool_map:
                        tool_to_call = tool_map[tool_name]
                        try:
                            # Preprocess the tool arguments before execution
                            preprocessed_args = self._preprocess_mcp_tool_args(tool_name, tool_args)
                            if preprocessed_args != tool_args:
                                logger.info(f"Preprocessed arguments for {tool_name}: {preprocessed_args}")
                                tool_args = preprocessed_args
                                
                            logger.info(f"Executing tool call: {tool_name} with args: {tool_args} (ID: {tool_call_id})")
                            
                            # --- Determine Function Identifier for Tracking/Limits --- #
                            function_identifier = None
                            if isinstance(tool_args, dict):
                                if tool_name == 'web_browser' and 'action' in tool_args:
                                    function_identifier = f"{tool_name}_{tool_args['action']}"
                                elif tool_name == 'reddit_search' and 'extract_result_index' in tool_args:
                                    function_identifier = f"{tool_name}_with_extract"
                                elif tool_name == 'reddit_search' and 'query' in tool_args:
                                    function_identifier = f"{tool_name}"
                                elif tool_name == 'reddit_extract_post':
                                    function_identifier = f"{tool_name}"
                                elif 'function' in tool_args and tool_args['function']:
                                    function_identifier = f"{tool_name}_{tool_args['function']}"
                            if not function_identifier:
                                function_identifier = tool_name # Fallback to tool name
                                logger.debug(f"Using generic tool name as function identifier for limits/tracking: {function_identifier}")
                            # --- End Function Identifier --- #
                            
                            # === Enforce Max Call Limit (using BASE tool total) ===
                            max_limit = self.tool_configs.get(base_tool_name, {}).get('max_calls', 3) # Default 3
                            # Get the *total* count for the base tool
                            current_total_count = processed_counts.get('base_tool_calls', {}).get(base_tool_name, 0)
                            
                            if current_total_count >= max_limit:
                                logger.warning(f"Skipping tool call {tool_name} ({function_identifier}) - Max call limit ({max_limit}) for base tool '{base_tool_name}' reached (Current Total: {current_total_count}).")
                                tool_messages.append(
                                    ToolMessage(content=f"Error: Max call limit ({max_limit}) reached for base tool '{base_tool_name}'. Cannot execute {function_identifier}.", tool_call_id=tool_call_id)
                                )
                                # Skip the rest of the try block for this tool call
                                continue 
                            # === End Limit Enforcement ===
                            
                            # <<< INCREMENT COUNTERS *BEFORE* EXECUTION >>>
                            # --- Increment base tool total count ONLY for content extraction actions --- 
                            is_content_extraction = (function_identifier == 'web_browser_navigate_and_extract' or function_identifier == 'web_browser_extract') # Add other extraction functions if needed
                            if is_content_extraction:
                                processed_counts['base_tool_calls'][base_tool_name] = current_total_count + 1
                                base_tool_increment_log = f" (Incremented Total: {processed_counts['base_tool_calls'][base_tool_name]})"
                            else:
                                base_tool_increment_log = f" (Total Unchanged: {current_total_count})" # Log that it wasn't incremented
                            # --- End Base Tool Increment Logic ---
                            
                            # Increment function-specific count
                            current_func_count = processed_counts.get('tool_functions', {}).get(function_identifier, 0)
                            if 'tool_functions' not in processed_counts:
                                processed_counts['tool_functions'] = {}
                            processed_counts['tool_functions'][function_identifier] = current_func_count + 1
                            logger.info(f"Tracked call: Base='{base_tool_name}'{base_tool_increment_log}, Function='{function_identifier}' (Count: {processed_counts['tool_functions'][function_identifier]}) Max Limit: {max_limit}")
                            # <<< END INCREMENT COUNTERS >>>

                            # Log the updated function counts for debugging
                            if 'tool_functions' in processed_counts:
                                logger.debug(f"Current function usage counts: {processed_counts['tool_functions']}")
                            
                            # Define base_tool_name as the first step inside the try block
                            # to ensure it's locally scoped and defined before await.
                            base_tool_name = tool_name
                            # <<< Store the raw tool output first >>>
                            try:
                                tool_output = await tool_to_call.arun(
                                    tool_input=tool_args,
                                    callbacks=run_config.get("callbacks")
                                )
                            except Exception as e:
                                logger.error(f"Error executing tool call {tool_name}: {e}", exc_info=True)
                                tool_output = f"Error executing tool call: {e}"

                            # --- Special handling for web_browser search results --- 
                            raw_tool_output = None
                            is_web_search_list_output = False
                            if tool_name == 'web_browser' and isinstance(tool_args, dict) and tool_args.get('action') == 'search' and isinstance(tool_output, list):
                                raw_tool_output = tool_output # Store the raw list
                                is_web_search_list_output = True
                                output_str = None # Don't use universal extractor for this case
                                logger.info(f"Received list output for web_browser search. Storing raw list.")
                            else:
                                # --- Use universal extractor for all other cases --- 
                                extracted_output = extract_mcp_content_universal(tool_output)
                                output_str = extracted_output if extracted_output is not None else ""

                            # Log and display the extracted output (truncated) or list type
                            preview_len = 1000
                            log_preview = ""
                            if is_web_search_list_output:
                                log_preview = f"List[{len(raw_tool_output)}] of search result dicts"
                                tool_content_for_history = f"Web Browser Search: Received {len(raw_tool_output)} search results as a list."
                            elif output_str:
                                log_preview = str(output_str)[:preview_len] + ("..." if len(str(output_str)) > preview_len else "")
                                # Don't use generic MCP Tool Output format for internal tools
                                if tool_name in INTERNAL_TOOLS:
                                    tool_content_for_history = f"{tool_name} Output:\n{log_preview}"
                                else:
                                    tool_content_for_history = f"MCP Tool Output ({tool_name}):\n{log_preview}"
                            else:
                                log_preview = "(Empty or None)"
                                tool_content_for_history = f"Warning: Tool '{tool_name}' returned empty output after extraction. Check MCP client/tool logs."
                                
                            logger.info(f"[MCP Tool Output] {tool_name}: {log_preview}")
                            # Don't log warning here if it was handled above
                            # if not output_str or not str(output_str).strip():
                            #     logger.warning(f"Tool '{tool_name}' returned empty output after extraction. Check MCP client/tool logs.")
                            #     tool_content_for_history = f"Warning: Tool '{tool_name}' returned empty output after extraction."
                            # else:
                            #     tool_content_for_history = f"MCP Tool Output ({tool_name}):\n{truncated_output}"

                            # --- Accumulate FULL content --- (Modify existing blocks)
                            extracted_data = None
                            if isinstance(output_str, str) and output_str.strip().startswith('{') and output_str.strip().endswith('}'):
                                try:
                                    extracted_data = json.loads(output_str)
                                except json.JSONDecodeError:
                                    extracted_data = None # Not valid JSON
                            
                            # Update flags (check tool_name and action again for clarity)
                            is_web_extraction = (tool_name == 'web_browser' and isinstance(tool_args, dict) and 
                               (tool_args.get('action') == 'navigate_and_extract' or tool_args.get('action') == 'extract'))
                            is_reddit_extraction = (
                                (tool_name == 'reddit_extract_post') or 
                                (tool_name == 'reddit_search' and isinstance(tool_args, dict) and tool_args.get("extract_result_index") is not None)
                            )
                            is_reddit_search_list = (tool_name == 'reddit_search' and not is_reddit_extraction) # Only if NOT extracting
                            is_web_search_list_output = is_web_search_list_output # Keep flag as set earlier

                            processed_successfully = False
                            content_added_this_call = False # Track if content relevant for condensation was added

                            # --- Generic MCP Tool or Internal Tool Processing --- #
                            if tool_name in INTERNAL_TOOLS:
                                # --- Process Internal Tools (web_browser, reddit) --- #
                                
                                # --- Process Reddit Extraction --- #
                                # Use output_str which contains the universally extracted content for reddit
                                if is_reddit_extraction and extracted_data and isinstance(extracted_data, dict):
                                    logger.info(f"Processing extracted Reddit content from tool: {tool_name}")
                                    # Determine URL: Use tool_args if available (from reddit_extract_post)
                                    # or extracted_data['url'] (from reddit_search with extract)
                                    if tool_name == 'reddit_extract_post' and isinstance(tool_args, dict):
                                        post_url = tool_args.get('post_url', 'Unknown URL')
                                    else: # Assume reddit_search with extract
                                        post_url = extracted_data.get('url', 'Unknown URL')
                                        
                                    post_content = extracted_data.get('post', '') or "" 
                                    comments_content = extracted_data.get('post_comments', '') or ""
                                    full_reddit_content = f"<h1>Post:</h1>\n{post_content}\n\n<h1>Comments:</h1>\n{comments_content}".strip()
                                    source_desc = f"Reddit Post: {post_url}"

                                    if not full_reddit_content or full_reddit_content == "<h1>Post:</h1>\n\n<h1>Comments:</h1>":
                                        logger.warning(f"Skipping summarization for {source_desc} as extracted content is empty or errored.")
                                        accumulated_content += f"\n\n--- Skipped Empty/Errored Reddit Post: {post_url} ---\n"
                                    else:
                                        try:
                                            # Store content with proper source attribution
                                            reddit_content_data = {
                                                "full_content": full_reddit_content,
                                                "title": f"Reddit content from {post_url}",
                                                "tool_name": tool_name,
                                                "tool_args": tool_args
                                            }
                                            # Store content with proper source type
                                            if post_url and post_url != "Unknown URL":
                                                self.content_manager.store_content(post_url, reddit_content_data, source_type="reddit")
                                                
                                            logger.info(f"Generating summary for {source_desc}")
                                            summary = await self.content_manager.get_summary(post_url, content=full_reddit_content, callbacks=self.callbacks)
                                            
                                            # Mark this URL as used in the summary
                                            if post_url and post_url != "Unknown URL":
                                                self.content_manager.mark_content_used_in_summary(post_url)
                                                
                                            # --- MODIFIED CONTENT ACCUMULATION ---
                                            # Store both summary and full content for the final report
                                            full_content_to_add = full_reddit_content # Specific for Reddit
                                            accumulated_content += (
                                                f"\n\n--- BEGIN PROCESSED CONTENT from {source_desc} ---\n"
                                                f"--- Full Content ---\n"
                                                f"{full_content_to_add}\n"
                                                f"--- END PROCESSED CONTENT from {source_desc} ---\n\n"
                                            )
                                            # --- END MODIFICATION ---
                                            logger.info(f"Added summary and full content for {source_desc} to accumulated_content (Summary length: {len(summary)}, Full length: {len(full_content_to_add)})")
                                            processed_successfully = True
                                            content_added_this_call = True # Summary was added
                                        except Exception as e:
                                            logger.error(f"Failed to summarize content for {source_desc}: {e}", exc_info=True)
                                            truncated_output = output_str[:10000] + "... [Content truncated due to summarization error]" if len(output_str) > 10000 else output_str
                                            # --- MODIFIED ERROR ACCUMULATION ---
                                            # Store truncated raw content when summarization fails
                                            full_content_to_add = output_str # Specific for web extraction
                                            # Use the already truncated version for the summary part in case of error
                                            summary_fallback = truncated_output
                                            accumulated_content += (
                                                f"\n\n--- BEGIN FAILED-SUMMARY CONTENT from {source_desc} ---\n"
                                                f"--- Full Content ---\n"
                                                f"{full_content_to_add}\n"
                                                f"--- END FAILED-SUMMARY CONTENT from {source_desc} ---\n\n"
                                            )
                                            # --- END MODIFICATION ---
                                            logger.warning(f"Using truncated raw content as summary fallback, but stored full content for final report for {source_desc}")
                                            processed_successfully = False # Summarization failed, treat as not fully processed
                                
                                # --- Process Reddit Search List --- #
                                elif is_reddit_search_list and isinstance(output_str, str) and "Searched Reddit for" in output_str:
                                    # Pass the #main-content HTML/text directly to the summarizer for table extraction.
                                    try:
                                        table_md = await self.content_manager.summarize_search_results_as_table(
                                            output_str,  # This should be the HTML/text of #main-content
                                            top_n=5,
                                            callbacks=self.callbacks,
                                            prompt_instructions="Extract a markdown table of Reddit search results with columns: Title, URL, Subreddit, Comment Count."
                                        )
                                        header = f"--- Reddit Search Results Table ---"
                                        accumulated_content += f"\n\n{header}\n{table_md}\n--- End Reddit Search Results Table ---\n"
                                        tool_content_for_history = table_md
                                        logger.info("Added markdown table of Reddit search results to accumulated_content and scratchpad (via summarizer).")
                                    except Exception as e:
                                        accumulated_content += f"\n\n--- Reddit Search Results (raw) ---\n{output_str}\n--- End Reddit Search Results ---\n"
                                        tool_content_for_history = output_str
                                        logger.warning(f"Could not summarize Reddit search results, added raw output instead: {e}")

                                # --- Process Web Search List --- #
                                # Use the flag set earlier to identify this case
                                elif is_web_search_list_output:
                                    # We already stored the raw list in raw_tool_output 
                                    
                                    # --- Process the list directly --- 
                                    table_rows = []
                                    try:
                                        # Use raw_tool_output which is guaranteed to be a list here
                                        logger.info(f"Processing web search results list with {len(raw_tool_output)} results.") 
                                        results_list = raw_tool_output 
                                        if not results_list:
                                            logger.warning("Search results list is empty.")
                                            tool_content_for_history = "Success: Retrieved search results, but the list was empty."
                                        else:
                                            # --- Generate Markdown Table --- 
                                            table_header = "| # | Title (URL) | Snippet |\n|---|---|---|"
                                            for i, result_data in enumerate(results_list):
                                                title = result_data.get('title', '') or '' # Default to empty string if None
                                                link = result_data.get('link', '') or ''   # Default to empty string if None
                                                snippet = result_data.get('snippet', '') or '' # Default to empty string if None
                                                # Escape pipes for Markdown
                                                title_md = title.replace('|', '\\|')
                                                link_md = link.replace('|', '\\|')
                                                # Limit snippet length in table for readability
                                                snippet_md = snippet.replace('|', '\\|')[:200] + ("..." if len(snippet) > 200 else "")
                                                table_rows.append(f"| {i+1} | [{title_md}]({link_md}) | {snippet_md} |")
                                            
                                            # Create the markdown table with proper newlines
                                            markdown_table = table_header + "\n" + "\n".join(table_rows)
                                            
                                            # Include the table directly in the history message content
                                            tool_content_for_history = (
                                                f"Success: Retrieved and processed {len(results_list)} search results for "
                                                f"'{tool_args.get('query', 'N/A')}':\n\n"
                                                f"{markdown_table}"
                                            )
                                            # <<< ADD DEBUG LOG 1 >>>
                                            logger.debug(f"[DEBUG] tool_content_for_history AFTER table assignment (first 300 chars):\n{str(tool_content_for_history)[:300]}")
                                            
                                            # Add detailed results to accumulated content
                                            # accumulated_content += f"..." # Removed line
                                            logger.info(f"Processed markdown table with {len(results_list)} search results (content NOT added to accumulated_content).") # Modified log message
                                            # content_added_this_call = True # Search results should not trigger condensation
                                    
                                    except Exception as e:
                                         # Keep the generic error handling for parsing the list itself
                                         logger.error(f"Unexpected error processing web search results list: {e}", exc_info=_log_traceback)
                                         tool_content_for_history = f"Error: Failed to process web search results list: {e}"
                                else:
                                    # Default case for other internal tools we don't have special handling for
                                    if output_str:
                                        accumulated_content += f"\n\n--- Output from {tool_name} ---\n{output_str}\n--- End Output ---\n"
                                        tool_content_for_history = f"Success: {tool_name} executed. [Output length: {len(str(output_str))}]"
                                        content_added_this_call = True
                                    else:
                                        tool_content_for_history = f"Warning: {tool_name} returned empty output after extraction."
                            else:
                                # --- Process Generic MCP Tool Output --- #
                                logger.info(f"Processing generic MCP tool output for {tool_name}")
                                
                                # Extract function name from tool if available
                                function_name = None
                                if isinstance(tool_args, dict) and 'action' in tool_args:
                                    function_name = tool_args['action']
                                elif isinstance(tool_args, dict) and 'function' in tool_args:
                                    function_name = tool_args['function']
                                
                                # Store content with tool name, function name, and parameters
                                mcp_content_data = {
                                    "full_content": output_str,
                                    "title": f"MCP Tool Output from {tool_name}",
                                    "tool_name": tool_name,
                                    "function_name": function_name,
                                    "tool_args": tool_args
                                }
                                # Use a synthetic URL or identifier if no URL is present
                                mcp_content_id = f"mcp_{tool_name}_{function_name or 'none'}_{tool_call_id}"
                                self.content_manager.store_content(mcp_content_id, mcp_content_data, source_type=tool_name)

                            # Append the final determined history message
                            # <<< ADD DEBUG LOG 2 >>>
                            logger.debug(f"[DEBUG] tool_content_for_history BEFORE ToolMessage creation (first 300 chars):\n{str(tool_content_for_history)[:300]}")
                            tool_messages.append(
                                ToolMessage(content=tool_content_for_history, tool_call_id=tool_call_id)
                            )
                            last_failed_tool_info = {k: v for k, v in last_failed_tool_info.items() if v.get('id') != tool_call_id} if last_failed_tool_info else None

                            # Check if condensation is needed after this successful add
                            if content_added_this_call:
                                content_added_since_last_condense += 1
                                if content_added_since_last_condense >= condense_frequency:
                                    needs_condensation = True # Trigger after loop

                        except McpError as mcp_exc:
                            error_summary = f"MCP Tool Error: {tool_name} failed. Reason: {str(mcp_exc)}"[:500]
                            # Check for specific invalid arguments error code
                            # <<< ADD LOGGING TO CHECK CORRECTION TRIGGER >>>
                            logger.info(f"Caught McpError for {tool_name}. Code: {mcp_exc.args[0].get('code') if mcp_exc.args and isinstance(mcp_exc.args[0], dict) else 'N/A'}. Checking if it's -32602...")
                            # Check for specific invalid arguments error code
                            is_invalid_args_error = False
                            if mcp_exc.args and isinstance(mcp_exc.args[0], dict) and mcp_exc.args[0].get('code') == -32602:
                                is_invalid_args_error = True
                                logger.info(f"Detected search_abstracts validation error. Attempting specialized correction.")
                            
                            if is_invalid_args_error:
                                logger.warning(f"Detected invalid arguments error for {tool_name}. Attempting correction.")
                                try:
                                    # First try direct correction without LLM for efficiency
                                    direct_correction = self._try_direct_correction(
                                        tool_name=tool_name, 
                                        failed_args=tool_args,
                                        error_message=str(mcp_exc)
                                    )
                                    
                                    if direct_correction:
                                        logger.info(f"Direct correction applied for {tool_name}: {direct_correction}")
                                        corrected_args_json = direct_correction
                                    else:
                                        # If direct correction failed, try LLM-based correction
                                        corrected_args_json = await self._get_tool_correction_suggestion(
                                            tool_name=tool_name, 
                                            failed_args=tool_args,
                                            error_message=str(mcp_exc),
                                            run_config=run_config
                                        )
                                    
                                    if corrected_args_json:
                                        error_summary += f"\n\n[Correction Suggestion]:\n```json\n{json.dumps(corrected_args_json, indent=2)}\n```"
                                        logger.info(f"Successfully added correction suggestion for {tool_name} to error message.")
                                        
                                        # Store the fact that this tool was corrected once
                                        if not hasattr(self, '_corrected_tools'):
                                            self._corrected_tools = set()
                                        self._corrected_tools.add(tool_name)
                                    else:
                                        # If no correction could be generated but tool has been corrected before,
                                        # treat this as "no content" instead of an error
                                        if hasattr(self, '_corrected_tools') and tool_name in self._corrected_tools:
                                            logger.warning(f"Tool {tool_name} already had one correction attempt. Treating as no content.")
                                            error_summary = f"No content available from {tool_name}. Moving on to the next research step."
                                            # Don't increment consecutive errors for this case
                                            tool_messages.append(
                                                ToolMessage(content=error_summary, tool_call_id=tool_call_id)
                                            )
                                            continue
                                        else:
                                            logger.warning(f"No valid correction suggestion generated for {tool_name}.")
                                except Exception as correction_err:
                                    logger.error(f"Error during correction process for {tool_name}: {correction_err}", exc_info=True)
                                    # error_summary remains the original error
                            else:
                                # Log non-argument MCP errors normally
                                logger.error(f"MCP Tool Error: {error_summary}", exc_info=False)
                                
                            # Append the (potentially enhanced) error summary
                            tool_messages.append(
                                ToolMessage(content=error_summary, tool_call_id=tool_call_id)
                            )
                            last_failed_tool_info = {"name": tool_name, "args": tool_args, "id": tool_call_id, "error": error_summary}
                            
                        except Exception as tool_exc:
                            # --- Summarize Error for Scratchpad --- 
                            error_summary = f"Tool Execution Error: {tool_name} failed. Reason: {str(tool_exc)}"[:500]
                            logger.error(error_summary, exc_info=_log_traceback)
                            tool_messages.append(
                                ToolMessage(content=error_summary, tool_call_id=tool_call_id)
                            )
                            last_failed_tool_info = {"name": tool_name, "args": tool_args, "id": tool_call_id, "error": error_summary}

                    elif not is_immediate_retry:
                        logger.warning(f"Tool '{tool_name}' requested but not found in available tools.")
                        tool_messages.append(
                            ToolMessage(content=f"Error: Tool '{tool_name}' not found.", tool_call_id=tool_call_id)
                        )

                # Add all tool messages (results or errors) to history AFTER processing all calls
                if tool_messages:
                    history.extend(tool_messages)
                    logger.debug(f"History after adding tool messages (length {len(history)}): {[msg.pretty_repr() for msg in history[-len(tool_messages):]]}")

                # Update consecutive errors based on the results of *this batch* of tool calls
                if all(msg.content.startswith("Error") for msg in tool_messages if msg.content): # Check if all non-empty results were errors
                     consecutive_errors += 1
                else:
                     consecutive_errors = 0 # Reset if at least one tool succeeded

            else: # No tool calls in the current_response
                logger.info("No tool calls requested by the LLM in the previous step.")
                # Add the LLM's thought/response to history if it wasn't a tool call message
                # and wasn't the initial planner response (already added)
                if current_response not in history[-2:]: # Check if it's already the last AI message
                    if isinstance(current_response, AIMessage): # Ensure it's an AI message
                         history.append(current_response)
                         logger.info(f"Added non-tool AI response to history.")
                         if current_response.content:
                             logger.info(f"LLM thought/response (preview): {current_response.content[:200]}...")
                             logger.debug(f"Full LLM thought/response: {current_response.content}")

                # Check termination condition: Stop if minimum results met and LLM didn't request tools
                total_processed = self._calculate_total_processed(processed_counts) # Use helper
                if total_processed >= MIN_REGULAR_WEB_PAGES:
                    logger.info(f"Minimum results ({MIN_REGULAR_WEB_PAGES}) processed and no further tool calls requested. Moving to summary.")
                    break
                # Or stop if it's past the first iteration and no tools were called (LLM decided to stop early)
                if iteration > 0 and not executed_tool_call_this_iter:
                    logger.warning(f"LLM finished without tool calls before min results ({MIN_REGULAR_WEB_PAGES}) processed. Forcing loop break.")
                    break

            # --- Perform Condensation if Triggered ---
            if needs_condensation:
                logger.info(f"Condensation triggered after {content_added_since_last_condense} content additions.")
                try:
                    # Split content into "previous content" and "most recent content"
                    # This assumes content is being appended with newlines between sections
                    content_parts = accumulated_content.split("\n\n")
                    
                    if len(content_parts) > 1:
                        # Get the most recent content section (the last part)
                        most_recent_content = content_parts[-1]
                        # Get all previous content (everything except the last part)
                        previous_content = "\n\n".join(content_parts[:-1])
                        
                        logger.info(f"Separating content: Previous content size: {len(previous_content)} chars, Most recent: {len(most_recent_content)} chars")
                        
                        # Only condense if there's enough previous content to work with
                        if len(previous_content) > 200:
                            # Condense only the previous content
                            condense_input = {"text": previous_content}
                            condensed_response = await self.condensation_chain.ainvoke(condense_input, config=run_config)
                            condensed_previous_content = condensed_response.content
                            
                            # Combine condensed previous content with most recent content
                            condensed_content_for_prompt = f"{condensed_previous_content}\n\n# Most Recent Content:\n\n{most_recent_content}"
                            
                            logger.info(f"Successfully condensed previous content. Previous content condensed from {len(previous_content)} to {len(condensed_previous_content)} chars. Combined length with recent content: {len(condensed_content_for_prompt)}")
                        else:
                            # If previous content is too small, just use the full accumulated content as is
                            logger.info(f"Previous content too small ({len(previous_content)} chars). Using full content without condensation.")
                            condensed_content_for_prompt = accumulated_content
                    else:
                        # If content can't be split (only one part), just condense it all
                        logger.info(f"Content cannot be separated. Condensing entire content ({len(accumulated_content)} chars).")
                        condense_input = {"text": accumulated_content}
                        condensed_response = await self.condensation_chain.ainvoke(condense_input, config=run_config)
                        condensed_content_for_prompt = condensed_response.content
                        logger.info(f"Successfully condensed entire content. New condensed length: {len(condensed_content_for_prompt)}")
                    
                    # Reset counter as condensation was performed
                    content_added_since_last_condense = 0
                except Exception as condense_err:
                    logger.error(f"Error during iterative condensation: {condense_err}. Using previous condensed content.", exc_info=_log_traceback)
                    # Keep previous condensed_content_for_prompt in case of error

            # --- 2. Check Termination Conditions ---
            total_processed = self._calculate_total_processed(processed_counts) # Use helper
            if consecutive_errors >= 3:
                logger.warning("Multiple consecutive errors (>=3) encountered. Continuing research but noting the errors.")
                # Add a message to the history indicating the errors, but don't abort
                error_msg = AIMessage(content=f"I've encountered {consecutive_errors} consecutive errors. I'll try a different approach.")
                history.append(error_msg)
                consecutive_errors = 0  # Reset the counter to allow continuing
            if total_processed >= effective_max_results:
                logger.info(f"Effective maximum results ({effective_max_results}) processed based on plan. Moving to summary.")
                break
            if iteration + 1 == max_iterations: # Check if NEXT iteration would exceed max (max_iterations is already based on effective_max_results)
                 logger.warning(f"Reached max iterations ({max_iterations}). Moving to summary.")
                 break

            # --- 3. Prepare Input and Call LLM for the *Next* Action ---
            
            # Format agent_scratchpad from recent history
            scratchpad_messages = []
            num_history = len(history)
            if num_history > 0 and isinstance(history[-1], ToolMessage):
                # Find the preceding AI message with tool calls
                for i in range(num_history - 2, -1, -1):
                    if isinstance(history[i], AIMessage) and history[i].tool_calls:
                        scratchpad_messages = history[i:] # Get AI message + all subsequent Tool messages
                        break
                if not scratchpad_messages and num_history >= 1: # Fallback if only ToolMessage found
                     scratchpad_messages = [history[-1]]
            elif num_history > 0 and isinstance(history[-1], AIMessage): # Last message was AI (no tool call or before tool execution)
                 scratchpad_messages = [history[-1]]
            elif num_history > 0: # Last message was Human
                 scratchpad_messages = [] # No AI/Tool interaction to put in scratchpad yet

            agent_scratchpad_str = "\n".join([msg.pretty_repr() for msg in scratchpad_messages])

            # Format tools list
            tools_list_str = "\n".join([f"- {tool.name}: {tool.description}" for tool in self.tools])

            # Format tool metadata (includes min/max calls)
            
            # === Construct other_processed_details string ===
            other_details_lines = []
            # Map internal keys to user-facing names if needed
            tool_type_map = { 
                'regular_web_pages': 'Web Pages',
                'reddit_posts': 'Reddit Posts',
                # Function-specific naming
                'web_browser_navigate_and_extract': 'Web Content Extraction',
                'web_browser_search': 'Web Search',
                'reddit_search': 'Reddit Search',
                'reddit_search_with_extract': 'Reddit Extract via Search',
                'reddit_extract_post': 'Reddit Post Extraction'
                # Note: Transcript tool mapping moved to dynamic handling via 'tool_functions' tracking
                # rather than hardcoding specific MCP tools like 'get_transcript'
            }
            # First add standard tools
            for key, count in processed_counts.items():
                # Check if count is an integer and greater than 0 before processing
                if key not in ['mcp_tools', 'tool_functions'] and isinstance(count, int) and count > 0: 
                    tool_display_name = tool_type_map.get(key, key.capitalize()) # Use map or capitalize key
                    other_details_lines.append(f"- {tool_display_name}: {count}")
            
            # Then add MCP tools if present
            if 'mcp_tools' in processed_counts and processed_counts['mcp_tools']:
                # Add a header for MCP tools section if there are any
                for mcp_tool_name, mcp_count in processed_counts['mcp_tools'].items():
                    if mcp_count > 0:
                        other_details_lines.append(f"- MCP Tool '{mcp_tool_name}': {mcp_count}")
            
            # Add function-specific counts (separated by a blank line for clarity)
            if 'tool_functions' in processed_counts and processed_counts['tool_functions']:
                other_details_lines.append("")
                other_details_lines.append("Function-Specific Usage:")
                # Sort by function name to ensure consistent order
                for func_name, func_count in sorted(processed_counts['tool_functions'].items()):
                    if func_count > 0:
                        # Use the mapping if available, otherwise use the raw function name
                        display_name = tool_type_map.get(func_name, func_name)
                        other_details_lines.append(f"- {display_name}: {func_count}")
            
            # Add base tool totals
            if processed_counts.get('base_tool_calls'): # Safely get base_tool_calls
                other_details_lines.append("")
                other_details_lines.append("Base Tool Totals:")
                for base_tool, total_count in sorted(processed_counts['base_tool_calls'].items()):
                    if total_count > 0:
                         other_details_lines.append(f"- {base_tool.capitalize()}: {total_count}")
            
            other_processed_details_str = "\n".join(other_details_lines)
            # === End Construct ===

            # === Update action_input ===
            # Calculate total processed using helper method
            total_processed = self._calculate_total_processed(processed_counts)

            # Build new context fields
            summary_so_far = condensed_content_for_prompt
            sources_visited = self._build_sources_visited(processed_counts, history)
            last_action = self._build_last_action(history)

            # Filter history to exclude resolved errors for LLM input
            filtered_history = self._filter_history_for_llm(history)

            # Generate tool usage tracker markdown table
            tool_usage_tracker_md = self._generate_tool_usage_tracker_md(processed_counts)

            action_input = {
                "topic": topic,
                "current_date": current_date,
                "summary_so_far": summary_so_far,
                "sources_visited": sources_visited,
                "last_action": last_action,
                "history": filtered_history, # <<< Use filtered history for LLM
                "regular_web_pages_processed_count": processed_counts.get('regular_web_pages', 0),
                "reddit_posts_processed_count": processed_counts.get('reddit_posts', 0),
                "other_processed_details": other_processed_details_str,
                "results_processed": total_processed,
                "recent_errors": [msg.content for msg in history[-5:] if isinstance(msg, ToolMessage) and msg.content.startswith("Error")],
                "invalid_extraction_urls": list(invalid_extraction_urls),
                "planned_tool_limits": getattr(self, 'planned_tool_limits', {}), # <-- Add planned tool limits dict
                "tool_usage_tracker_md": tool_usage_tracker_md, # <-- Add generated markdown table
            }
            # === End Update action_input ===

            # --- Debug Logging for Next Call ---
            logger.debug(f"--- Preparing for LLM action call in iteration {iteration + 1} ---")
            logger.debug(f"Action Input Keys: {list(action_input.keys())}")
            logger.debug(f"  summary_so_far: {summary_so_far[:200]}")
            logger.debug(f"  sources_visited: {sources_visited}")
            logger.debug(f"  last_action: {last_action}")
            logger.debug(f"  regular_web_pages_processed_count: {action_input['regular_web_pages_processed_count']}")
            logger.debug(f"  reddit_posts_processed_count: {action_input['reddit_posts_processed_count']}")
            logger.debug(f"  other_processed_details: '{action_input['other_processed_details']}'")
            logger.debug(f"  results_processed (total): {action_input['results_processed']}")
            history_tail = history[-5:]
            logger.debug(f"History tail for LLM invoke (length {len(history)}): {[msg.pretty_repr() for msg in history_tail]}")

            # <<< START FIX: Structure final error ToolMessage content for Gemini >>>
            history_for_llm_call = action_input['history'] # Get the history intended for the LLM
            if (history_for_llm_call and
                isinstance(history_for_llm_call[-1], ToolMessage) and
                isinstance(history_for_llm_call[-1].content, str) and
                history_for_llm_call[-1].content.startswith("Error:")):
                
                try:
                    # Create a copy to avoid modifying the original history object directly
                    # if modifying in place causes issues elsewhere. If not, modify directly.
                    # For safety, let's create a deep copy for this specific call.
                    history_for_llm_call = copy.deepcopy(history_for_llm_call)
                    
                    original_error_content = history_for_llm_call[-1].content
                    # Wrap the error string in a simple JSON object
                    structured_error = json.dumps({"error": original_error_content})
                    history_for_llm_call[-1].content = structured_error
                    logger.info(f"Structured the final error ToolMessage content for Gemini API call: {structured_error}")
                    # Update the history in action_input for *this specific invoke call*
                    action_input['history'] = history_for_llm_call
                except Exception as e:
                    logger.error(f"Failed to structure final error ToolMessage content: {e}", exc_info=True)
                    # Proceed with the potentially problematic history if structuring fails
            # <<< END FIX >>>

            try:
                # Remove MEMORY_KEY and agent_scratchpad from LLM input
                logger.info(f"Invoking action_iteration_chain for iteration {iteration + 2}...")
                # Use the potentially modified action_input['history']
                next_response: AIMessage = await self.action_iteration_chain.ainvoke(
                    action_input, # Contains the potentially modified history
                    config=run_config
                )
                logger.debug(f"Action Iteration Raw Response: {next_response}")
                if hasattr(next_response, 'response_metadata') and next_response.response_metadata:
                    logger.info("[LLM API DEBUG] Response metadata: %s", next_response.response_metadata)
                # --- Reflection Logging ---
                reflection_log = {
                    'iteration': iteration + 1,
                    'action_input': action_input,
                    'llm_response': str(next_response),
                    'chosen_action': getattr(next_response, 'tool_calls', None),
                    'reasoning': getattr(next_response, 'content', None),
                }
                logger.debug(f"[REFLECTION] {reflection_log}")
                
                # === BEGIN ADDED LOGGING ===
                is_claude = "anthropic" in str(type(self.llm_client)).lower()
                log_prefix = "[Claude Response Check]" if is_claude else "[Gemini Response Check]"
                logger.info(f"{log_prefix} Raw Response Content Type: {type(next_response.content)}")
                logger.info(f"{log_prefix} Raw Response Content: {next_response.content}")
                logger.info(f"{log_prefix} Raw Response tool_calls Type: {type(next_response.tool_calls)}")
                logger.info(f"{log_prefix} Raw Response tool_calls: {next_response.tool_calls}")
                
                # === NEW STRUCTURED CONFIRMATION CHECK ===
                # Check if we have structured tool_calls but no actual tool calls
                if not next_response.tool_calls and isinstance(next_response.content, str) and "tool" in next_response.content.lower():
                    # First check for our structured confirmation format
                    tool_name, tool_params = self._extract_structured_confirmation(next_response.content)
                    
                    if tool_name:
                        logger.info(f"No tool_calls found, but structured confirmation detected for tool: {tool_name}")
                        # Create a synthetic tool call from the structured confirmation
                        tool_call_id = f"structured_confirmation_{iteration}"
                        synthetic_tool_call = {
                            'name': tool_name,
                            'args': tool_params,
                            'id': tool_call_id
                        }
                        
                        # Set the tool_calls on the next_response
                        next_response = AIMessage(
                            content=next_response.content,
                            tool_calls=[synthetic_tool_call],
                            response_metadata=next_response.response_metadata,
                            id=next_response.id
                        )
                        
                        logger.info(f"Created synthetic tool call for {tool_name} with params: {tool_params}")
                    elif is_claude:
                        # Special handling for Claude which sometimes mentions tools without proper tool_calls
                        logger.warning(f"{log_prefix} Response has text mentioning a tool but no structured tool_calls attribute or confirmation block.")
                    else:
                        # For Gemini models that indicate a tool should be used but don't provide either
                        logger.warning(f"{log_prefix} Response indicates tool use but provides neither tool_calls nor structured confirmation block.")
                        
                        # Explicit fallback for case where LLM indicates tool use but doesn't provide structured data
                        # We could add a clarification request here if needed in the future
                # === END NEW STRUCTURED CONFIRMATION CHECK ===
                
                # Assign the response to current_response for the *next* loop iteration
                current_response = next_response
                # Add the AI response (which might contain tool calls) to history for the next cycle
                history.append(current_response)
                logger.debug(f"Added AI response for next iteration to history (length {len(history)}): {current_response.pretty_repr()}")

            except Exception as e:
                # Enhanced error logging
                error_type = type(e).__name__
                error_msg = str(e)
                
                # Check if it's a 500 error
                is_server_error = False
                if "500" in error_msg or "Internal Server" in error_msg:
                    is_server_error = True
                    logger.error(f"[LLM API 500 ERROR] Server error detected during iteration {iteration + 1}")
                    logger.error(f"[LLM API 500 ERROR] Error details: {error_type} - {error_msg}")
                    logger.error(f"[LLM API 500 ERROR] Model: {self.model_name}, History length: {len(history) - 1} messages")
                    logger.error(f"[LLM API 500 ERROR] Last few history messages: {[type(msg).__name__ for msg in history[-3:] if history[-3:] != [None, None, None] and history[-3:] != [None, None, None, None]]}")
                else:
                    # Standard error logging
                    logger.error(f"Error during action iteration LLM call {iteration + 1}: {error_type} - {error_msg}", exc_info=_log_traceback)
                
                # Store accumulated content for later access before failing
                self._current_accumulated_content = accumulated_content
                
                # Stop processing immediately if the LLM call itself fails
                return f"Research failed due to error in LLM action call: {error_msg}"

        # --- Loop Finished ---

        # --- Phase 4: Final Summary --- 
        logger.info("Phase 4: Generating Final Summary...")
        self.current_stage = "summarization" # For thinking logic
        try:
            # Prepare final content for summary
            total_processed = self._calculate_total_processed(processed_counts) # Use helper method instead of sum
            # --- Debug: Log what is being passed to the summarizer ---
            logger.debug(f"Accumulated Content for Summary (first 1000 chars):\n{accumulated_content[:1000]}\n--- END PREVIEW ---")
            # --- Warn if condensation is too aggressive ---
            if len(accumulated_content) > 1000 and len(condensed_content_for_prompt) > 1000:
                ratio = len(condensed_content_for_prompt) / len(accumulated_content) if accumulated_content else 0 # Avoid division by zero
                if ratio < 0.3:
                    logger.warning(f"Condensed content is less than 30% the length of accumulated_content! (Condensed: {len(condensed_content_for_prompt)}, Accumulated: {len(accumulated_content)}, Ratio: {ratio:.2f})")
            # --- End condensation warning ---
            
            # Store accumulated content for later access
            self._current_accumulated_content = accumulated_content
            
            return accumulated_content # Return the full content for final summary
        except Exception as e:
            logger.error(f"Error preparing final summary data: {e}", exc_info=True)
            # Add basic error information to the accumulated content
            error_msg = f"\n\n### ERROR DURING SUMMARY PREPARATION\n\nAn error occurred while preparing the final summary data: {str(e)}\n\n"
            # <<< MODIFICATION: Return accumulated content even on error >>>
            self._current_accumulated_content = accumulated_content + error_msg
            return accumulated_content + error_msg
        finally:
            # Log total time
            end_time = datetime.now()
            if start_time is not None:
                logger.info(f"Research process finished in {end_time - start_time}")
            else:
                logger.info(f"Research process finished at {end_time} (start_time unknown)")

    def _sanitize_history_for_gemini(self, history: List[BaseMessage], topic: str) -> List[BaseMessage]:
        """Sanitizes the message history to ensure it follows Gemini's requirements:
        1. First message must be a HumanMessage
        2. Function calls must be immediately followed by function responses
        3. No consecutive function calls or responses
        4. No empty message content parts (to prevent Gemini's "contents.parts must not be empty" error)
        """
        # Debug: Log the incoming messages to identify potential problematic messages
        logger.warning(f"Sanitizing history with {len(history)} messages for Gemini API")
        for i, msg in enumerate(history):
            content_type = type(msg.content).__name__
            content_preview = str(msg.content)[:50] + "..." if isinstance(msg.content, str) and len(str(msg.content)) > 50 else msg.content
            logger.warning(f"Message {i}: {type(msg).__name__} with content type {content_type}, preview: {content_preview}")
            
            # Extra check for list content
            if isinstance(msg.content, list):
                for j, part in enumerate(msg.content):
                    part_type = type(part).__name__
                    part_preview = str(part)[:30] + "..." if isinstance(part, str) and len(str(part)) > 30 else part
                    logger.warning(f"  Part {j}: type={part_type}, preview={part_preview}")
        
        if not history:
            # If history is empty, just return a HumanMessage with the topic
            return [HumanMessage(content=f"Research the topic: {topic}")]
            
        # Enhanced safety: pre-filter to remove ANY messages with empty content
        filtered_history = []
        for i, msg in enumerate(history):
            # Skip any message with empty content - very aggressive filtering
            if self._has_empty_content(msg):
                logger.warning(f"Pre-filtering: Removing message at index {i} with empty content")
                continue
                
            # For list content, filter out empty parts
            if isinstance(msg.content, list):
                valid_parts = []
                for part in msg.content:
                    # Skip None or empty strings or empty dicts
                    if part is None or (isinstance(part, str) and not part.strip()) or (isinstance(part, dict) and not part):
                        continue
                    valid_parts.append(part)
                    
                # If we removed parts, create a new message
                if len(valid_parts) != len(msg.content):
                    logger.warning(f"Pre-filtering: Removed {len(msg.content) - len(valid_parts)} empty parts from message at index {i}")
                    if valid_parts:  # Only add if we have valid parts
                        # Create a new message of the same type with filtered content
                        new_msg = type(msg)(
                            content=valid_parts,
                            tool_calls=getattr(msg, 'tool_calls', None)
                        )
                        filtered_history.append(new_msg)
                else:
                    # No empty parts, add original message
                    filtered_history.append(msg)
            else:
                # Not a list, add original message
                filtered_history.append(msg)
            
        # If we've filtered out all messages, return a fallback
        if not filtered_history:
            logger.warning("Pre-filtering removed all messages. Using fallback topic message.")
            return [HumanMessage(content=f"Research the topic: {topic}")]
            
        # Replace history with pre-filtered version
        history = filtered_history
            
        # Ensure the first message is a HumanMessage
        if not isinstance(history[0], HumanMessage):
            logger.warning("First message in history is not a HumanMessage during sanitization. Adding initial topic.")
            history.insert(0, HumanMessage(content=f"Research the topic: {topic}"))
        
        # Check for tool call sequence validity
        sanitized_history = [history[0]]  # Start with the first message (HumanMessage)
        
        # Add AIMessage with tool_calls and corresponding ToolMessage
        tool_messages_added = False
        
        for i in range(1, len(history)):
            msg = history[i]
            
            # CRITICAL: Skip any message with empty content to prevent Gemini errors
            if self._has_empty_content(msg):
                logger.warning(f"Skipping message at index {i} with empty content or empty parts.")
                continue
                
            is_ai_message = isinstance(msg, AIMessage)
            has_explicit_tool_calls = is_ai_message and getattr(msg, 'tool_calls', None)
            has_embedded_tool_call = False
            
            # Check for embedded JSON tool call if no explicit ones
            if is_ai_message and not has_explicit_tool_calls and isinstance(msg.content, list):
                for item in msg.content:
                     if isinstance(item, str) and item.strip().startswith('```json'):
                         # Basic check for JSON block existence
                         has_embedded_tool_call = True
                         logger.debug(f"Sanitizer detected embedded JSON tool call pattern in msg at index {i}.")
                         break # Found one, no need to check further items in this message

            # --- Logic for adding AIMessage --- 
            if is_ai_message:
                # Ensure message doesn't have empty parts before adding
                cleaned_msg = self._ensure_valid_content(msg)
                if cleaned_msg is None:
                    logger.warning(f"Skipping AIMessage at index {i} - content couldn't be sanitized.")
                    continue
                    
                # Condition to add: Previous message must allow an AI message to follow
                # (e.g., Human message, or Tool message if the previous AI message requested tools)
                can_follow_previous = isinstance(sanitized_history[-1], (HumanMessage, ToolMessage))
                
                # If the *current* AI message has/implies tool calls, the *previous* message
                # in the *original* history should ideally be Human or an AI message without tool calls.
                # However, the primary rule is the sequence in the sanitized list.
                if can_follow_previous:
                    sanitized_history.append(cleaned_msg)
                    # Set flag if this message contains/implies tool calls, needed for next step (ToolMessage)
                    tool_messages_added = has_explicit_tool_calls or has_embedded_tool_call
                else:
                     logger.warning(f"Skipping AIMessage at original index {i} as it cannot follow {type(sanitized_history[-1]).__name__} in sanitized sequence.")
                     
            # --- Logic for adding ToolMessage --- 
            elif isinstance(msg, ToolMessage):
                # Ensure message doesn't have empty parts before adding
                cleaned_msg = self._ensure_valid_content(msg)
                if cleaned_msg is None:
                    logger.warning(f"Skipping ToolMessage at index {i} - content couldn't be sanitized.")
                    continue
                    
                # Condition to add: Must follow an AIMessage that had/implied tool calls
                previous_was_ai_with_tools = False
                if isinstance(sanitized_history[-1], AIMessage):
                    prev_ai_msg = sanitized_history[-1]
                    previous_was_ai_with_tools = (
                        getattr(prev_ai_msg, 'tool_calls', None) or # Explicit calls
                        (isinstance(prev_ai_msg.content, list) and any( # Embedded calls
                            isinstance(item, str) and item.strip().startswith('```json') 
                            for item in prev_ai_msg.content
                         ))
                    )
                    
                if previous_was_ai_with_tools:
                    sanitized_history.append(cleaned_msg)
                    tool_messages_added = False # Reset flag after tool response
                else:
                    logger.warning(f"Skipping ToolMessage at original index {i} as it does not follow an AIMessage with tool calls in sanitized sequence.")
            
            # --- Logic for adding HumanMessage --- 
            elif isinstance(msg, HumanMessage):
                 # Ensure message doesn't have empty parts before adding
                 cleaned_msg = self._ensure_valid_content(msg)
                 if cleaned_msg is None:
                     logger.warning(f"Skipping HumanMessage at index {i} - content couldn't be sanitized.")
                     continue
                     
                 # Can always add HumanMessage if it follows AI or Tool message
                 if isinstance(sanitized_history[-1], (AIMessage, ToolMessage)):
                     sanitized_history.append(cleaned_msg)
                     tool_messages_added = False # Reset flag for new turn
                 else:
                      # Avoid consecutive HumanMessages if something went wrong
                      logger.warning(f"Skipping HumanMessage at original index {i} as it cannot follow {type(sanitized_history[-1]).__name__} in sanitized sequence.")

        # Final check - ensure all messages have valid content before returning
        final_history = []
        for i, msg in enumerate(sanitized_history):
            # Double check message for empty content
            if not self._has_empty_content(msg):
                final_history.append(msg)
            else:
                logger.warning(f"Final validation: Removing message at index {i} with empty content.")
                
        # Add a fallback if we somehow end up with an empty history
        if not final_history:
            logger.warning("Sanitization resulted in empty history. Adding fallback topic message.")
            final_history = [HumanMessage(content=f"Research the topic: {topic}")]
            
        # Final check: Log messages and content types for debugging
        for i, msg in enumerate(final_history):
            content_type = type(msg.content).__name__
            if isinstance(msg.content, list):
                parts_info = f"[{len(msg.content)} parts]"
                content_info = f"{content_type}{parts_info}"
            else:
                content_len = len(str(msg.content)) if msg.content else 0
                content_info = f"{content_type}({content_len} chars)"
                
            logger.debug(f"Final message {i}: {type(msg).__name__} with content {content_info}")
                
        logger.info(f"Sanitized history: Original={len(history)}, Filtered={len(filtered_history)}, Final={len(final_history)} messages")
        
        # Add additional check for the exact message positions that might have issues
        for i, msg in enumerate(final_history):
            if i == 6:  # The position that's causing errors
                logger.warning(f"SPECIAL CHECK for position 6: {type(msg).__name__} with content type {type(msg.content).__name__}")
                if isinstance(msg.content, list):
                    for j, part in enumerate(msg.content):
                        part_type = type(part).__name__ 
                        part_preview = str(part)[:30] + "..." if isinstance(part, str) and len(str(part)) > 30 else part
                        logger.warning(f"  Position 6, Part {j}: type={part_type}, preview={part_preview}")
        
        return final_history

    def _optimize_history_for_primary_model(self, history: List[BaseMessage], topic: str, max_turns: int = 3) -> List[BaseMessage]:
        """Optimize message history to reduce token usage by keeping only the most relevant recent context.
        
        This function:
        1. Keeps only the most recent tool interactions (the last `max_turns` turns)
        2. Removes internal reasoning from AI messages to reduce tokens
        3. Ensures the history maintains a valid conversation flow
        
        Args:
            history: The full conversation history
            topic: The research topic for fallback messages
            max_turns: Maximum number of recent turns to keep (default: 3)
            
        Returns:
            Optimized message history with reduced token usage
        """
        if not history:
            return [HumanMessage(content=f"Research the topic: {topic}")]
        
        # Always keep the first message (instructions)
        optimized_history = [history[0]]
        
        # Group messages into interaction turns (each turn = AI message + following tool messages)
        turns = []
        current_turn = []
        
        for i in range(1, len(history)):
            msg = history[i]
            
            # Start a new turn when we see an AI message
            if isinstance(msg, AIMessage):
                if current_turn:
                    turns.append(current_turn)
                current_turn = [msg]
            # Add tool messages to the current turn
            elif isinstance(msg, ToolMessage) and current_turn:
                current_turn.append(msg)
            # Human messages generally start a new context
            elif isinstance(msg, HumanMessage):
                if current_turn:
                    turns.append(current_turn)
                current_turn = [msg]
            # Other message types just get added to current turn
            else:
                current_turn.append(msg)
        
        # Add the last turn if it exists
        if current_turn:
            turns.append(current_turn)
            
        # Keep only the most recent N turns plus the first turn (which might have instructions)
        relevant_turns = turns[-max_turns:] if len(turns) > max_turns else turns
        
        # Reconstruct the history from the selected turns
        for turn in relevant_turns:
            optimized_history.extend(turn)
            
        # Process AI messages to remove verbose reasoning
        for i, msg in enumerate(optimized_history):
            if isinstance(msg, AIMessage) and isinstance(msg.content, str):
                # Look for reasoning patterns and condense them
                content = msg.content
                
                # Reasoning pattern detection (simplified version of what's in chainlit_callbacks.py)
                reasoning_patterns = [
                    r"\*\*(?:1\.|Step 1:).*?Thinking Step-by-Step.*?\*\*",
                    r"\*\*Reasoning:?\*\*",
                    r"(?:^|\n)(?:1\.|Step 1:).*?Thinking",
                    r"(?:^|\n)Let me think through this",
                    r"(?:^|\n)I'll analyze this step by step",
                ]
                
                # Check if content contains any reasoning patterns
                has_reasoning = any(re.search(pattern, content, re.IGNORECASE) for pattern in reasoning_patterns)
                
                if has_reasoning:
                    # Find conclusion/action after reasoning
                    conclusion_patterns = [
                        r"\*\*(?:Conclusion|Summary|Action|Next Steps|Plan):?\*\*",
                        r"(?:^|\n)(?:Conclusion|Summary|Action|Next Steps|Plan):",
                        r"(?:^|\n)Based on (?:my|this) (?:analysis|reasoning)",
                        r"ACTION CONFIRMATION:",
                    ]
                    
                    # Extract just the conclusion part if we can find it
                    for pattern in conclusion_patterns:
                        match = re.search(pattern, content, re.IGNORECASE)
                        if match:
                            # Get everything from the conclusion onward
                            start_idx = match.start()
                            optimized_content = (
                                "[Reasoning condensed] " + 
                                content[start_idx:].strip()
                            )
                            # Replace original message content with optimized version
                            optimized_history[i] = AIMessage(
                                content=optimized_content, 
                                tool_calls=msg.tool_calls if hasattr(msg, 'tool_calls') else None
                            )
                            break
        
        logger.info(f"Optimized history for primary model: Original={len(history)}, Optimized={len(optimized_history)} messages")
        return optimized_history

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async exit handler - ensure MCP client is properly closed."""
        logger.info("Executing ResearcherAgent.__aexit__...")
        if hasattr(self, 'mcp_client') and self.mcp_client:
            try:
                logger.info("Closing MCP client resources in __aexit__")
                await self.mcp_client.__aexit__(exc_type, exc_val, exc_tb)
                logger.info("MCP client closed successfully in __aexit__")
            except Exception as e:
                logger.error(f"Error closing MCP client in __aexit__: {e}", exc_info=True)
        return False  # Don't suppress exceptions

    async def _generate_summary(
        self, 
        topic: str, 
        accumulated_content: str,
        run_config: Optional[RunnableConfig] = None,
        content_counts: Optional[Dict[str, int]] = None
    ) -> str:
        """Generate a final research summary with sources.
        
        Args:
            topic: The research topic
            accumulated_content: All accumulated content text
            run_config: Optional LangChain runnable config
            content_counts: Optional dictionary of content type counts
            
        Returns:
            Formatted summary with references
        """
        callbacks = self._extract_callbacks(run_config)
        
        try:
            # First try to get summary over content
            if self.content_manager and accumulated_content:
                # Generate a summary using the content manager
                logger.info(f"Generating final summary for topic: '{topic}'")
                # Pass topic to _summarize_content
                summary = await self._summarize_content(topic, accumulated_content) # <<< CHANGE: Pass topic
                
                # Generate sources section from content manager's tracked sources
                sources_section = self.content_manager.generate_sources_section()
                
                # Combine summary and sources in the required format
                formatted_summary = f"<summary>\n{summary}\n</summary>\n\n{sources_section}"
                
                logger.info(f"Generated formatted summary with sources section")
                return formatted_summary
            else:
                # Fallback if no content manager or accumulated content
                logger.warning("No content manager or accumulated content for summary. Using fallback approach.")
                # Generate empty sources section
                empty_sources = "<sources>\nNo sources were collected during research.\n</sources>"
                return f"<summary>\nNo research content was collected on the topic '{topic}'.\n</summary>\n\n{empty_sources}"
        except Exception as e:
            # Handle errors gracefully
            logger.error(f"Error generating summary: {e}", exc_info=True)
            error_summary = f"An error occurred while generating the research summary: {e}"
            error_sources = "<sources>\nSources unavailable due to error.\n</sources>"
            return f"<summary>\n{error_summary}\n</summary>\n\n{error_sources}"

    def _get_token_usage_statistics(self) -> str:
        """Get token usage statistics for the current research run.
        
        Returns:
            Formatted string with token usage statistics only.
        """
        try:
            stats_sections = []
            token_data_found = False
            
            # Display names for the three main models
            display_names = {
                'claude-3-7-sonnet': 'Claude 3.7 Sonnet',
                'gemini-2.5-pro': 'Gemini 2.5 Pro', 
                'gemini-2.0-flash': 'Gemini 2.0 Flash',
                # Add display name for 2.5 Flash Preview
                'gemini-2.5-flash-preview-04-17': 'Gemini 2.5 Flash',
                'gemini-2.5-pro-preview-03-25': 'Gemini 2.5 Pro', # Map specific preview name
            }
            
            # Check for callbacks to extract token data
            if hasattr(self, 'callbacks') and self.callbacks:
                # First, look for TokenCallbackManager
                for handler in self.callbacks:
                    if type(handler).__name__ == 'TokenCallbackManager':
                        # <<< PRIORITIZE the fallback logic accessing the processor directly >>>
                        if hasattr(handler, 'handler') and hasattr(handler.handler, 'token_cost_processor'):
                             processor = handler.handler.token_cost_processor
                             if hasattr(processor, 'token_usage') and processor.token_usage:
                                 # Convert internal token usage format to display format
                                 model_usage = {}
                                 logger.debug(f"Reconstructing model_usage from TokenCallbackManager's processor.token_usage: {processor.token_usage}") # DEBUG
                                 for model_name, usage in processor.token_usage.items():
                                     # Ensure cost is calculated or available here if needed later, 
                                     # though _format_token_table recalculates it.
                                     model_usage[model_name] = {
                                         'input_tokens': usage.get('prompt', 0),
                                         'output_tokens': usage.get('completion', 0),
                                         # 'total_cost': processor.total_cost # Cost is per-model, not total here
                                     }
                                 token_table = self._format_token_table(model_usage, display_names)
                                 stats_sections.append(token_table)
                                 token_data_found = True
                                 logger.info("Successfully extracted token data via TokenCallbackManager's processor.") # INFO
                                 break # Exit handler loop once data is found
                        # <<< Fallback to handler.model_usage if direct processor access fails >>>
                        elif hasattr(handler, 'model_usage') and handler.model_usage:
                             logger.warning("Using TokenCallbackManager.model_usage property as fallback.") # WARN
                             model_usage = handler.model_usage 
                             token_table = self._format_token_table(model_usage, display_names)
                             stats_sections.append(token_table)
                             token_data_found = True
                             break # Exit handler loop once data is found
                        else:
                            logger.warning("TokenCallbackManager found, but failed to get data from processor or model_usage property.")
                
                # If not found via TokenCallbackManager, try TokenUsageCallbackHandler
                if not token_data_found:
                    for handler in self.callbacks:
                         if type(handler).__name__ == 'TokenUsageCallbackHandler' and hasattr(handler, 'token_cost_processor'):
                             # Logic to extract from TokenUsageCallbackHandler...
                             processor = handler.token_cost_processor
                             if hasattr(processor, 'token_usage') and processor.token_usage:
                                 # Convert format
                                 model_usage = {}
                                 for model_name, usage in processor.token_usage.items():
                                     model_usage[model_name] = {
                                         'input_tokens': usage.get('prompt', 0),
                                         'output_tokens': usage.get('completion', 0),
                                         'total_cost': processor.total_cost
                                     }
                                 token_table = self._format_token_table(model_usage, display_names)
                                 stats_sections.append(token_table)
                                 token_data_found = True
                                 break
            
            # If no token data was found from callbacks
            if not token_data_found:
                stats_sections.append("### Token Usage\n\nDetailed token usage statistics not available.")
            
            # --- REMOVED Content Processed and Function Usage Sections --- 
            
            return "\n\n".join(stats_sections) # Return only token stats
            
        except Exception as e:
            logger.error(f"Error generating token usage statistics: {e}", exc_info=True)
            return "Error generating token usage statistics."

    def _format_token_table(self, model_usage: Dict[str, Dict[str, Any]], display_names: Dict[str, str]) -> str:
        """Helper method to format token usage data into a markdown table.
        
        Args:
            model_usage: Dictionary of model usage data
            display_names: Mapping of model keys to display names
            
        Returns:
            Formatted markdown table as a string
        """
        token_table = [
            "### Token Usage",
            "",
            "| Model | Input Tokens | Output Tokens | Total | Cost |",
            "|-------|--------------|---------------|-------|------|"
        ]
        
        total_input = 0
        total_output = 0
        total_cost = 0.0
        
        # Process each model
        for model_name in sorted(model_usage.keys()):
            usage = model_usage[model_name]
            
            # Get friendly display name
            display_name = display_names.get(model_name.lower(), model_name)
            
            # Extract token counts
            input_tokens = int(usage.get('input_tokens', 0))
            output_tokens = int(usage.get('output_tokens', 0))
            model_total = input_tokens + output_tokens
            
            # Get cost if available
            if 'total_cost' in usage:
                model_cost = float(usage['total_cost'])
            else:
                # Calculate based on model name
                from src.token_callback import _calculate_cost
                model_cost = _calculate_cost(model_name, input_tokens, output_tokens)
            
            # Only add to the table if there was actual usage
            if model_total > 0:
                token_table.append(
                    f"| **{display_name}** | {input_tokens:,} | {output_tokens:,} | {model_total:,} | ${model_cost:.4f} |"
                )
                
                # Add to totals
                total_input += input_tokens
                total_output += output_tokens
                total_cost += model_cost
        
        # Add total row if we had any models
        if total_input > 0 or total_output > 0:
            token_table.append("| **TOTAL** | **{:,}** | **{:,}** | **{:,}** | **${:.4f}** |".format(
                total_input, total_output, total_input + total_output, total_cost
            ))
        else:
            token_table.append("| *No token usage recorded* | | | | |")
            
        return "\n".join(token_table)

    async def _summarize_content(self, topic: str, content: str) -> str:
        """Summarize the provided content using the primary LLM and SUMMARY_PROMPT."""
        if not self.final_summary_llm: 
            logger.error("No primary LLM available. Cannot generate summary.")
            return "[Summary generation failed: Primary LLM not available]"
        logger.info(f"Attempting final summarization using PRIMARY LLM and SUMMARY_PROMPT for topic: '{topic}'")
        try:
            prompt_str = SUMMARY_PROMPT.format(topic=topic, accumulated_content=content)
            messages = [HumanMessage(content=prompt_str)]
            callbacks = getattr(self, 'callbacks', None)
            # Pass tags to suppress callback handler message
            run_config = {"callbacks": callbacks, "tags": ["final_summary_llm"]} if callbacks else {"tags": ["final_summary_llm"]}
            response = await self.final_summary_llm.ainvoke(messages, config=run_config)
            summary_text = getattr(response, 'content', None)
            if summary_text:
                logger.info(f"Successfully generated final summary using PRIMARY LLM (length: {len(summary_text)} chars)")
                return summary_text 
            else:
                logger.error("Primary LLM returned empty response or no content for final summary.")
                return "[Summary generation failed: LLM returned empty response]"
        except Exception as e:
            logger.error(f"Exception during final summarization with primary LLM: {e}", exc_info=True)
            return f"[Summary generation failed: {e}]"

    def _extract_structured_confirmation(self, content: str) -> Tuple[Optional[str], Dict[str, Any]]:
        """Extract the structured action confirmation if present.
        
        Args:
            content: The text content to search for a structured confirmation.
            
        Returns:
            A tuple of (tool_name, params) or (None, {}) if not found.
        """
        if not content or not isinstance(content, str):
            return None, {}
            
        confirmation_pattern = r"ACTION CONFIRMATION:\s*Tool:\s*(\w+)\s*Parameters:\s*([\s\S]+?)END CONFIRMATION"
        match = re.search(confirmation_pattern, content)
        
        if not match:
            return None, {}
            
        tool_name = match.group(1).strip()
        params_text = match.group(2).strip()
        
        # Parse parameters from bulleted list
        params = {}
        param_lines = re.findall(r'-\s*(\w+):\s*(.+?)(?:\n|$)', params_text)
        for key, value in param_lines:
            params[key.strip()] = value.strip()
        
        return tool_name, params

    async def _post_process_summary(self, summary: str, accumulated_content: str) -> str:
        """Post-process the summary to extract sections, fetch titles for URLs, and format."""
        logger.debug(f"Raw summary input for post-processing:\n{summary}")
        # Define section patterns
        summary_pattern = r"<summary>\\s*([\\s\\S]*?)\\s*</summary>"
        sources_pattern = r"<sources>\\s*([\\s\\S]*?)\\s*</sources>"
        
        try:
            # Extract the main summary content
            summary_match = re.search(summary_pattern, summary, re.DOTALL)
            summary_content = summary_match.group(1).strip() if summary_match else ""
            
            # Extract the sources section
            sources_match = re.search(sources_pattern, summary, re.DOTALL)
            raw_sources_content = sources_match.group(1).strip() if sources_match else ""

            # Fallback logic if structured format wasn't used
            if not summary_match:
                logger.warning("Structured summary format not detected, using fallback extraction")
                summary_content = summary
                raw_sources_content = "" # Initialize sources_content for fallback
                
                # Try to extract and clean up any references section in traditional format
                ref_section_match = re.search(r'(### References|### SOURCE LINKS)\\s*\\n', summary, re.IGNORECASE | re.DOTALL)
                if ref_section_match:
                    parts = re.split(r'(### References|### SOURCE LINKS)\\s*\\n', summary, flags=re.IGNORECASE | re.DOTALL, maxsplit=1)
                    if len(parts) >= 2:
                        summary_content = parts[0].strip()
                        raw_sources_content = parts[-1].strip()
                        
                        # Check for empty/irrelevant sources section
                        if ('(No URLs were present' in raw_sources_content or 
                            not raw_sources_content or 
                            raw_sources_content.isspace() or
                            "no sources cited" in raw_sources_content.lower()):
                            logger.info("Empty/irrelevant references section detected in traditional format, removing")
                            raw_sources_content = ""
                        else:
                            citation_refs = set(re.findall(r'\\[(\\d+)\\]', summary_content))
                            if not citation_refs:
                                logger.info("No citations found in summary, removing references section")
                                raw_sources_content = ""
                
                # If fallback didn't find sources, just return summary
                if not raw_sources_content:
                    return summary_content

            # --- Clean and Process Sources ---
            
            # Clean up raw sources content - remove template instructions or empty placeholders
            if (raw_sources_content and 
               ('[only include' in raw_sources_content.lower() or 
                '[if no sources' in raw_sources_content.lower() or 
                'leave this section empty' in raw_sources_content.lower() or
                raw_sources_content.strip() == "" or
                raw_sources_content.strip().isspace())):
                logger.info("Removing source instruction text or empty content from output")
                raw_sources_content = ""

            formatted_sources_lines = []
            if raw_sources_content:
                # Regex to find URLs, potentially within markdown links like [n](url) or just bare URLs
                # It also captures the potential citation number/prefix (like '[1]', '1.', '-')
                url_pattern = re.compile(r'^\s*([\[\d\]\.\-\*]+)?\s*(?:\[.*?\]\((?P<md_url>https?://[^\s\)]+)\)|(?P<bare_url>https?://[^\s]+))', re.MULTILINE)
                
                urls_to_fetch = []
                source_lines = raw_sources_content.split('\n')
                original_line_map = {} # Map URL to its original line structure

                for line in source_lines:
                    match = url_pattern.search(line)
                    if match:
                        url = match.group('md_url') or match.group('bare_url')
                        if url:
                            prefix = match.group(1) if match.group(1) else '*' # Default prefix
                            cleaned_url = self._clean_url(url) # Clean URL before fetching/storing
                            urls_to_fetch.append(cleaned_url)
                            original_line_map[cleaned_url] = {'prefix': prefix.strip(), 'original_url': url}
                    elif line.strip(): # Keep non-url lines as is if they exist
                         formatted_sources_lines.append(line)


                if urls_to_fetch:
                    async with aiohttp.ClientSession() as session:
                        tasks = [self._fetch_url_title(session, u) for u in urls_to_fetch]
                        titles = await asyncio.gather(*tasks, return_exceptions=True)

                    processed_urls = set()
                    title_map = {}
                    for url, title_result in zip(urls_to_fetch, titles):
                         if url not in processed_urls:
                            if isinstance(title_result, str):
                                title_map[url] = title_result
                            else:
                                title_map[url] = None # Mark as failed/no title
                                if not isinstance(title_result, asyncio.TimeoutError): # Log errors other than timeout
                                     logger.debug(f"Title fetch failed for {url}: {title_result}")
                            processed_urls.add(url)

                    # Reconstruct lines based on fetched titles
                    temp_formatted_lines = []
                    added_urls = set()
                    for url in urls_to_fetch:
                        if url in original_line_map and url not in added_urls:
                             line_info = original_line_map[url]
                             title = title_map.get(url)
                             prefix = line_info['prefix']
                             original_url = line_info['original_url'] # Use the original URL for display

                             if title:
                                 temp_formatted_lines.append(f"{prefix} {title} - {original_url}")
                             else:
                                 temp_formatted_lines.append(f"{prefix} {original_url}")
                             added_urls.add(url)
                    # Append any non-URL lines first, then the formatted URL lines
                    formatted_sources_lines.extend(temp_formatted_lines)


            # Prepare the final formatted output (summary + optional references)
            result = summary_content
            if formatted_sources_lines:
                formatted_sources_content = "\n".join(formatted_sources_lines)
                result += f"\n\n### References\n{formatted_sources_content}"
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing summary: {e}", exc_info=True)
            # Fallback to the original summary if processing fails
            logger.warning("Using original summary due to processing error")
            # Attempt to return the original summary *before* potential source modification
            original_summary_match = re.search(summary_pattern, summary, re.DOTALL)
            original_sources_match = re.search(sources_pattern, summary, re.DOTALL)
            
            fallback_summary = original_summary_match.group(1).strip() if original_summary_match else summary
            fallback_sources = original_sources_match.group(1).strip() if original_sources_match else ""
            
            if fallback_sources:
                 # Basic check if fallback sources look like URLs before appending
                 if 'http' in fallback_sources:
                     return fallback_summary + "\n\n### References\n" + fallback_sources
                 else: 
                     return fallback_summary # Don't append if sources look invalid
            else:
                 return fallback_summary

    def _clean_url(self, url):
        """Remove tracking parameters and fragments from a URL, except for Reddit URLs."""
        if 'reddit.com' in url:
            return url  # Keep full Reddit URLs
        parsed = urllib.parse.urlparse(url)
        cleaned = parsed._replace(query='', fragment='')
        return urllib.parse.urlunparse(cleaned)

    def _build_sources_visited(self, processed_counts, history):
        """Helper to build a deduplicated list of sources/URLs/tools already used, with cleaned URLs."""
        sources = set()
        # From processed_counts (web pages, reddit, mcp tools)
        if 'mcp_tools' in processed_counts:
            for tool, count in processed_counts['mcp_tools'].items():
                if count > 0:
                    sources.add(f"mcp:{tool}")
        if 'tool_functions' in processed_counts:
            for func, count in processed_counts['tool_functions'].items():
                if count > 0:
                    sources.add(f"func:{func}")
        # From history (look for URLs in tool messages)
        for msg in history:
            if hasattr(msg, 'content') and isinstance(msg.content, str):
                # Simple URL extraction
                for url in re.findall(r'https?://\S+', msg.content):
                    sources.add(self._clean_url(url))
        return sorted(sources)

    def _build_last_action(self, history):
        """Helper to extract the last AI reasoning and last tool call/result from history, with tool result summary if long."""
        last_ai = None
        last_tool = None
        for msg in reversed(history):
            if last_tool is None and hasattr(msg, 'tool_call_id'):
                last_tool = msg
            if last_ai is None and hasattr(msg, 'tool_calls'):
                last_ai = msg
            if last_ai and last_tool:
                break
        tool_result = getattr(last_tool, 'content', None) if last_tool else None
        if tool_result and isinstance(tool_result, str) and len(tool_result) > 500:
            tool_result = tool_result[:500] + '... [truncated]'
        return {
            "reasoning": getattr(last_ai, 'content', None) if last_ai else None,
            "tool_call": getattr(last_ai, 'tool_calls', None) if last_ai else None,
            "tool_result": tool_result
        }

    def _get_tool_intent(self, message: BaseMessage, history: List[BaseMessage]) -> Optional[Tuple[str, str]]:
        """Helper to determine the intent (tool_name, canonical_arg) for a ToolMessage."""
        if not isinstance(message, ToolMessage) or not message.tool_call_id:
            return None

        # Find the corresponding AIMessage with the tool call
        tool_name = None
        tool_args = None
        for i in range(len(history) - 1, -1, -1):
            prev_msg = history[i]
            if isinstance(prev_msg, AIMessage) and prev_msg.tool_calls:
                for call in prev_msg.tool_calls:
                    if call.get("id") == message.tool_call_id:
                        tool_name = call.get("name")
                        tool_args = call.get("args")
                        break
                if tool_name: # Found the call
                    break
            # Stop searching if we go past the likely relevant AIMessage
            if i < len(history) - 5: # Heuristic limit
                 break

        if not tool_name or not tool_args:
            return None

        # Determine canonical argument based on tool name
        canonical_arg = None
        if isinstance(tool_args, dict):
            if tool_name == "get_transcripts":
                canonical_arg = tool_args.get("url")
            elif tool_name == "web_browser" and tool_args.get("action") == "search":
                canonical_arg = tool_args.get("query")
            elif tool_name == "web_browser" and (tool_args.get("action") == "navigate_and_extract" or tool_args.get("action") == "extract"):
                canonical_arg = tool_args.get("url")
            elif tool_name == "reddit_search":
                canonical_arg = tool_args.get("query")
            elif tool_name == "reddit_extract_post":
                canonical_arg = tool_args.get("post_url")
            # Add other tools as needed

        if isinstance(canonical_arg, str):
            return (tool_name, canonical_arg)
        else:
            # Fallback if args format is unexpected or canonical arg not found/string
            return (tool_name, str(tool_args))

    def _filter_history_for_llm(self, history: List[BaseMessage]) -> List[BaseMessage]:
        """Filter out resolved error messages from the history for LLM input."""
        successful_intents = set()
        intent_map = {}

        # First pass: Identify successful intents
        temp_intent_map = {} # Map tool_call_id to intent for this pass
        for i, msg in enumerate(history):
            if isinstance(msg, AIMessage) and msg.tool_calls:
                for call in msg.tool_calls:
                    call_id = call.get("id")
                    # Simple intent: (tool_name, primary_arg_value_or_hash)
                    tool_name = call.get("name")
                    tool_args = call.get("args")
                    canonical_arg_val = str(tool_args) # Default to stringified args
                    if isinstance(tool_args, dict):
                        if tool_name == "get_transcripts": canonical_arg_val = tool_args.get("url", str(tool_args))
                        elif tool_name == "web_browser" and tool_args.get("action") == "search": canonical_arg_val = tool_args.get("query", str(tool_args))
                        elif tool_name == "web_browser" and (tool_args.get("action") == "navigate_and_extract" or tool_args.get("action") == "extract"): canonical_arg_val = tool_args.get("url", str(tool_args))
                        # Add more tool-specific canonical args if needed
                    
                    if call_id and tool_name and canonical_arg_val:
                         intent = (tool_name, canonical_arg_val)
                         temp_intent_map[call_id] = intent
            elif isinstance(msg, ToolMessage) and msg.tool_call_id:
                intent = temp_intent_map.get(msg.tool_call_id)
                if intent and not msg.content.startswith("Error"):
                    successful_intents.add(intent)

        # Second pass: Build filtered history
        filtered_history = []
        final_intent_map = {} # Map tool_call_id to intent for the filtering pass
        for i, msg in enumerate(history):
            if isinstance(msg, AIMessage) and msg.tool_calls:
                filtered_history.append(msg)
                # Rebuild intent map for filtering pass
                for call in msg.tool_calls:
                    call_id = call.get("id")
                    tool_name = call.get("name")
                    tool_args = call.get("args")
                    canonical_arg_val = str(tool_args) # Default
                    if isinstance(tool_args, dict):
                        if tool_name == "get_transcripts": canonical_arg_val = tool_args.get("url", str(tool_args))
                        elif tool_name == "web_browser" and tool_args.get("action") == "search": canonical_arg_val = tool_args.get("query", str(tool_args))
                        elif tool_name == "web_browser" and (tool_args.get("action") == "navigate_and_extract" or tool_args.get("action") == "extract"): canonical_arg_val = tool_args.get("url", str(tool_args))
                    
                    if call_id and tool_name and canonical_arg_val:
                         intent = (tool_name, canonical_arg_val)
                         final_intent_map[call_id] = intent
            elif isinstance(msg, ToolMessage) and msg.tool_call_id:
                intent = final_intent_map.get(msg.tool_call_id)
                if intent:
                    is_successful_intent = intent in successful_intents
                    is_error_message = msg.content.startswith("Error")
                    
                    if is_successful_intent:
                        # If the intent eventually succeeded, only keep non-error messages
                        if not is_error_message:
                            filtered_history.append(msg)
                    else:
                        # If the intent never succeeded, keep all messages (including errors)
                        filtered_history.append(msg)
                else:
                     # ToolMessage without a mapped intent (shouldn't happen ideally)
                     logger.warning(f"Could not map tool_call_id {msg.tool_call_id} to an intent. Including message.")
                     filtered_history.append(msg)
            elif not isinstance(msg, ToolMessage): # Keep HumanMessages etc.
                 filtered_history.append(msg)
        
        logger.info(f"History filtering: Original={len(history)}, Filtered={len(filtered_history)}. Successful intents identified: {len(successful_intents)}")
        return filtered_history

    def _get_tool_metadata_string(self) -> str:
        """Return a markdown-formatted string of tool metadata for the prompt.
        
        Returns:
            Markdown string listing available tools with descriptions and limits.
        """
        lines = []
        lines.append('| Tool Name | Description | Min Calls | Max Calls |')
        lines.append('|-----------|-------------|-----------|-----------|')
        
        # Add entries for each tool
        for tool in self.tools:
            desc = getattr(tool, 'description', 'No description').split('\n')[0]
            tool_name = tool.name
            # Get min/max from tool_configs if available
            if tool_name in self.tool_configs:
                min_calls = self.tool_configs[tool_name].get('min_calls', 0)
                max_calls = self.tool_configs[tool_name].get('max_calls', 3)
            else:
                min_calls = 0
                max_calls = 3
            lines.append(f"| {tool_name} | {desc} | {min_calls} | {max_calls} |")
        
        # Add default entries for common tools if they're not in self.tools
        default_tools = {
            'web_browser': 'Search the web and extract content from web pages',
            'reddit_search': 'Search Reddit for posts and comments',
            'reddit_extract_post': 'Extract content from a Reddit post URL'
        }
        
        tool_names = [tool.name for tool in self.tools]
        for name, desc in default_tools.items():
            if name not in tool_names:
                if name in self.tool_configs:
                    min_calls = self.tool_configs[name].get('min_calls', 0)
                    max_calls = self.tool_configs[name].get('max_calls', 3)
                else:
                    if name == 'web_browser':
                        min_calls = MIN_REGULAR_WEB_PAGES
                        max_calls = MAX_REGULAR_WEB_PAGES
                    elif name in ['reddit_search', 'reddit_extract_post']:
                        min_calls = MIN_POSTS_PER_SEARCH
                        max_calls = MAX_POSTS_PER_SEARCH
                    else:
                        min_calls = 0
                        max_calls = 3
                lines.append(f"| {name} | {desc} | {min_calls} | {max_calls} |")
                
        return '\n'.join(lines)

    # Add a helper to generate the Tool Usage Tracker markdown table
    def _generate_tool_usage_tracker_md(self, processed_counts: Dict[str, int]) -> str:
        """Generate a markdown table showing tool usage status.
        
        Args:
            processed_counts: Dictionary tracking tool usage counts
            
        Returns:
            Markdown table string showing tool usage status
        """
        tracker_lines = ["| Tool Name | Used | Min Planned | Max Planned |",
                        "|-----------|------|-------------|-------------|"]
        
        for tool_name, config in self.tool_configs.items():
            min_planned = config.get('min_calls', 0)
            max_planned = config.get('max_calls', 3)
            
            # Get usage count for this tool
            used = 0
            if tool_name in processed_counts:
                used = processed_counts[tool_name]
            elif tool_name == 'web_browser' and 'regular_web_pages' in processed_counts:
                used = processed_counts['regular_web_pages']
            elif tool_name in ['reddit_search', 'reddit_extract_post'] and 'reddit_posts' in processed_counts:
                used = processed_counts['reddit_posts']
            elif 'base_tool_calls' in processed_counts and tool_name in processed_counts['base_tool_calls']:
                used = processed_counts['base_tool_calls'][tool_name]
            
            tracker_lines.append(f"| {tool_name} | {used} | {min_planned} | {max_planned} |")
            
        return "\n".join(tracker_lines)

    async def _get_tool_correction_suggestion(
        self,
        tool_name: str,
        failed_args: Dict[str, Any],
        error_message: str,
        run_config: Optional[RunnableConfig] = None
    ) -> Optional[Dict[str, Any]]:
        """Helper method to get correction suggestions for failed tool calls.
        
        Args:
            tool_name: Name of the tool that failed
            failed_args: The arguments that caused the failure
            error_message: The error message from the failed call
            run_config: Optional run configuration for the LLM call
            
        Returns:
            Corrected arguments dictionary or None if correction failed
        """
        logger.info(f"Attempting to get correction suggestion for tool {tool_name}")
        
        # First, check if we can determine a correction directly from the error message
        # This helps for common nested field errors without requiring LLM calls
        corrected_args = self._try_direct_correction(tool_name, failed_args, error_message)
        if corrected_args:
            logger.info(f"Direct correction applied for {tool_name}: {corrected_args}")
            return corrected_args
            
        # If direct correction failed, proceed with LLM-based correction
        # First, find the tool object to get its description
        tool_to_use = None
        for tool in self.tools:
            if tool.name == tool_name:
                tool_to_use = tool
                break
                
        if not tool_to_use:
            logger.warning(f"Cannot generate correction for unknown tool: {tool_name}")
            return None
            
        # Get tool description
        tool_description = getattr(tool_to_use, 'description', '')
        if not tool_description:
            logger.warning(f"Tool {tool_name} has no description, cannot generate correction")
            return None
        
        # Create correction prompt input
        correction_input = {
            "tool_name": tool_name,
            "failed_args": str(failed_args),  # Convert dict to string for prompt
            "error_message": str(error_message),
            "tool_description": tool_description
        }
        
        try:
            # Build and invoke correction chain
            from config.prompts import TOOL_CORRECTION_PROMPT
            from langchain_core.output_parsers import StrOutputParser
            
            # Use the primary llm for correction logic
            correction_llm = self.llm_client
            correction_chain = TOOL_CORRECTION_PROMPT | correction_llm | StrOutputParser()
            
            # Set up callbacks for the LLM call
            if run_config is None:
                run_config = {"callbacks": self.callbacks}
                
            logger.info(f"Invoking correction LLM for {tool_name}...")
            corrected_args_str = await correction_chain.ainvoke(correction_input, config=run_config)
            logger.info(f"Correction LLM response for {tool_name}: {corrected_args_str}")
            
            # Parse the correction and convert to a dictionary
            try:
                # Clean potential markdown code fences
                if corrected_args_str.startswith("```json"):
                    corrected_args_str = corrected_args_str.replace("```json", "").replace("```", "").strip()
                elif corrected_args_str.startswith("```"):
                    corrected_args_str = corrected_args_str.replace("```", "").strip()
                    
                corrected_args_json = json.loads(corrected_args_str)
                if not isinstance(corrected_args_json, dict):
                    raise ValueError(f"Expected dict but got {type(corrected_args_json)}")
                    
                logger.info(f"Successfully parsed correction for {tool_name}: {json.dumps(corrected_args_json)}")
                return corrected_args_json
                
            except json.JSONDecodeError as json_err:
                logger.warning(f"Failed to parse correction JSON for {tool_name}: {json_err}. Raw LLM output: {corrected_args_str}")
                return None
                
        except Exception as correction_err:
            logger.error(f"Error during correction generation for {tool_name}: {correction_err}", exc_info=True)
            return None

    def _try_direct_correction(self, tool_name: str, failed_args: Dict[str, Any], error_message: str) -> Optional[Dict[str, Any]]:
        """Attempt to directly correct arguments based on error patterns without requiring LLM.
        
        This handles common patterns like missing nested fields in a generic way.
        
        Args:
            tool_name: Name of the tool that failed
            failed_args: The arguments that caused the failure
            error_message: The error message from the failed call
            
        Returns:
            Corrected arguments or None if direct correction is not possible
        """
        # First apply our general preprocessing logic
        processed_args = self._preprocess_mcp_tool_args(tool_name, failed_args)
        
        # If preprocessing changed the arguments, it might have fixed the issue
        if processed_args != failed_args:
            logger.info(f"Preprocessor modified arguments for {tool_name}, attempting with preprocessed args")
            return processed_args
        
        # Check for nested path errors (e.g., "request.term")
        nested_field_match = re.search(r"(\w+)\.(\w+).*Field required", error_message)
        
        if nested_field_match:
            parent_field, child_field = nested_field_match.groups()
            
            # Generic pattern: any matching top-level key should be moved into the required nested structure
            # Common synonyms that might be applicable across multiple tools
            synonyms = {
                "query": ["term", "q", "search", "keywords"],
                "content": ["text", "input", "body"],
                "text": ["content", "input", "body"],
                "term": ["query", "q", "search", "keywords"]
            }
            
            # Check if we have any key that could be a synonym for the missing field
            for key in failed_args:
                # Direct match to the missing child field
                if key == child_field:
                    # Create a new object with the nested structure
                    corrected = {parent_field: {child_field: failed_args[key]}}
                    # Copy over any other fields that might be at the parent level
                    if parent_field in failed_args and isinstance(failed_args[parent_field], dict):
                        for k, v in failed_args[parent_field].items():
                            if k != child_field:  # Don't overwrite our correction
                                corrected[parent_field][k] = v
                    return corrected
                
                # Check if the key is a synonym for the required field
                for base_field, synonyms_list in synonyms.items():
                    if child_field == base_field and key in synonyms_list:
                        # Create a new object with the nested structure
                        corrected = {parent_field: {child_field: failed_args[key]}}
                        # Copy over any other fields that might be at the parent level
                        if parent_field in failed_args and isinstance(failed_args[parent_field], dict):
                            for k, v in failed_args[parent_field].items():
                                if k != child_field:  # Don't overwrite our correction
                                    corrected[parent_field][k] = v
                        return corrected
                    elif key == base_field and child_field in synonyms_list:
                        # Create a new object with the nested structure
                        corrected = {parent_field: {child_field: failed_args[key]}}
                        # Copy over any other fields that might be at the parent level
                        if parent_field in failed_args and isinstance(failed_args[parent_field], dict):
                            for k, v in failed_args[parent_field].items():
                                if k != child_field:  # Don't overwrite our correction
                                    corrected[parent_field][k] = v
                        return corrected
        
        # Check for completely missing parent object
        parent_missing_match = re.search(r"(\w+)\s+Field required", error_message)
        
        if parent_missing_match and not nested_field_match:
            parent_field = parent_missing_match.group(1)
            
            # Create empty parent object if it's missing
            if parent_field not in failed_args:
                # Look for any parameter that might make sense to put in this object
                # Common patterns for parameter movements
                # If we found certain keys at the top level, move them into parent object
                potential_child_keys = ["query", "term", "text", "content", "input", "url", "id", "prompt"]
                corrected = {parent_field: {}}
                
                # Copy any potential child keys from the top level
                for key in potential_child_keys:
                    if key in failed_args:
                        corrected[parent_field][key] = failed_args[key]
                
                # If we made any changes, return the corrected args
                if corrected[parent_field]:
                    # Copy all other top-level keys that weren't moved
                    for key, value in failed_args.items():
                        if key not in potential_child_keys:
                            corrected[key] = value
                    return corrected
                
                # If we couldn't find any keys to move, just create empty parent
                return {parent_field: {}, **{k: v for k, v in failed_args.items() if k != parent_field}}
                
        # Fall back to standard LLM-based correction if we can't detect a pattern
        return None

    def normalize_tool_args(self, tool_args: Dict[str, Any], tool: Any) -> Dict[str, Any]:
        """Normalize tool arguments to match the tool's expected schema.
        
        Args:
            tool_args: The arguments dictionary to normalize
            tool: The tool object that will receive these arguments
            
        Returns:
            Normalized arguments dictionary
        """
        # First use the global normalize_tool_args helper function
        normalized_args = normalize_tool_args(tool_args, tool)
        
        # Now apply our preprocessing logic to handle JSON strings and other normalization
        tool_name = getattr(tool, 'name', None)
        if tool_name:
            normalized_args = self._preprocess_mcp_tool_args(tool_name, normalized_args)
        
        # Special handling for specific tool types or patterns
        if tool_name == 'web_browser':
            # Ensure action parameter is a string
            if 'action' in normalized_args:
                normalized_args['action'] = str(normalized_args['action'])
                
            # For navigate_and_extract actions, ensure url is a string
            if normalized_args.get('action') == 'navigate_and_extract' and 'url' in normalized_args:
                if not isinstance(normalized_args['url'], str):
                    normalized_args['url'] = str(normalized_args['url'])
                    
            # For search actions, ensure query is a string
            if normalized_args.get('action') == 'search' and 'query' in normalized_args:
                if not isinstance(normalized_args['query'], str):
                    normalized_args['query'] = str(normalized_args['query'])
                    
        # Special handling for reddit tools
        elif tool_name in ['reddit_search', 'reddit_extract_post']:
            if tool_name == 'reddit_search' and 'query' in normalized_args:
                if not isinstance(normalized_args['query'], str):
                    normalized_args['query'] = str(normalized_args['query'])
                    
            if 'post_url' in normalized_args and not isinstance(normalized_args['post_url'], str):
                normalized_args['post_url'] = str(normalized_args['post_url'])
        
        # Return the normalized arguments
        return normalized_args

    def _has_empty_content(self, message: BaseMessage) -> bool:
        """Checks if a message has empty content or empty parts that would cause Gemini errors.
        
        Args:
            message: The message to check
            
        Returns:
            True if the message has empty content that would cause errors, False otherwise
        """
        # Check for None content
        if message.content is None:
            return True
            
        # Check for empty string content
        if isinstance(message.content, str) and not message.content.strip():
            return True
            
        # Check for list content with empty items
        if isinstance(message.content, list):
            # Empty list
            if not message.content:
                return True
                
            # List with empty items
            for item in message.content:
                if item is None:
                    return True
                if isinstance(item, str) and not item.strip():
                    return True
                if isinstance(item, dict) and not item:
                    return True
        
        # Content seems valid
        return False
        
    def _ensure_valid_content(self, message: BaseMessage) -> Optional[BaseMessage]:
        """Ensures a message has valid content that won't cause Gemini errors.
        
        This function:
        1. Removes empty parts from message content
        2. Ensures string content is not empty
        3. Returns a copy of the message with sanitized content
        
        Args:
            message: The message to sanitize
            
        Returns:
            A copy of the message with sanitized content, or None if the content can't be sanitized
        """
        # Handle None content more aggressively
        if message.content is None:
            logger.warning("Replacing None content with placeholder text in message")
            return type(message)(content="[No content available]")
            
        # More aggressively handle string content
        if isinstance(message.content, str):
            # Ensure string content is not empty
            content = message.content.strip()
            if not content:
                logger.warning("Replacing empty string content with placeholder text")
                return type(message)(content="[Empty content replaced]")
            # Return a new message to ensure we have a fresh object
            return type(message)(content=content)
            
        # Handle list content more carefully
        if isinstance(message.content, list):
            # Empty list is a problem
            if not message.content:
                logger.warning("Replacing empty list content with placeholder text")
                return type(message)(content="[Empty list content replaced]")
                
            # Filter out empty parts from list content
            valid_parts = []
            for item in message.content:
                # Skip None values
                if item is None:
                    continue
                    
                # Handle string items
                if isinstance(item, str):
                    if item.strip():  # Only add non-empty strings
                        valid_parts.append(item.strip())  # Ensure no leading/trailing whitespace
                    continue
                        
                # Handle dict items
                if isinstance(item, dict):
                    if item:  # Only add non-empty dicts
                        # Recursively clean dict values
                        cleaned_dict = {}
                        for k, v in item.items():
                            if isinstance(v, str):
                                v_clean = v.strip()
                                if v_clean:  # Only add non-empty string values
                                    cleaned_dict[k] = v_clean
                            elif v is not None:  # Add any non-None value
                                cleaned_dict[k] = v
                        if cleaned_dict:  # Only add if we have values left
                            valid_parts.append(cleaned_dict)
                    continue
                
                # Any other non-None type, add as is
                valid_parts.append(item)
                
            # If we have valid parts, create a new message with them
            if valid_parts:
                # Create new message of same type with filtered content
                return type(message)(
                    content=valid_parts,
                    tool_calls=getattr(message, 'tool_calls', None)
                )
            else:
                # No valid parts found, create a message with placeholder content
                logger.warning("All content parts were invalid or empty, replacing with placeholder")
                return type(message)(content="[No valid content parts]")
        
        # Additional checks for other types
        # If a message somehow has a zero-length content or otherwise problematic content
        content_str = str(message.content)
        if not content_str or content_str.isspace():
            logger.warning(f"Content conversion to string resulted in empty content of type {type(message.content).__name__}")
            return type(message)(content="[Content replaced]")
        
        # If all checks pass, return a copy of the message to ensure we have a fresh object
        return message

    async def _fetch_url_title(self, session: aiohttp.ClientSession, url: str) -> Optional[str]:
        """Asynchronously fetch the title of a URL."""
        if not url or not url.startswith(('http://', 'https://')):
            return None
        try:
            # Add a common user-agent to avoid blocking
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
            async with session.get(url, timeout=5, headers=headers, ssl=False) as response: # Added ssl=False for potential SSL issues
                response.raise_for_status()  # Raise an exception for bad status codes
                # Limit reading size to avoid memory issues with large pages
                html_content = await response.text(encoding='utf-8', errors='ignore')
                
                # Extract title using regex (simple approach)
                title_match = re.search(r'<title>(.*?)</title>', html_content, re.IGNORECASE | re.DOTALL)
                if title_match:
                    title = title_match.group(1).strip()
                    # Clean up title (optional: decode HTML entities, remove extra whitespace)
                    import html
                    title = html.unescape(title)
                    title = re.sub(r'\s+', ' ', title).strip()
                    return title if title else None 
                return None
        except asyncio.TimeoutError:
            logger.warning(f"Timeout fetching title for URL: {url}")
            return None
        except aiohttp.ClientError as e:
            logger.warning(f"HTTP ClientError fetching title for URL {url}: {e}")
            return None
        except Exception as e:
            logger.warning(f"Error fetching title for URL {url}: {e}")
            return None

    def _extract_callbacks(self, run_config: Optional[Dict[str, Any]] = None) -> Optional[List[BaseCallbackHandler]]:
        """Extract callbacks from a runnable config.
        
        Args:
            run_config: Optional runnable config dictionary
            
        Returns:
            List of callback handlers or None
        """
        if run_config is None:
            return self.callbacks
            
        return run_config.get("callbacks", self.callbacks)

    # Add this method before the _run_research_core method
    def _preprocess_mcp_tool_args(self, tool_name: str, tool_args: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess arguments for MCP tools before they're sent to the tool.
        
        This method handles common argument normalization patterns for MCP tools:
        - Converting JSON strings to dictionaries
        - Managing nested parameters
        - Handling quoted string values
        
        Args:
            tool_name: The name of the tool being called
            tool_args: The arguments dictionary to preprocess
            
        Returns:
            Normalized arguments dictionary
        """
        if not tool_args:
            return tool_args
            
        # Create a copy to avoid modifying the original
        normalized_args = tool_args.copy()
        
        # Process any string values that might be JSON
        for key, value in list(normalized_args.items()):
            if isinstance(value, str) and value.strip().startswith('{') and value.strip().endswith('}'):
                try:
                    parsed_dict = json.loads(value)
                    if isinstance(parsed_dict, dict):
                        normalized_args[key] = parsed_dict
                        logger.debug(f"Converted string to dict for {key}: {parsed_dict}")
                except json.JSONDecodeError:
                    # Not valid JSON, leave as is
                    pass
            elif isinstance(value, str):
                # Remove quotes from string values (common model output issue)
                if value.startswith('"') and value.endswith('"') or value.startswith("'") and value.endswith("'"):
                    normalized_args[key] = value[1:-1]
                    logger.debug(f"Removed quotes from {key}: {normalized_args[key]}")
        
        # Generic handling for nested parameters without tool-specific logic
        # Look for common parameter movements (e.g., query to request.term)
        common_nesting_patterns = [
            # Check for [top-level key] + [object without that key as nested field]
            # For example: query at top level + request object without 'term' or 'query' inside
            {"from_key": "query", "to_object": "request", "to_key": "term"},
            {"from_key": "query", "to_object": "request", "to_key": "query"},
            {"from_key": "text", "to_object": "request", "to_key": "content"},
            {"from_key": "input", "to_object": "request", "to_key": "content"}
        ]
        
        # Check each pattern
        for pattern in common_nesting_patterns:
            from_key = pattern["from_key"]
            to_object = pattern["to_object"]
            to_key = pattern["to_key"]
            
            # If we have the top-level key and the object exists but doesn't have the nested key
            if (from_key in normalized_args and 
                to_object in normalized_args and 
                isinstance(normalized_args[to_object], dict) and
                to_key not in normalized_args[to_object]):
                
                # Move the value to the nested location
                normalized_args[to_object][to_key] = normalized_args[from_key]
                # Remove the top-level key to avoid confusion
                del normalized_args[from_key]
                logger.debug(f"Moved {from_key} to {to_object}.{to_key}: {normalized_args}")
                break  # Only apply one pattern
        
        # Check for keys inside objects that might need to be renamed
        # For example: request.query -> request.term
        for key, value in normalized_args.items():
            if isinstance(value, dict):
                common_field_mappings = {
                    "query": "term",
                    "q": "term",
                    "search": "term",
                    "text": "content",
                    "input": "content"
                }
                
                for from_field, to_field in common_field_mappings.items():
                    # If object has source field but not destination field
                    if from_field in value and to_field not in value:
                        value[to_field] = value[from_field]
                        # Keep original key for backward compatibility
                        # unless we're certain it should be removed
                        logger.debug(f"Added {key}.{to_field} from {key}.{from_field}")
        
        return normalized_args