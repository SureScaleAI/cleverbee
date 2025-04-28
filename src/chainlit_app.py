import asyncio
import logging
import os
import sys
from datetime import datetime
import contextlib
from typing import Optional, List
import re

# --- Ensure project root is in path for imports --- #
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --- End Path Setup --- #

import chainlit as cl
from chainlit.input_widget import TextInput # For user input if needed later
from chainlit.action import Action # Import Action for custom buttons

# === Caching Setup ===
import langchain
# from langchain.cache import InMemoryCache # Keep previous for reference
from langchain_community.cache import SQLiteCache
from src.advanced_cache import NormalizingCache # Reverted to absolute import
# langchain.llm_cache = InMemoryCache()
langchain.llm_cache = NormalizingCache(database_path=".langchain.db")
# print("INFO: LangChain LLM Caching enabled (In-Memory).")
print("INFO: LangChain LLM Caching enabled (NormalizingCache at .langchain.db).")
# === End Caching Setup ===

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables early
from dotenv import load_dotenv
load_dotenv()

# Project imports (adjust paths/names as needed)
from config.settings import PRIMARY_MODEL_TYPE, LOCAL_MODEL_NAME, LOG_LEVEL, AVAILABLE_TOOLS, SUMMARIZER_MODEL, USE_LOCAL_SUMMARIZER_MODEL, NEXT_STEP_MODEL # Import new settings
from src.agent.researcher_agent import ResearcherAgent
from src.llm_clients.factory import get_llm_client
from src.token_callback import TokenCallbackManager, TokenCostProcess, TokenUsageCallbackHandler # Import the new TokenUsageCallbackHandler
# Import the adapter and helper function
# <<< Removed MCPToolWrapper import >>>
from src.tools.tool_registry import load_mcp_server_configs
# We'll create this callback handler next
# from src.chainlit_callbacks import ChainlitCallbackHandler 
from src.chainlit_callbacks import ChainlitCallbackHandler # Import the handler
from src.browser import PlaywrightBrowserTool # Needed for cleanup
from src.browser_manager import browser_manager # Import the browser manager
from src.cache_monitor import initialize_cache_monitoring # Import cache monitoring
from src.content_manager import ContentManager

PlaywrightBrowserTool.model_rebuild()

# Apply log level from config (ensure config is loaded)
try:
    import config.settings # Trigger loading if not already done
    log_level_str = getattr(config.settings, 'LOG_LEVEL', 'INFO').upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    logging.getLogger().setLevel(log_level)
    # Set Chainlit's logger level too
    logging.getLogger("chainlit").setLevel(log_level) 
    logger.info(f"Logging level set to: {log_level_str} ({log_level})")
except Exception as e:
    logger.warning(f"Could not apply log level from config: {e}")

# ADDED: Import TokenCostProcess
from src.token_callback import TokenCostProcess

# ADDED: Create the central token cost processor (for this app instance)
token_cost_processor = TokenCostProcess()
logger.info("Created central TokenCostProcess instance for Chainlit app.")

# Initialize cache monitoring (after langchain cache is set up, before any LLM calls)
# MODIFIED: Pass the processor instance
cache_monitor = initialize_cache_monitoring(token_cost_processor=token_cost_processor)
if cache_monitor:
    logger.info("Cache monitoring initialized. Token savings from cache hits will be tracked.")
else:
    logger.warning("Cache monitoring not initialized. Token savings from cache hits will not be tracked.")
logger.info(f"[DIAGNOSTIC] Logging level: {logging.getLogger().getEffectiveLevel()} ({logging.getLevelName(logging.getLogger().getEffectiveLevel())})")
logger.info(f"[DIAGNOSTIC] Cache instance: {langchain.llm_cache}")
logger.info(f"[DIAGNOSTIC] Cache monitor initialized: {cache_monitor is not None}")
if cache_monitor:
    logger.info(f"[DIAGNOSTIC] Cache monitor details: {cache_monitor}")
print("Reached cache monitor init (chainlit_app.py)")

@cl.on_chat_start
async def start_chat():
    logger.info("Starting new chat session...")

    # --- Loading Indicator ---
    loading_msg = cl.Message(content="🔄 Initializing CleverBee... Please wait.", author="System")
    await loading_msg.send()
    cl.user_session.set("loading_msg", loading_msg)

    # --- Token Tracking Setup for this Session ---
    session_token_cost_processor = TokenCostProcess()
    session_token_handler = TokenUsageCallbackHandler(session_token_cost_processor)
    chainlit_callback = ChainlitCallbackHandler(token_processor=session_token_cost_processor)
    session_callbacks = [session_token_handler, chainlit_callback]
    cl.user_session.set("token_usage_handler", session_token_handler)
    cl.user_session.set("chainlit_callback", chainlit_callback)
    cl.user_session.set("token_cost_processor", session_token_cost_processor)

    # --- Initialize Browser Manager ---
    try:
        await browser_manager.initialize_browser()
        logger.info("Browser manager initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize browser manager: {e}", exc_info=True)
        await cl.ErrorMessage(content=f"Failed to initialize browser: {e}").send()
        return

    # --- Initialize LLM Client ---
    try:
        llm_client = get_llm_client(
            provider=PRIMARY_MODEL_TYPE,
            callbacks=session_callbacks,
            use_retry_wrapper=True,  # Enable retry functionality
            max_retries=3,  # Set maximum retries to 3
            max_tokens=config.settings.MAX_CONTINUED_CONVERSATION_TOKENS  # <--- ADDED
        )
        cl.user_session.set("llm_client", llm_client)
        logger.info(f"LLM Client ({PRIMARY_MODEL_TYPE}) initialized and stored in session.")
    except Exception as e:
        error_message = f"Failed to initialize the language model: {e}"
        logger.error(error_message, exc_info=True)
        
        # Provide a more user-friendly error message with potential solutions
        user_error_message = f"Failed to initialize the language model: {str(e)}"
        if "Can't instantiate abstract class" in str(e):
            user_error_message += "\n\nThis appears to be an issue with the LLM wrapper. The application will be fixed in the next update. Please try again later."
        elif "API key" in str(e).lower():
            user_error_message += "\n\nPlease check that your API keys are correctly set in your environment variables."
        
        await cl.ErrorMessage(content=user_error_message).send()
        return

    # --- Initialize Researcher Agent ---
    try:
        mcp_configs = load_mcp_server_configs()
        agent = ResearcherAgent(
            llm_client=llm_client,
            callbacks=session_callbacks,
            mcp_server_configs=mcp_configs
        )
        await agent._setup_chains_and_tools()
        cl.user_session.set("research_agent", agent)
        logger.info("ResearcherAgent initialized and stored in user session.")
    except Exception as e:
        logger.error(f"Failed to initialize ResearcherAgent: {e}", exc_info=True)
        await cl.ErrorMessage(content=f"Failed to initialize research agent: {e}").send()
        return

    # --- Initialize ContentManager ---
    try:
        content_manager = ContentManager(
            primary_llm=llm_client,
            summarization_llm=getattr(agent, 'summarization_llm', None),
            chunk_size=getattr(agent, 'content_manager', None).splitter._chunk_size if hasattr(agent, 'content_manager') and hasattr(agent.content_manager, 'splitter') else 1000,
            chunk_overlap=getattr(agent, 'content_manager', None).splitter._chunk_overlap if hasattr(agent, 'content_manager') and hasattr(agent.content_manager, 'splitter') else 100
        )
        cl.user_session.set("content_manager", content_manager)
        logger.info("ContentManager initialized and stored in user session.")
    except Exception as e:
        logger.error(f"Failed to initialize ContentManager: {e}", exc_info=True)
        await cl.ErrorMessage(content=f"Failed to initialize content manager: {e}").send()
        return

    # --- Report Tool Status (Markdown, Grouped, with Introductions) ---
    try:
        loaded_tool_names = [tool.name for tool in agent.tools]
        mcp_configs = mcp_configs or {}
        mcp_tools_loaded = any(tool.name not in AVAILABLE_TOOLS for tool in agent.tools)
        standard_tools_loaded = any(tool.name in AVAILABLE_TOOLS for tool in agent.tools)

        # --- Build Markdown Tool Report ---
        standard_tools = {}
        mcp_tools_by_server = {}
        for tool in agent.tools:
            if tool.name not in AVAILABLE_TOOLS:
                server_name = "MCP Tools"
                if server_name not in mcp_tools_by_server:
                    mcp_tools_by_server[server_name] = []
                mcp_tools_by_server[server_name].append(tool)
            else:
                tool_class_name = tool.__class__.__name__
                if tool_class_name not in standard_tools:
                    standard_tools[tool_class_name] = tool
        tool_descriptions = []
        if standard_tools:
            tool_descriptions.append("### 🔍 Standard Tools:")
            handled_standard_tools = set()
            if "PlaywrightBrowserTool" in standard_tools:
                tool_descriptions.append("- **Web Search & Browser**: Provides web searching and page content extraction capabilities.")
                handled_standard_tools.add("PlaywrightBrowserTool")
            if "RedditSearchTool" in standard_tools or "RedditExtractPostTool" in standard_tools:
                tool_descriptions.append("- **Reddit Tools**: Allows searching Reddit and extracting post content.")
                handled_standard_tools.add("RedditSearchTool")
                handled_standard_tools.add("RedditExtractPostTool")
            for name, tool_instance in standard_tools.items():
                if name not in handled_standard_tools:
                    desc = getattr(tool_instance, 'description', 'No description').split('\n')[0]
                    tool_descriptions.append(f"- **{name}**: {desc}")
        if mcp_tools_by_server:
            tool_descriptions.append("\n### 🧩 Specialized Tools (MCP):")
            mcp_config_keys = list(mcp_configs.keys())
            loaded_mcp_tools = [tool for group in mcp_tools_by_server.values() for tool in group]
            internal_to_config = {}
            if len(loaded_mcp_tools) == len(mcp_config_keys):
                for tool, config_key in zip(loaded_mcp_tools, mcp_config_keys):
                    internal_to_config[tool.name] = config_key
            else:
                for tool in loaded_mcp_tools:
                    found = False
                    for config_key in mcp_config_keys:
                        if tool.name in config_key or config_key in tool.name:
                            internal_to_config[tool.name] = config_key
                            found = True
                            break
                    if not found:
                        internal_to_config[tool.name] = tool.name
            for server_name, tools in mcp_tools_by_server.items():
                for tool in tools:
                    mcp_tool_name = internal_to_config.get(tool.name, tool.name)
                    desc = mcp_configs.get(mcp_tool_name, {}).get('description', getattr(tool, 'description', 'No description available')).split('\n')[0]
                    internal_name = tool.name
                    tool_descriptions.append(f"- **{mcp_tool_name}**: {desc} (`{internal_name}`)")
        tool_list_str = "\n".join(tool_descriptions)
        status_content = f"### ✅ Tools ready!\n{tool_list_str}"
        if mcp_configs and not mcp_tools_loaded:
            status_content += "\n\n⚠️ _Note: Specialized MCP tools failed to load._"
        if not standard_tools_loaded:
            status_content += "\n\n⚠️ _Note: Standard tools (like Browser/Reddit) failed to load._"
        await cl.Message(content=status_content, author="System").send()
        logger.info(f"Tools loaded successfully: {loaded_tool_names}")

        # --- Agent Introductions (Markdown, Custom Authors) ---
        primary_author = getattr(ChainlitCallbackHandler, 'PRIMARY_AUTHOR', 'Head Researcher')
        summary_author = getattr(ChainlitCallbackHandler, 'SUMMARY_AUTHOR', 'Summarizer Agent')
        next_step_author = getattr(ChainlitCallbackHandler, 'NEXT_STEP_AUTHOR', 'Next Step Agent')
        # Get primary LLM model name
        primary_model_name = "Unknown Primary Model"
        if hasattr(agent, 'llm_client') and agent.llm_client:
            client = agent.llm_client
            if hasattr(client, 'model_name'):
                primary_model_name = client.model_name
            elif hasattr(client, 'model'):
                primary_model_name = client.model
            elif hasattr(client, 'model_path'):
                try:
                    primary_model_name = os.path.basename(client.model_path)
                except Exception:
                    primary_model_name = "Local Model (path error)"
            if primary_model_name and '/' in primary_model_name:
                primary_model_name = primary_model_name.split('/')[-1]
        await cl.Message(
            content=f"👩‍🔬 **Hello! I'm the Head Researcher** (`{primary_model_name}`). I'll be your primary researcher, planning and analyzing final content in detail.",
            author=primary_author
        ).send()
        # Summarizer
        if SUMMARIZER_MODEL:
            summarizer_model_name = SUMMARIZER_MODEL.split('/')[-1]
            local_tag = " (Local)" if USE_LOCAL_SUMMARIZER_MODEL else ""
            await cl.Message(
                content=f"📝 **Hi there! I'm the Summarizer Agent** (`{summarizer_model_name}`{local_tag}). I'll help with summarizing content along the way.",
                author=summary_author
            ).send()
        # Next Step
        next_step_model_name = NEXT_STEP_MODEL.split('/')[-1]
        await cl.Message(
            content=f"🧭 **Hello! I'm the Next Step Agent** (`{next_step_model_name}`). I'll help with next-step reasoning and decision making.",
            author=next_step_author
        ).send()
        # Welcome
        await cl.Message(
            content="Welcome to CleverBee! Please enter your research query.",
            author=primary_author
        ).send()
    except Exception as e:
        logger.error(f"Failed to report tool status or send introductions: {e}", exc_info=True)
        await cl.ErrorMessage(content=f"Failed to report tool status or send introductions: {e}").send()

    # --- Remove Loading Indicator ---
    try:
        loading_msg = cl.user_session.get("loading_msg")
        if loading_msg:
            await loading_msg.delete()
    except Exception:
        pass

@cl.on_message
async def handle_message(message: cl.Message):
    # Check if we're in a follow-up query after research
    if cl.user_session.get("research_completed"):
        # This is a follow-up query, present options through buttons
        topic = cl.user_session.get("research_topic")
        follow_up_query = message.content
        
        actions = [
            cl.Action(
                name="use_summary",
                label="Use Final Summary",
                description="Cheaper, but limited context",
                payload={"choice": "summary"},
                icon="sparkles"
            ),
            cl.Action(
                name="use_full_content",
                label="Use Full Content",
                description="More expensive, but deeper knowledge",
                payload={"choice": "full_content"},
                icon="library"
            )
        ]
        
        await cl.Message(
            content="""
### Before we continue analyzing your question

Please choose how you'd like me to respond:

- **Use Final Summary**: I'll analyze your question using only the research summary - faster and cheaper but with more limited context
- **Use Full Content**: I'll use all the detailed content gathered during research - more comprehensive but requires more processing

Both options include the original research topic and research plan for context.
            """,
            actions=actions,
            author="Researcher"
        ).send()
        
        # Store the follow-up query to be used when an action is selected
        cl.user_session.set("follow_up_query", follow_up_query)
        return

    # If we get here, it's a new research topic
    topic = message.content
    logger.info(f"Received research topic: {topic}")

    agent: ResearcherAgent = cl.user_session.get("research_agent")
    token_manager: TokenUsageCallbackHandler = cl.user_session.get("token_usage_handler")
    chainlit_callback: ChainlitCallbackHandler = cl.user_session.get("chainlit_callback")
    content_manager: ContentManager = cl.user_session.get("content_manager")
    llm_client = cl.user_session.get("llm_client")

    if not agent or not token_manager or not chainlit_callback or not content_manager or not llm_client:
        await cl.ErrorMessage(content="Agent components not initialized correctly. Please restart the chat.").send()
        return

    start_time = datetime.now()
    current_date = datetime.now().strftime("%Y-%m-%d")
    callbacks_for_run = [chainlit_callback, token_manager]

    processing_msg = cl.Message(content=f"🔄 Researching **'{topic}'**... Please wait.", author="Researcher")
    await processing_msg.send()
    cl.user_session.set("research_running", True)

    try:
        # Store the initial research topic
        cl.user_session.set("research_topic", topic)
        
        # The agent's run_research method returns the final summary
        summary = await agent.run_research(topic, current_date=current_date, callbacks=callbacks_for_run)
        
        # Get the accumulated content from the agent
        accumulated_content = agent.current_accumulated_content if hasattr(agent, "current_accumulated_content") else ""
        
        # Try to extract the research plan from the accumulated content
        research_plan = ""
        if accumulated_content:
            # Look for the initial plan section which typically follows a pattern
            plan_markers = [
                "### Planned Tool Calls", 
                "## Research Plan",
                "# Research Plan",
                "Research Plan"
            ]
            for marker in plan_markers:
                if marker in accumulated_content:
                    plan_start = accumulated_content.find(marker)
                    # Find the end of the plan section (next heading or end of content)
                    possible_end_markers = ["---", "##", "# ", "\n\n"]
                    plan_end = len(accumulated_content)
                    for end_marker in possible_end_markers:
                        pos = accumulated_content.find(end_marker, plan_start + len(marker))
                        if pos > -1 and pos < plan_end:
                            plan_end = pos
                    research_plan = accumulated_content[plan_start:plan_end].strip()
                    break
        
        # Store both the summary, accumulated content, and research plan for later use
        cl.user_session.set("research_summary", summary)
        cl.user_session.set("accumulated_content", accumulated_content)
        cl.user_session.set("research_plan", research_plan)
        cl.user_session.set("research_completed", True)
        
        # ---Extract and display summary and sources as-is ---
        def strip_tags(text):
            # Remove all <summary>, </summary>, <sources>, </sources> tags, anywhere in the string
            return re.sub(r'</?\s*(summary|sources)\s*>', '', text, flags=re.IGNORECASE)
        
        # Log the raw summary for debugging
        logger.info(f"Raw summary from LLM:\n{summary}")

        summary_match = re.search(r'<summary>(.*?)</summary>', summary, re.DOTALL | re.IGNORECASE)
        sources_match = re.search(r'<sources>(.*?)</sources>', summary, re.DOTALL | re.IGNORECASE)

        if not summary_match:
            logger.warning("Could not find <summary> tags in summary. Raw summary:\n" + summary)
        if not sources_match:
            logger.warning("Could not find <sources> tags in summary. Raw summary:\n" + summary)

        summary_md = strip_tags(summary_match.group(1).strip()) if summary_match else strip_tags(summary.strip())
        sources_md = strip_tags(sources_match.group(1).strip()) if sources_match else ""
        
        # Remove leading/trailing code block markers if present
        summary_md = re.sub(r"^```(?:\\w+)?\\n?", "", summary_md)  # Remove opening ```
        summary_md = re.sub(r"\\n?```$", "", summary_md)            # Remove closing ```
        sources_md = re.sub(r"^```(?:\\w+)?\\n?", "", sources_md)  # Remove opening ``` from sources
        sources_md = re.sub(r"\\n?```$", "", sources_md)            # Remove closing ``` from sources
        
        end_time = datetime.now()
        logger.info(f"Research process finished in {end_time - start_time}")
        if processing_msg:
            processing_msg.content=f"✅ Researching **'{topic}'**"
            await processing_msg.update()
        await chainlit_callback.display_final_token_summary()
        # --- Render summary and sources as clean markdown ---
        display_content = f"**Final Summary:**\n\n{summary_md}\n\n"
        if sources_md:
            display_content += f"### References\n{sources_md}\n\n"
        display_content += "---\nYou can now ask follow-up questions about this research. I'll give you the option to use either just this summary or the full detailed content I've gathered for answering your questions."
        await cl.Message(content=display_content, author="Researcher").send()
    except Exception as e:
        logger.error(f"Error during research: {e}", exc_info=True)
        if processing_msg:
            processing_msg.content=f"❌ Error during research for **'{topic}'**"
            await processing_msg.update()
        await cl.ErrorMessage(content=f"An error occurred during research: {e}").send()
    finally:
        cl.user_session.set("research_running", False)
        logger.info("Research flag cleared in user session")

@cl.action_callback("use_summary")
async def on_use_summary(action: cl.Action):
    """Handle action to use only the summary for follow-up responses."""
    from config.prompts import POST_RESEARCH_PROMPT
    from langchain_core.messages import HumanMessage
    
    # Get the stored data
    topic = cl.user_session.get("research_topic")
    summary = cl.user_session.get("research_summary")
    research_plan = cl.user_session.get("research_plan", "")
    follow_up_query = cl.user_session.get("follow_up_query")
    
    # Get the LLM client and agent
    llm_client = cl.user_session.get("llm_client")
    agent = cl.user_session.get("research_agent")
    chainlit_callback = cl.user_session.get("chainlit_callback") 
    token_manager = cl.user_session.get("token_usage_handler")
    
    if not all([topic, summary, follow_up_query, llm_client]):
        await cl.Message(content="❌ Missing data for follow-up response. Please restart the chat.", author="System").send()
        return
    
    # Format the research plan section
    research_plan_section = ""
    if research_plan:
        research_plan_section = f"<research_plan>\n{research_plan}\n</research_plan>"
    
    # Create a prompt with the summary content
    prompt = POST_RESEARCH_PROMPT.format(
        topic=topic,
        research_plan_section=research_plan_section,
        content_type="summary",
        content=summary,
        follow_up_query=follow_up_query,
        content_type_description="research summary"
    )
    
    try:
        # Get response from LLM - Using proper LangChain message format
        messages = [[HumanMessage(content=prompt)]]
        
        # Process potential tool calls and handle errors
        # Track tool calls, errors, and corrections
        tool_errors = []
        tool_corrections = []
        tool_successes = []
        
        # Create callbacks list and metadata dict separately
        callbacks = [token_manager]  # Only include token manager for now
        
        # First try to get a response
        response = await llm_client.agenerate(
            messages=messages,
            callbacks=callbacks
        )
        
        # >>> ADDED: Log LLM parameters before response processing <<<
        try:
            # Attempt to extract model parameters from the LLM client
            logger.info("LLM Configuration Parameters:")
            
            # Extract from the llm_client object itself
            if hasattr(llm_client, 'model_kwargs'):
                logger.info(f"Model kwargs: {llm_client.model_kwargs}")
            
            if hasattr(llm_client, 'model_name'):
                logger.info(f"Model name: {llm_client.model_name}")
                
            # Check for different parameter attributes that might exist
            for param_name in ['temperature', 'max_tokens', 'max_output_tokens', 'max_token_limit', 
                              'top_p', 'top_k', 'model', 'streaming']:
                if hasattr(llm_client, param_name):
                    logger.info(f"LLM parameter: {param_name}={getattr(llm_client, param_name)}")
            
            # Check for _default_params if it exists
            if hasattr(llm_client, '_default_params'):
                logger.info(f"Default params: {llm_client._default_params}")
                
            # Check invocation params from the most recent response if available
            if hasattr(response, 'llm_output') and response.llm_output and 'token_usage' in response.llm_output:
                logger.info(f"Token usage from response: {response.llm_output['token_usage']}")
                
            # If this is a Gemini model, check for specific attributes
            if 'gemini' in str(llm_client.__class__).lower() or (hasattr(llm_client, 'model_name') and 'gemini' in llm_client.model_name.lower()):
                logger.info("Detected Gemini model - checking for specific parameters")
                # Additional Gemini-specific parameters
                for param_name in ['google_api_key', 'generation_config', 'safety_settings']:
                    if hasattr(llm_client, param_name):
                        param_value = getattr(llm_client, param_name)
                        # Don't log actual API keys
                        if param_name == 'google_api_key':
                            logger.info(f"LLM has google_api_key configured: {param_value is not None}")
                        else:
                            logger.info(f"Gemini parameter: {param_name}={param_value}")
        except Exception as e:
            logger.warning(f"Failed to extract LLM parameters: {e}")
        # >>> END ADDED LLM PARAMETER LOGGING <<<
        
        response_text = response.generations[0][0].text if response and response.generations else "No response generated."
        
        # Check for tool calls in the response
        tool_calls = []
        if response and response.generations and response.generations[0]:
            generation = response.generations[0][0]
            if hasattr(generation, 'message') and hasattr(generation.message, 'tool_calls'):
                tool_calls = generation.message.tool_calls
        
        # If tool calls exist, process them
        if tool_calls:
            # Create a message to show tool usage
            tool_msg = cl.Message(content="Processing tool calls for your query...", author="Researcher")
            await tool_msg.send()
            
            # Process each tool call
            for tool_call in tool_calls:
                tool_name = tool_call.get("name", "unknown_tool")
                tool_args = tool_call.get("args", {})
                
                try:
                    # Find the tool in the agent's tools
                    tool_to_use = next((t for t in agent.tools if t.name == tool_name), None)
                    
                    if not tool_to_use:
                        raise ValueError(f"Tool '{tool_name}' not found")
                    
                    # Normalize arguments for schema validation
                    if hasattr(agent, 'normalize_tool_args'):
                        normalized_args = agent.normalize_tool_args(tool_args, tool_to_use)
                    else:
                        # Fallback if not available
                        normalized_args = tool_args
                    
                    # Execute the tool
                    tool_result = await tool_to_use.arun(
                        normalized_args, 
                        callbacks=callbacks
                    )
                    
                    # Record success
                    tool_successes.append({
                        "tool": tool_name,
                        "result": tool_result if len(str(tool_result)) < 200 else f"{str(tool_result)[:200]}..."
                    })
                    
                    # Update the messages with tool result
                    additional_msg = HumanMessage(content=f"Tool result for {tool_name}: {tool_result}")
                    messages[0].append(additional_msg)
                    
                except Exception as tool_error:
                    # Handle tool error with MCP schema correction suggestions if available
                    error_msg = str(tool_error)
                    tool_errors.append({
                        "tool": tool_name,
                        "error": error_msg
                    })
                    
                    # Try to get schema correction if available
                    correction_suggestion = None
                    if hasattr(agent, '_get_tool_correction_suggestion'):
                        try:
                            correction_suggestion = await agent._get_tool_correction_suggestion(
                                tool_name=tool_name, 
                                failed_args=tool_args,
                                error_message=error_msg
                            )
                        except Exception as corr_error:
                            logger.error(f"Error getting correction suggestion: {corr_error}")
                    
                    if correction_suggestion:
                        tool_corrections.append({
                            "tool": tool_name,
                            "original_args": tool_args,
                            "corrected_args": correction_suggestion
                        })
                        
                        # Try with corrected args
                        try:
                            tool_result = await tool_to_use.arun(
                                correction_suggestion, 
                                callbacks=callbacks
                            )
                            
                            # Record success with correction
                            tool_successes.append({
                                "tool": tool_name,
                                "result": tool_result if len(str(tool_result)) < 200 else f"{str(tool_result)[:200]}...",
                                "corrected": True
                            })
                            
                            # Update the messages with tool result
                            additional_msg = HumanMessage(content=f"Tool result for {tool_name} (after correction): {tool_result}")
                            messages[0].append(additional_msg)
                            
                        except Exception as retry_error:
                            # Still failed after correction
                            error_msg = str(retry_error)
                            tool_errors.append({
                                "tool": tool_name,
                                "error": f"Still failed after correction: {error_msg}",
                                "attempted_correction": True
                            })
                            
                            # Add error message to context
                            additional_msg = HumanMessage(content=f"Tool {tool_name} failed: {error_msg}")
                            messages[0].append(additional_msg)
            
            # Remove the tools message
            await tool_msg.remove()
            
            # Final regeneration with all tool results or errors
            if tool_successes or tool_errors:
                # Get final response with tool results incorporated
                final_response = await llm_client.agenerate(
                    messages=messages,
                    callbacks=callbacks
                )
                
                response_text = final_response.generations[0][0].text if final_response and final_response.generations else response_text
        
        # Send response
        # >>> ADDED: Log the first and last parts of the response to avoid log truncation <<<
        logger.info(f"Attempting to send full content response (length: {len(response_text)} chars).")
        preview_length = 500  # Characters to show at start and end
        if len(response_text) > preview_length * 2:
            logger.info(f"Response content preview (first/last {preview_length} chars):")
            logger.info(f"START: {response_text[:preview_length]}...")
            logger.info(f"...END: {response_text[-preview_length:]}")
        else:
            logger.info(f"Full response content:\n{response_text}")
        # >>> END LOGGING CHANGES <<<
        
        # Send the response as a single message
        await cl.Message(content=response_text, author="Researcher").send()
        
        # Show tool usage summary if any tools were used
        if tool_successes or tool_errors:
            summary_parts = []
            
            if tool_successes:
                summary_parts.append("**Tools Successfully Used:**")
                for success in tool_successes:
                    corrected = " (after correction)" if success.get("corrected") else ""
                    summary_parts.append(f"- {success['tool']}{corrected}")
            
            if tool_errors and not all(error.get("attempted_correction") for error in tool_errors):
                summary_parts.append("\n**Tool Errors:**")
                for error in tool_errors:
                    if not error.get("attempted_correction"):
                        summary_parts.append(f"- {error['tool']}: {error['error']}")
            
            if summary_parts:
                await cl.Message(
                    content="\n".join(summary_parts),
                    author="System"
                ).send()
                
    except Exception as e:
        logger.error(f"Error generating follow-up response: {e}", exc_info=True)
        await cl.Message(content=f"❌ Error analyzing your question: {e}", author="Researcher").send()

@cl.action_callback("use_full_content")
async def on_use_full_content(action: cl.Action):
    """Handle action to use the full accumulated content for follow-up responses."""
    from config.prompts import POST_RESEARCH_PROMPT
    from langchain_core.messages import HumanMessage
    
    # Get the stored data
    topic = cl.user_session.get("research_topic")
    accumulated_content = cl.user_session.get("accumulated_content")
    research_plan = cl.user_session.get("research_plan", "")
    follow_up_query = cl.user_session.get("follow_up_query")
    
    # Get the LLM client and agent
    llm_client = cl.user_session.get("llm_client")
    agent = cl.user_session.get("research_agent")
    chainlit_callback = cl.user_session.get("chainlit_callback")
    token_manager = cl.user_session.get("token_usage_handler")
    
    if not all([topic, accumulated_content, follow_up_query, llm_client]):
        await cl.Message(content="❌ Missing data for follow-up response. Please restart the chat.", author="System").send()
        return
    
    # Format the research plan section
    research_plan_section = ""
    if research_plan:
        research_plan_section = f"<research_plan>\n{research_plan}\n</research_plan>"
    
    # Create a prompt with the full accumulated content
    prompt = POST_RESEARCH_PROMPT.format(
        topic=topic,
        research_plan_section=research_plan_section,
        content_type="full_content",
        content=accumulated_content,
        follow_up_query=follow_up_query,
        content_type_description="full research content"
    )
    
    try:
        # Get response from LLM - Using proper LangChain message format
        messages = [[HumanMessage(content=prompt)]]
        
        # Process potential tool calls and handle errors
        # Track tool calls, errors, and corrections
        tool_errors = []
        tool_corrections = []
        tool_successes = []
        
        # Create callbacks list and metadata dict separately
        callbacks = [token_manager]  # Only include token manager for now
        
        # First try to get a response
        response = await llm_client.agenerate(
            messages=messages,
            callbacks=callbacks
        )
        
        # >>> ADDED: Log LLM parameters before response processing <<<
        try:
            # Attempt to extract model parameters from the LLM client
            logger.info("LLM Configuration Parameters:")
            
            # Extract from the llm_client object itself
            if hasattr(llm_client, 'model_kwargs'):
                logger.info(f"Model kwargs: {llm_client.model_kwargs}")
            
            if hasattr(llm_client, 'model_name'):
                logger.info(f"Model name: {llm_client.model_name}")
                
            # Check for different parameter attributes that might exist
            for param_name in ['temperature', 'max_tokens', 'max_output_tokens', 'max_token_limit', 
                              'top_p', 'top_k', 'model', 'streaming']:
                if hasattr(llm_client, param_name):
                    logger.info(f"LLM parameter: {param_name}={getattr(llm_client, param_name)}")
            
            # Check for _default_params if it exists
            if hasattr(llm_client, '_default_params'):
                logger.info(f"Default params: {llm_client._default_params}")
                
            # Check invocation params from the most recent response if available
            if hasattr(response, 'llm_output') and response.llm_output and 'token_usage' in response.llm_output:
                logger.info(f"Token usage from response: {response.llm_output['token_usage']}")
                
            # If this is a Gemini model, check for specific attributes
            if 'gemini' in str(llm_client.__class__).lower() or (hasattr(llm_client, 'model_name') and 'gemini' in llm_client.model_name.lower()):
                logger.info("Detected Gemini model - checking for specific parameters")
                # Additional Gemini-specific parameters
                for param_name in ['google_api_key', 'generation_config', 'safety_settings']:
                    if hasattr(llm_client, param_name):
                        param_value = getattr(llm_client, param_name)
                        # Don't log actual API keys
                        if param_name == 'google_api_key':
                            logger.info(f"LLM has google_api_key configured: {param_value is not None}")
                        else:
                            logger.info(f"Gemini parameter: {param_name}={param_value}")
        except Exception as e:
            logger.warning(f"Failed to extract LLM parameters: {e}")
        # >>> END ADDED LLM PARAMETER LOGGING <<<
        
        response_text = response.generations[0][0].text if response and response.generations else "No response generated."
        
        # Check for tool calls in the response
        tool_calls = []
        if response and response.generations and response.generations[0]:
            generation = response.generations[0][0]
            if hasattr(generation, 'message') and hasattr(generation.message, 'tool_calls'):
                tool_calls = generation.message.tool_calls
        
        # If tool calls exist, process them
        if tool_calls:
            # Create a message to show tool usage
            tool_msg = cl.Message(content="Processing tool calls for your query...", author="Researcher")
            await tool_msg.send()
            
            # Process each tool call
            for tool_call in tool_calls:
                tool_name = tool_call.get("name", "unknown_tool")
                tool_args = tool_call.get("args", {})
                
                try:
                    # Find the tool in the agent's tools
                    tool_to_use = next((t for t in agent.tools if t.name == tool_name), None)
                    
                    if not tool_to_use:
                        raise ValueError(f"Tool '{tool_name}' not found")
                    
                    # Normalize arguments for schema validation
                    if hasattr(agent, 'normalize_tool_args'):
                        normalized_args = agent.normalize_tool_args(tool_args, tool_to_use)
                    else:
                        # Fallback if not available
                        normalized_args = tool_args
                    
                    # Execute the tool
                    tool_result = await tool_to_use.arun(
                        normalized_args, 
                        callbacks=callbacks
                    )
                    
                    # Record success
                    tool_successes.append({
                        "tool": tool_name,
                        "result": tool_result if len(str(tool_result)) < 200 else f"{str(tool_result)[:200]}..."
                    })
                    
                    # Update the messages with tool result
                    additional_msg = HumanMessage(content=f"Tool result for {tool_name}: {tool_result}")
                    messages[0].append(additional_msg)
                    
                except Exception as tool_error:
                    # Handle tool error with MCP schema correction suggestions if available
                    error_msg = str(tool_error)
                    tool_errors.append({
                        "tool": tool_name,
                        "error": error_msg
                    })
                    
                    # Try to get schema correction if available
                    correction_suggestion = None
                    if hasattr(agent, '_get_tool_correction_suggestion'):
                        try:
                            correction_suggestion = await agent._get_tool_correction_suggestion(
                                tool_name=tool_name, 
                                failed_args=tool_args,
                                error_message=error_msg
                            )
                        except Exception as corr_error:
                            logger.error(f"Error getting correction suggestion: {corr_error}")
                    
                    if correction_suggestion:
                        tool_corrections.append({
                            "tool": tool_name,
                            "original_args": tool_args,
                            "corrected_args": correction_suggestion
                        })
                        
                        # Try with corrected args
                        try:
                            tool_result = await tool_to_use.arun(
                                correction_suggestion, 
                                callbacks=callbacks
                            )
                            
                            # Record success with correction
                            tool_successes.append({
                                "tool": tool_name,
                                "result": tool_result if len(str(tool_result)) < 200 else f"{str(tool_result)[:200]}...",
                                "corrected": True
                            })
                            
                            # Update the messages with tool result
                            additional_msg = HumanMessage(content=f"Tool result for {tool_name} (after correction): {tool_result}")
                            messages[0].append(additional_msg)
                            
                        except Exception as retry_error:
                            # Still failed after correction
                            error_msg = str(retry_error)
                            tool_errors.append({
                                "tool": tool_name,
                                "error": f"Still failed after correction: {error_msg}",
                                "attempted_correction": True
                            })
                            
                            # Add error message to context
                            additional_msg = HumanMessage(content=f"Tool {tool_name} failed: {error_msg}")
                            messages[0].append(additional_msg)
            
            # Remove the tools message
            await tool_msg.remove()
            
            # Final regeneration with all tool results or errors
            if tool_successes or tool_errors:
                # Get final response with tool results incorporated
                final_response = await llm_client.agenerate(
                    messages=messages,
                    callbacks=callbacks
                )
                
                response_text = final_response.generations[0][0].text if final_response and final_response.generations else response_text
        
        # Send response
        # >>> ADDED: Log the first and last parts of the response to avoid log truncation <<<
        logger.info(f"Attempting to send full content response (length: {len(response_text)} chars).")
        preview_length = 500  # Characters to show at start and end
        if len(response_text) > preview_length * 2:
            logger.info(f"Response content preview (first/last {preview_length} chars):")
            logger.info(f"START: {response_text[:preview_length]}...")
            logger.info(f"...END: {response_text[-preview_length:]}")
        else:
            logger.info(f"Full response content:\n{response_text}")
        # >>> END LOGGING CHANGES <<<
        
        # Send the response as a single message
        await cl.Message(content=response_text, author="Researcher").send()
        
        # Show tool usage summary if any tools were used
        if tool_successes or tool_errors:
            summary_parts = []
            
            if tool_successes:
                summary_parts.append("**Tools Successfully Used:**")
                for success in tool_successes:
                    corrected = " (after correction)" if success.get("corrected") else ""
                    summary_parts.append(f"- {success['tool']}{corrected}")
            
            if tool_errors and not all(error.get("attempted_correction") for error in tool_errors):
                summary_parts.append("\n**Tool Errors:**")
                for error in tool_errors:
                    if not error.get("attempted_correction"):
                        summary_parts.append(f"- {error['tool']}: {error['error']}")
            
            if summary_parts:
                await cl.Message(
                    content="\n".join(summary_parts),
                    author="System"
                ).send()
                
    except Exception as e:
        logger.error(f"Error generating follow-up response: {e}", exc_info=True)
        await cl.Message(content=f"❌ Error analyzing your question: {e}", author="Researcher").send()

@cl.on_chat_end
async def end_chat():
    logger.info("Chat session ended. Cleaning up resources...")
    is_research_running = cl.user_session.get("research_running", False)
    agent: ResearcherAgent = cl.user_session.get("research_agent")
    if not is_research_running:
        try:
            logger.info("Research not running, closing browser pages but keeping browser alive...")
            await browser_manager.close_all_pages()
            logger.info("Browser pages closed successfully.")
        except Exception as e:
            logger.error(f"Error closing browser pages: {e}", exc_info=True)
    else:
        logger.info("Research still running, keeping browser and pages alive for ongoing operations.")
    if agent:
        logger.info("Cleaning up agent resources...")
        # Any additional agent-specific cleanup if needed

# Placeholder for running logic if needed directly (usually run via `chainlit run`)
# if __name__ == "__main__":
#     # This part is typically not needed as Chainlit CLI handles running the app
#     pass 
