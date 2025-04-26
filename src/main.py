import argparse
import asyncio
import logging
import os
import sys
from datetime import datetime
import time

# Add project root to the Python path to allow importing 'config'
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Set up basic logging configuration initially
# We will set the effective level after loading config
logging.basicConfig(format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s', stream=sys.stdout)
logger = logging.getLogger(__name__)

# Now import other modules that might use logging
from dotenv import load_dotenv
# from src.agent import ResearcherAgent # Import from old agent file (REMOVE)
from src.agent.researcher_agent import ResearcherAgent # Import from NEW agent file
# Remove direct client imports
# from src.llm_clients.claude import ClaudeClient 
# from src.llm_clients.gemini import GeminiClient
# Import the factory function instead
from src.llm_clients.factory import get_llm_client

import config.settings # This loads the config
# Import specific settings we need for client selection
from config.settings import (
    # Common Settings
    LOG_LEVEL,
    LOCAL_MODELS_DIR,
    USE_LOCAL_SUMMARIZER_MODEL,
    SUMMARIZER_MODEL,
    SUMMARY_MAX_TOKENS,
    # LLM Provider Settings
    PRIMARY_MODEL_TYPE,  # Use PRIMARY_MODEL_TYPE instead of MAIN_LLM_PROVIDER
    CLAUDE_MODEL_NAME,
    GEMINI_MODEL_NAME,
    LOCAL_MODEL_NAME,  # Add LOCAL_MODEL_NAME
    # Content Processing Settings
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    # Tool Configuration
    TOOLS_CONFIG,
    # Memory Settings
    CONVERSATION_MEMORY_MAX_TOKENS,
    # Token Usage Settings
    TRACK_TOKEN_USAGE,
    LOG_COST_SUMMARY,
    GEMINI_API_KEY,
    ANTHROPIC_API_KEY,
    ENABLE_THINKING, # Import setting
    THINKING_BUDGET,  # Import setting
    # Cache Settings
    ENABLE_ADVANCED_CACHE,
    CACHE_DB_PATH,
    CACHE_SCHEMA,
)

# --- Apply Log Level from Config --- 
# Get the root logger and set the level based on the loaded config
log_level_str = getattr(config.settings, 'LOG_LEVEL', 'INFO').upper()
log_level = getattr(logging, log_level_str, logging.INFO)
logging.getLogger().setLevel(log_level) 
logger.info(f"Logging level set to: {log_level_str} ({log_level})")
# ----------------------------------

# Import browser tool AFTER setting log level potentially?
# Might affect browser tool logging level if it initializes its own logger early.
from src.browser import PlaywrightBrowserTool
# Import for caching and token tracking
from langchain.globals import set_llm_cache
# Import caching options
from langchain_community.cache import InMemoryCache
from src.advanced_cache import initialize_normalizing_cache, NormalizingCache
# MODIFIED: Import TokenCostProcess and TokenUsageCallbackHandler directly
from src.token_callback import TokenCostProcess, TokenUsageCallbackHandler
from src.cache_monitor import initialize_cache_monitoring

# Initialize LangChain caching with advanced normalizing cache to improve performance and reduce costs
cache_db_path = os.path.join(project_root, CACHE_DB_PATH)
logger.info(f"Setting up LLM cache with database at {cache_db_path}")

if ENABLE_ADVANCED_CACHE:
    logger.info("Initializing advanced normalizing cache for better hit rates")
    llm_cache = initialize_normalizing_cache(db_path=cache_db_path)
    set_llm_cache(llm_cache)
    import langchain
    langchain.llm_cache = llm_cache  # Ensure global is set for monitor
else:
    logger.info("Using basic in-memory cache (no normalization)")
    llm_cache = InMemoryCache()
    set_llm_cache(llm_cache)
    import langchain
    langchain.llm_cache = llm_cache

# ADDED: Create the central token cost processor
token_cost_processor = TokenCostProcess()
logger.info("Created central TokenCostProcess instance.")

# Initialize cache monitoring to track token usage from cache hits
# MODIFIED: Pass the processor instance
cache_monitor = initialize_cache_monitoring(token_cost_processor=token_cost_processor)
logger.info(f"[DIAGNOSTIC] Logging level: {logging.getLogger().getEffectiveLevel()} ({logging.getLevelName(logging.getLogger().getEffectiveLevel())})")
logger.info(f"[DIAGNOSTIC] Cache instance: {llm_cache}")
logger.info(f"[DIAGNOSTIC] Cache monitor initialized: {cache_monitor is not None}")
if cache_monitor:
    logger.info(f"[DIAGNOSTIC] Cache monitor details: {cache_monitor}")
print("Reached cache monitor init")

# ADDED: Create the single callback handler instance
token_usage_handler = TokenUsageCallbackHandler(token_cost_processor=token_cost_processor)
logger.info("Created central TokenUsageCallbackHandler instance.")

async def run_agent(topic: str):
    """Initializes and runs the research agent.
    
    Args:
        topic: The research topic to investigate
        
    Returns:
        int: Status code (0 for success, non-zero for failure)
    """
    logger.info(f"Starting research for topic: '{topic}'")
    
    # Prepare callbacks list using the single handler instance
    callbacks = [token_usage_handler] # Use the instance created above

    # --- Use LLM Client Factory --- 
    logger.info(f"Using {PRIMARY_MODEL_TYPE} provider via factory.")
    llm_client = get_llm_client(
        provider=PRIMARY_MODEL_TYPE,
        callbacks=callbacks,  # Pass callbacks to the factory
        use_retry_wrapper=True,  # Enable retry functionality
        max_retries=3  # Set maximum retries to 3
    )

    # Instantiate the agent with the selected client
    # Use settings directly
    agent = ResearcherAgent(
        llm_client=llm_client,
        callbacks=callbacks # Pass the single handler instance
    )

    try:
        # Pass the prepared callbacks list to the agent's run method
        # No need to pass thinking flags here
        current_date = datetime.now().strftime("%Y-%m-%d")
        summary = await agent.run_research(topic, current_date=current_date, callbacks=callbacks) # Pass current date and callbacks

        # Log token usage stats directly from the single handler instance
        logger.info(f"Research complete. Token usage summary:")
        # Use the processor within the handler for summary data
        token_usage_handler.token_cost_processor.log_summary()

        # Log cache statistics if available
        if isinstance(llm_cache, NormalizingCache):
            logger.info(llm_cache.print_stats())

        # Print the research summary
        print(f"\n{'='*20} Research Summary {'='*20}")
        print(f"Topic: {topic}\n")
        print(summary)
        print(f"{'='*58}\n")

        # Close browser resources explicitly after agent run
        # Find the browser tool instance if it was loaded
        browser_tool_instance = next((tool for tool in agent.tools if isinstance(tool, PlaywrightBrowserTool)), None)
        if browser_tool_instance:
             logger.info("Cleaning up browser resources...")
             await browser_tool_instance.clean_up()
             logger.info("Browser resources cleaned up.")

        return 0  # Success
    except Exception as e:
        logger.error(f"Error during research run: {e}", exc_info=True)
        
        # Clean up browser resources even on failure
        try:
            browser_tool_instance = next((tool for tool in agent.tools if isinstance(tool, PlaywrightBrowserTool)), None)
            if browser_tool_instance:
                logger.info("Cleaning up browser resources after error...")
                await browser_tool_instance.clean_up()
                logger.info("Browser resources cleaned up after error.")
        except Exception as cleanup_error:
            logger.error(f"Error during browser cleanup: {cleanup_error}")
            
        return 1  # Failure

def setup_agent():
    """Set up the research agent with proper LLM client via factory."""
    try:
        # Prepare callbacks list for LLM calls during setup
        callbacks = [token_usage_handler]

        # --- Use LLM Client Factory for Setup --- 
        logger.info(f"Using {PRIMARY_MODEL_TYPE} provider via factory for agent setup.")
        llm_client = get_llm_client(
            provider=PRIMARY_MODEL_TYPE,
            callbacks=callbacks,  # Pass callbacks to the factory
            use_retry_wrapper=True,  # Enable retry functionality
            max_retries=3  # Set maximum retries to 3
        )

        # Get topic from args
        parser = argparse.ArgumentParser(description="Run a research assistant for the specified topic.")
        parser.add_argument("--topic", required=True, help="The topic to research")
        args = parser.parse_args()
        
        # Create agent using settings loaded from config
        agent = ResearcherAgent(
            llm_client=llm_client,
            callbacks=callbacks
        )
        
        return agent, args.topic
    except (ValueError, RuntimeError) as client_error:
        logger.error(f"Failed to set up agent client using factory: {client_error}", exc_info=True)
        raise # Re-raise error during setup
    except Exception as e:
        logger.error(f"Failed to set up agent: {e}", exc_info=True)
        raise

def main():
    # Load environment variables from .env file
    load_dotenv()
    logger.info("Loaded environment variables from .env file.")

    try:
        agent, topic = setup_agent()
        
        # Add current year to topic if no year is mentioned for time relevance
        current_year = datetime.now().year
        
        # Check if the topic already contains a year (2020-current_year)
        if not any(str(year) in topic for year in range(2020, current_year+1)):
            original_topic = topic
            topic = f"{topic} {current_year}"
            logger.info(f"Added current year to research topic: '{original_topic}' → '{topic}'")
        
        # Run agent with retry logic for critical errors
        max_retries = 2  # Maximum number of retries for the entire research pipeline
        retry_count = 0
        last_error = None
        
        while retry_count <= max_retries:
            try:
                return_code = asyncio.run(run_agent(topic))
                # If we get here, it was successful
                return return_code
            except KeyboardInterrupt:
                logger.info("Process interrupted by user.")
                return 1
            except Exception as e:
                last_error = e
                retry_count += 1
                
                # Check if this is a known critical error that might benefit from a retry
                error_str = str(e)
                retryable_errors = [
                    "contents.parts must not be empty",
                    "GenerateContentRequest.contents",
                    "Invalid argument provided to Gemini",
                    "timeout", 
                    "Timeout",
                    "connection",
                    "server error",
                    "Server error"
                ]
                
                # Special handling for Gemini empty parts error
                if "contents.parts must not be empty" in error_str or "GenerateContentRequest.contents" in error_str:
                    logger.error("Detected Gemini API empty content error. This is usually due to an empty message in the conversation history.")
                    if retry_count <= max_retries:
                        # Clear cache to force regeneration of conversation
                        logger.info("Attempting to clear cache to potentially fix the error...")
                        if llm_cache:
                            try:
                                llm_cache.clear()
                                logger.info("Cache cleared successfully.")
                            except Exception as cache_error:
                                logger.warning(f"Failed to clear cache: {cache_error}")
                        
                        retry_delay = 10  # Longer delay for this specific error
                        logger.error(
                            f"Retrying with fresh conversation in {retry_delay} seconds "
                            f"(attempt {retry_count}/{max_retries})..."
                        )
                        time.sleep(retry_delay)
                    else:
                        logger.error("Max retries exceeded for Gemini API empty content error.")
                        return 1
                
                elif any(err in error_str for err in retryable_errors) and retry_count <= max_retries:
                    retry_delay = 5 * retry_count  # Increasing delay between retries
                    logger.error(
                        f"Encountered retryable error: {e}. "
                        f"Retrying entire research pipeline in {retry_delay} seconds "
                        f"(attempt {retry_count}/{max_retries})..."
                    )
                    time.sleep(retry_delay)
                else:
                    # Non-retryable error or max retries exceeded
                    logger.error(
                        f"Unhandled exception in main: {e}. "
                        f"No more retries (attempt {retry_count}/{max_retries})", 
                        exc_info=True
                    )
                    return 1
        
        # If we've exhausted all retries
        logger.error(f"Failed after {max_retries} retries. Last error: {last_error}")
        return 1
    except Exception as e:
        logger.error(f"Unhandled exception in main: {e}", exc_info=True)
        return 1
    finally:
        logger.info("Research process finished.")

if __name__ == "__main__":
    sys.exit(main()) 