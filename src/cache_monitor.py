import logging
import json
import sqlite3
from typing import Any, Dict, List, Optional, Union, Callable
import functools
import re
import time
import hashlib

from langchain.schema import Generation, LLMResult
# from langchain.cache import SQLiteCache
from langchain_community.cache import SQLiteCache
import langchain

from src.token_callback import TokenUsageCallbackHandler
from src.advanced_cache import NormalizingCache
from config.settings import TRACK_TOKEN_USAGE
from src.token_callback import TokenCostProcess # Import from correct location

# --- Import Tiktoken helper --- 
from src.browser import get_token_count_for_text
# --------------------------

logger = logging.getLogger(__name__)

class CacheMonitorWrapper:
    """Wrapper for LangChain's cache to monitor and track cache usage."""
    
    def __init__(self, token_cost_processor: TokenCostProcess, db_path: str = ".langchain.db"):
        """Initialize the cache monitor wrapper.
        
        Args:
            token_cost_processor: The instance managing token costs.
            db_path: Path to the SQLite database used by LangChain for caching.
        """
        self.db_path = db_path
        # Use the passed-in token cost processor
        if token_cost_processor:
            self.token_handler = token_cost_processor # Directly use the passed processor
        else:
            logger.error("No token_cost_processor provided to CacheMonitorWrapper. Cache hit cost tracking will fail.")
            self.token_handler = None # Set to None to avoid AttributeError later
        
        # --- Improved Cache Detection and Wrapping ---
        cache_instance = langchain.llm_cache
        if cache_instance is None:
            logger.warning("LangChain LLM cache is not set. Cache token monitoring will be disabled.")
            self._original_lookup = None # Ensure no wrapping occurs
            return

        logger.info(f"Attempting to wrap cache instance: {type(cache_instance).__name__}")
        
        if isinstance(cache_instance, NormalizingCache):
            logger.info("Detected NormalizingCache, wrapping its lookup method.")
            # Access the original lookup method saved by NormalizingCache itself
            if hasattr(cache_instance, '_original_lookup') and callable(cache_instance._original_lookup):
                self._original_lookup = cache_instance._original_lookup
                # Monkey patch the main lookup method of the NormalizingCache instance
                cache_instance.lookup = self._wrapped_lookup
                logger.info("Successfully wrapped NormalizingCache.lookup for token monitoring.")
            else:
                logger.error("NormalizingCache detected, but its _original_lookup method is missing or not callable. Cannot wrap.")
                self._original_lookup = None
        elif isinstance(cache_instance, SQLiteCache):
            logger.info("Detected standard SQLiteCache, wrapping its lookup method.")
            self._original_lookup = cache_instance.lookup
            # Monkey patch the lookup method
            cache_instance.lookup = self._wrapped_lookup
            logger.info("Successfully wrapped SQLiteCache.lookup for token monitoring.")
        else:
            logger.warning(f"LangChain cache is not NormalizingCache or SQLiteCache (found: {type(cache_instance).__name__}). Cache token tracking might not work as expected.")
            # Attempt to wrap the standard lookup attribute if it exists
            if hasattr(cache_instance, 'lookup') and callable(cache_instance.lookup):
                self._original_lookup = cache_instance.lookup
                cache_instance.lookup = self._wrapped_lookup
                logger.info(f"Attempted to wrap lookup method for unknown cache type: {type(cache_instance).__name__}")
            else:
                logger.warning(f"Could not find a callable 'lookup' method on cache type {type(cache_instance).__name__}. Cache monitoring disabled.")
                self._original_lookup = None
        # ------------------------------------------
    
    def _prompt_hash_and_preview(self, prompt: str) -> str:
        h = hashlib.sha256(prompt.encode()).hexdigest()[:8]
        preview = prompt[:80].replace('\n', ' ')
        return f"hash={h}, preview=\"{preview}\""
    
    def _wrapped_lookup(self, *args, **kwargs):
        print("[DEBUG] CacheMonitorWrapper._wrapped_lookup called")
        """Wrapped lookup method that tracks token usage for cache hits."""
        prompt = args[0] if args else kwargs.get("prompt")
        llm_string = args[1] if len(args) > 1 else kwargs.get("llm_string")
        log_id = self._prompt_hash_and_preview(prompt)
        logger.info(f"[CACHE] Lookup: model={llm_string}, {log_id}")
        result = self._original_lookup(*args, **kwargs)
        if result is not None:
            logger.info(f"[CACHE] HIT: model={llm_string}, {log_id}")
            self._record_cache_hit(prompt, llm_string, result)
        else:
            logger.info(f"[CACHE] MISS: model={llm_string}, {log_id}")
        return result
    
    def _record_cache_hit(self, prompt: str, llm_string: str, result: List[Generation]):
        """Record cache hit statistics, including estimating token counts."""
        try:
            # --- Simplified Model Name Extraction --- 
            # Extract model name from llm_string more reliably
            # We rely on the format used by LangChain typically: includes model name
            model_name = "unknown"
            if llm_string:
                 # Basic extraction - might need refinement based on actual llm_string format
                 # Example: <lc_kwargs={... 'model_name': 'claude-3-7-sonnet-20240219' ...}> -> claude-3-7-sonnet-20240219
                 match = re.search(r"model(?:_name)?['\"]?:\s*['\"]?([\w\d\.\-_/]+)['\"]?", llm_string, re.IGNORECASE)
                 if match:
                      model_name = match.group(1)
                      logger.debug(f"Extracted model_name '{model_name}' from llm_string for cache hit.")
                 else:
                      logger.warning(f"Could not extract model_name from llm_string for cache hit: {llm_string}")
            # ----------------------------------------
            
            # Try to extract token usage from the cached entry
            input_tokens, output_tokens = self._extract_token_info(prompt, result, llm_string)
            
            log_id = self._prompt_hash_and_preview(prompt)
            logger.info(f"[CACHE] HIT TOKENS: model={model_name}, {log_id}, input_tokens={input_tokens}, output_tokens={output_tokens}")
            
            # Record the cache hit in our token handler
            if self.token_handler:
                self.token_handler.record_cache_hit(
                    model_name=model_name, 
                    input_tokens=input_tokens,
                    output_tokens=output_tokens
                )
                logger.debug(f"Recorded cache hit for model '{model_name}' with {input_tokens} input and {output_tokens} output tokens")
            else:
                logger.warning(f"Cache hit detected for model '{model_name}', but token handler is not available. Cannot record cost savings.")
            
        except Exception as e:
            logger.warning(f"Error recording cache hit: {e}")
    
    def _extract_model_name(self, llm_string: str) -> str:
        """Extract model name from LLM string, identifying common cloud providers and local models."""
        llm_string_lower = llm_string.lower()
        
        # Simplified - relying on llm_string parsing in _record_cache_hit
        # Basic check for keywords as a fallback
        if "claude" in llm_string_lower: return "claude" # Fallback guess
        if "gemini" in llm_string_lower: return "gemini" # Fallback guess
        if "llama" in llm_string_lower: return "local-llama" # Fallback guess
        if "mistral" in llm_string_lower: return "local-mistral" # Fallback guess
        
        # Check for local model patterns
        # Add more specific patterns if known (e.g., check for specific base paths)
        elif ".gguf" in llm_string_lower or "/" in llm_string: # Check for common suffix or path separator
             return "local-model"
        
        # Last resort fallback
        logger.warning(f"Could not determine model type from llm_string: {llm_string}. Classifying as 'unknown'.")
        return "unknown"
    
    def _extract_token_info(self, prompt: str, result: List[Generation], llm_string: str) -> tuple[int, int]:
        """Extract token usage information from cached result."""
        input_tokens = 0
        output_tokens = 0
        
        # Try to extract tokens from SQLite DB if possible
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Query for token usage metadata
            cursor.execute(
                "SELECT value FROM langchain_cache_metadata WHERE prompt_key=? AND llm_string=? AND key LIKE 'token_usage_%'",
                (prompt, llm_string)
            )
            
            rows = cursor.fetchall()
            for row in rows:
                try:
                    metadata = json.loads(row[0])
                    if 'prompt_tokens' in metadata:
                        input_tokens = metadata['prompt_tokens']
                    if 'input_tokens' in metadata:
                        input_tokens = metadata['input_tokens']
                    if 'completion_tokens' in metadata:
                        output_tokens = metadata['completion_tokens']
                    if 'output_tokens' in metadata:
                        output_tokens = metadata['output_tokens']
                    
                    # If we found token info, return it
                    if input_tokens > 0 or output_tokens > 0:
                        return input_tokens, output_tokens
                        
                except (json.JSONDecodeError, KeyError) as e:
                    logger.debug(f"Error processing token metadata: {e}")
                    continue
        
        except sqlite3.Error as e:
            logger.debug(f"SQLite error: {e}")
        finally:
            if conn:
                conn.close()
        
        # If we couldn't get data from cache metadata, make a rough estimate using tiktoken
        if input_tokens == 0:
            # Use helper function for tiktoken estimation
            input_tokens = get_token_count_for_text(prompt)
        
        if output_tokens == 0 and result:
            # Estimate output tokens using tiktoken
            total_output_text = "".join(gen.text for gen in result if hasattr(gen, 'text'))
            output_tokens = get_token_count_for_text(total_output_text)
        
        return input_tokens, output_tokens

    @staticmethod
    def extract_generations_metadata(generations: List[Generation]) -> Dict[str, Any]:
        """Extract metadata from generations to aid in recording cache hits."""
        metadata = {}
        for gen in generations:
            if hasattr(gen, 'generation_info') and gen.generation_info:
                if 'token_usage' in gen.generation_info:
                    metadata['token_usage'] = gen.generation_info['token_usage']
            
            # For more modern LangChain structures with AIMessage
            if hasattr(gen, 'message'):
                if hasattr(gen.message, 'usage_metadata') and gen.message.usage_metadata:
                    metadata['usage_metadata'] = gen.message.usage_metadata
        
        return metadata

def initialize_cache_monitoring(token_cost_processor: TokenCostProcess):
    """Initialize cache monitoring for the application.

    Args:
        token_cost_processor: The instance managing token costs.
    """
    try:
        # Pass the processor to the wrapper
        cache_monitor = CacheMonitorWrapper(token_cost_processor=token_cost_processor)
        logger.info("Cache monitoring initialized successfully")
        return cache_monitor
    except Exception as e:
        logger.error(f"Failed to initialize cache monitoring: {e}")
        return None 