import logging
import time  # Add if needed
from typing import Dict, Any, List, Optional, Union, Tuple
import asyncio
from uuid import UUID
import re

from langchain_core.callbacks import AsyncCallbackHandler, BaseCallbackHandler
from langchain_core.outputs import LLMResult

from config.settings import (
    TRACK_TOKEN_USAGE,
    LOG_COST_SUMMARY,
    CLAUDE_COST_PER_1K_INPUT_TOKENS,
    CLAUDE_COST_PER_1K_OUTPUT_TOKENS,
    GEMINI_COST_PER_1K_INPUT_TOKENS,
    GEMINI_COST_PER_1K_OUTPUT_TOKENS,
    GEMINI_FLASH_COST_PER_1K_INPUT,
    GEMINI_FLASH_COST_PER_1K_OUTPUT,
    GEMINI_25_FLASH_PREVIEW_COST_PER_1K_INPUT,
    GEMINI_25_FLASH_PREVIEW_COST_PER_1K_OUTPUT,
    CLAUDE_MODEL_NAME,
    GEMINI_MODEL_NAME,
    SUMMARIZER_MODEL,
    USE_LOCAL_SUMMARIZER_MODEL,
    PRIMARY_MODEL_NAME,
    PRIMARY_MODEL_TYPE,
    NEXT_STEP_MODEL
)

logger = logging.getLogger(__name__)
logger.info("Executing src/token_callback.py module level code...")

# Helper function to check if a model is Gemini
def is_gemini_model(model_name: str) -> bool:
    """Check if the model is a Gemini model based on name."""
    model_name = model_name.lower()
    return 'gemini' in model_name

# Mapping of model types to their pricing configurations
MODEL_PRICING = {
    # Claude models
    "claude": {
        "input_cost_per_1k": CLAUDE_COST_PER_1K_INPUT_TOKENS,
        "output_cost_per_1k": CLAUDE_COST_PER_1K_OUTPUT_TOKENS
    },
    # Main Gemini model
    "gemini_main": {
        "input_cost_per_1k": GEMINI_COST_PER_1K_INPUT_TOKENS,
        "output_cost_per_1k": GEMINI_COST_PER_1K_OUTPUT_TOKENS
    }
    # Local models handled in _calculate_cost
}

# Global dictionary to store token usage per model
token_usage_global: Dict[str, Dict[str, int]] = {}

def _get_token_counts(llm_output: Any) -> Tuple[int, int]:
    """Helper to extract prompt and completion tokens from LLM output.
    
    Flexible implementation that handles various input types:
    - Dict with standard token keys
    - String with token info
    - None values
    - Other unexpected formats
    
    Returns:
        Tuple of (prompt_tokens, completion_tokens)
    """
    # Handle None case
    if llm_output is None:
        return 0, 0
    
    # Handle string case (attempt to extract numbers)
    if isinstance(llm_output, str):
        try:
            # Look for patterns like "prompt_tokens: 123" or "input: 123, output: 456"
            prompt_match = re.search(r'(?:prompt|input)(?:_tokens)?[:\s]+(\d+)', llm_output, re.IGNORECASE)
            completion_match = re.search(r'(?:completion|output)(?:_tokens)?[:\s]+(\d+)', llm_output, re.IGNORECASE)
            
            prompt_tokens = int(prompt_match.group(1)) if prompt_match else 0
            completion_tokens = int(completion_match.group(1)) if completion_match else 0
            
            return prompt_tokens, completion_tokens
        except Exception:
            # If parsing fails, just return zeros
            return 0, 0
    
    # Handle dict-like case
    if hasattr(llm_output, 'get'):
        # First try standard 'token_usage' key
        usage = llm_output.get('token_usage')
        if usage and hasattr(usage, 'get'):
            prompt_tokens = usage.get('prompt_tokens', usage.get('input_tokens', 0))
            completion_tokens = usage.get('completion_tokens', usage.get('output_tokens', 0))
            
            # Handle different types
            try:
                prompt_tokens = int(prompt_tokens) if prompt_tokens is not None else 0
                completion_tokens = int(completion_tokens) if completion_tokens is not None else 0
                return prompt_tokens, completion_tokens
            except (ValueError, TypeError):
                pass  # Fall through to next checks
        
        # Direct key access if no token_usage
        try:
            prompt_tokens = llm_output.get('prompt_token_count', llm_output.get('input_tokens', 0))
            completion_tokens = llm_output.get('completion_token_count', llm_output.get('output_tokens', 0))
            
            prompt_tokens = int(prompt_tokens) if prompt_tokens is not None else 0
            completion_tokens = int(completion_tokens) if completion_tokens is not None else 0
            return prompt_tokens, completion_tokens
        except (ValueError, TypeError):
            pass  # Fall through to default return
    
    # Try to extract from any object with attributes
    if hasattr(llm_output, 'prompt_tokens') and hasattr(llm_output, 'completion_tokens'):
        try:
            prompt_tokens = int(llm_output.prompt_tokens)
            completion_tokens = int(llm_output.completion_tokens)
            return prompt_tokens, completion_tokens
        except (ValueError, TypeError, AttributeError):
            pass  # Fall through to default return
    
    # Default case: no tokens found
    return 0, 0

def _calculate_cost(model_name: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Calculate the cost based on the model name and token counts."""
    model_name_lower = model_name.lower()
    
    # Use startswith for more specific prefix matching
    # Order matters: check most specific prefixes first
    if model_name_lower.startswith("claude-3-7"): # Assuming any claude-3-7 variant has same cost
        cost_per_1k_input = CLAUDE_COST_PER_1K_INPUT_TOKENS
        cost_per_1k_output = CLAUDE_COST_PER_1K_OUTPUT_TOKENS
        logger.debug(f"Using Claude 3.7 Sonnet pricing: ${cost_per_1k_input}/1K input, ${cost_per_1k_output}/1K output")
    elif model_name_lower.startswith("gemini-2.5-pro"): # Matches gemini-2.5-pro-preview-XX, etc.
        cost_per_1k_input = GEMINI_COST_PER_1K_INPUT_TOKENS
        cost_per_1k_output = GEMINI_COST_PER_1K_OUTPUT_TOKENS
        logger.debug(f"Using Gemini 2.5 Pro pricing: ${cost_per_1k_input}/1K input, ${cost_per_1k_output}/1K output")
    elif model_name_lower.startswith("gemini-2.5-flash"): # Matches gemini-2.5-flash-preview-XX
        cost_per_1k_input = GEMINI_25_FLASH_PREVIEW_COST_PER_1K_INPUT # Use preview costs for now
        cost_per_1k_output = GEMINI_25_FLASH_PREVIEW_COST_PER_1K_OUTPUT
        logger.debug(f"Using Gemini 2.5 Flash Preview pricing: ${cost_per_1k_input}/1K input, ${cost_per_1k_output}/1K output")
    elif model_name_lower.startswith("gemini-2.0-flash"): # Matches gemini-2.0-flash, gemini-2.0-flash-001 etc.
        cost_per_1k_input = GEMINI_FLASH_COST_PER_1K_INPUT
        cost_per_1k_output = GEMINI_FLASH_COST_PER_1K_OUTPUT
        logger.debug(f"Using Gemini 2.0 Flash pricing ({model_name}): ${cost_per_1k_input}/1K input, ${cost_per_1k_output}/1K output")
    else:
        logger.warning(f"Unknown model prefix for cost calculation: {model_name}. Using Gemini 2.5 Pro pricing as fallback.")
        cost_per_1k_input = GEMINI_COST_PER_1K_INPUT_TOKENS
        cost_per_1k_output = GEMINI_COST_PER_1K_OUTPUT_TOKENS

    # Calculate final cost
    cost = ((prompt_tokens / 1000) * cost_per_1k_input) + \
           ((completion_tokens / 1000) * cost_per_1k_output)
    return cost

class TokenCostProcess:
    """Handles tracking and summarizing token usage and costs."""
    def __init__(self):
        self.total_cost: float = 0.0
        self.token_usage: Dict[str, Dict[str, int]] = {}
        # Add cache tracking
        self.cache_hits: Dict[str, Dict[str, int]] = {}
        self.total_cache_input_tokens = 0
        self.total_cache_output_tokens = 0

    @property
    def model_usage(self):
        """Returns the token usage data including cache hits."""
        result = {}
        # Process actual LLM calls
        for model_name, usage in self.token_usage.items():
            model_key = str(model_name)
            prompt_tokens = usage.get('prompt', 0)
            completion_tokens = usage.get('completion', 0)
            model_cost = _calculate_cost(model_name, prompt_tokens, completion_tokens)
            result[model_key] = {
                'input_tokens': prompt_tokens,
                'output_tokens': completion_tokens,
                'total_tokens': prompt_tokens + completion_tokens,
                'total_cost': model_cost,
                'cache_hits': 0, # Initialize cache hits for this model
                'cache_input_tokens': 0,
                'cache_output_tokens': 0
            }
            logger.debug(f"TokenCostProcess.model_usage (LLM): Model={model_key}, Input={prompt_tokens}, Output={completion_tokens}, Cost=${model_cost:.4f}")

        # Add cache hit data
        for model_name, cache_data in self.cache_hits.items():
            model_key = str(model_name)
            hits = cache_data.get('hits', 0)
            input_tokens = cache_data.get('input_tokens', 0)
            output_tokens = cache_data.get('output_tokens', 0)
            saved_cost = _calculate_cost(model_name, input_tokens, output_tokens)
            
            if model_key not in result:
                # If model only had cache hits, create an entry
                result[model_key] = {
                    'input_tokens': 0,
                    'output_tokens': 0,
                    'total_tokens': 0,
                    'total_cost': 0.0,
                    'cache_hits': hits,
                    'cache_input_tokens': input_tokens,
                    'cache_output_tokens': output_tokens,
                    'saved_cost': saved_cost
                }
            else:
                # Add cache data to existing entry
                result[model_key]['cache_hits'] = hits
                result[model_key]['cache_input_tokens'] = input_tokens
                result[model_key]['cache_output_tokens'] = output_tokens
                result[model_key]['saved_cost'] = saved_cost
                
            logger.debug(f"TokenCostProcess.model_usage (Cache): Model={model_key}, Hits={hits}, SavedInput={input_tokens}, SavedOutput={output_tokens}, SavedCost=${saved_cost:.4f}")
            
        if not result:
            logger.warning(f"TokenCostProcess.model_usage returning empty result. TokenUsage: {self.token_usage}, CacheHits: {self.cache_hits}")
        else:
            logger.debug(f"TokenCostProcess.model_usage returning data for {len(result)} models (including cache)")
            
        return result

    def update_usage(self, model_name: str, prompt_tokens: Any, completion_tokens: Any):
        """Update token usage for a specific model (actual LLM call)."""
        model_name = str(model_name) if model_name is not None else "unknown"

        # Normalize model name (Simplified - relies on accurate name from handler)
        model_key = model_name
            
        logger.debug(f"Updating LLM usage for model_key '{model_key}'")

        if model_key not in self.token_usage:
            self.token_usage[model_key] = {'prompt': 0, 'completion': 0, 'total': 0}
        
        try:
            prompt_count = int(prompt_tokens) if prompt_tokens is not None else 0
            completion_count = int(completion_tokens) if completion_tokens is not None else 0
        except (ValueError, TypeError):
             prompt_count, completion_count = 0, 0
             logger.warning(f"Could not convert LLM tokens to int for {model_key}. P='{prompt_tokens}', C='{completion_tokens}'.")

        self.token_usage[model_key]['prompt'] += prompt_count
        self.token_usage[model_key]['completion'] += completion_count
        self.token_usage[model_key]['total'] += prompt_count + completion_count
        logger.debug(f"Updated token usage for {model_key}: +{prompt_count} prompt, +{completion_count} completion. New totals: {self.token_usage[model_key]['prompt']} prompt, {self.token_usage[model_key]['completion']} completion, {self.token_usage[model_key]['total']} total.")

    def update_cache_hit(self, model_name: str, input_tokens: int, output_tokens: int):
        """Update statistics for a cache hit."""
        model_name = str(model_name) if model_name is not None else "unknown"
        # Normalize model name similarly to update_usage if needed, but rely on cache monitor's extraction for now
        model_key = model_name 
        logger.debug(f"Updating CACHE usage for model_key '{model_key}'")
        
        if model_key not in self.cache_hits:
            self.cache_hits[model_key] = {'hits': 0, 'input_tokens': 0, 'output_tokens': 0}
        
        self.cache_hits[model_key]['hits'] += 1
        self.cache_hits[model_key]['input_tokens'] += input_tokens
        self.cache_hits[model_key]['output_tokens'] += output_tokens
        self.total_cache_input_tokens += input_tokens
        self.total_cache_output_tokens += output_tokens
        logger.debug(f"Cache hit recorded for {model_key}: +1 hit, +{input_tokens} input_saved, +{output_tokens} output_saved. New totals: {self.cache_hits[model_key]['hits']} hits, {self.cache_hits[model_key]['input_tokens']} saved_in, {self.cache_hits[model_key]['output_tokens']} saved_out.")

    def update_cost(self, cost: float):
        """Update the total cost."""
        self.total_cost += cost

    def get_cost_summary(self) -> str:
        """Generate a string summary of token usage and cost."""
        if not LOG_COST_SUMMARY:
            return ""
            
        summary_lines = ["--- Token Usage & Cost Summary ---"]
        total_prompt = 0
        total_completion = 0
        total_tokens = 0
        
        # Calculate totals for LLM calls
        for model, usage in self.token_usage.items():
            prompt = usage.get('prompt', 0)
            completion = usage.get('completion', 0)
            cost = _calculate_cost(model, prompt, completion)
            summary_lines.append(
                f"- {model}: Prompt={prompt:,}, Completion={completion:,}, Total={prompt + completion:,}, Cost=${cost:.4f}"
            )
            total_prompt += prompt
            total_completion += completion
            total_tokens += prompt + completion

        summary_lines.append(" ")
        summary_lines.append("--- Cache Usage --- ")
        total_saved_cost = 0.0
        if not self.cache_hits:
             summary_lines.append("- No cache hits recorded.")
        else:
            for model, cache_data in self.cache_hits.items():
                hits = cache_data.get('hits', 0)
                saved_input = cache_data.get('input_tokens', 0)
                saved_output = cache_data.get('output_tokens', 0)
                saved_cost = _calculate_cost(model, saved_input, saved_output)
                summary_lines.append(
                    f"- {model} (Cache): Hits={hits:,}, Saved Input={saved_input:,}, Saved Output={saved_output:,}, Saved Cost=${saved_cost:.4f}"
                )
                total_saved_cost += saved_cost
                
        summary_lines.append(" ")
        summary_lines.append("--- Totals ---")
        summary_lines.append(f"- Total LLM Tokens: {total_tokens:,} (Prompt: {total_prompt:,}, Completion: {total_completion:,})")
        summary_lines.append(f"- Total LLM Cost: ${self.total_cost:.4f}")
        summary_lines.append(f"- Total Cache Hits: {sum(d.get('hits', 0) for d in self.cache_hits.values()):,}")
        summary_lines.append(f"- Total Saved Tokens (Cache): {self.total_cache_input_tokens + self.total_cache_output_tokens:,} (Input: {self.total_cache_input_tokens:,}, Output: {self.total_cache_output_tokens:,})")
        summary_lines.append(f"- Estimated Saved Cost (Cache): ${total_saved_cost:.4f}")
        summary_lines.append(f"- Estimated Total Cost (incl. savings): ${self.total_cost + total_saved_cost:.4f}")
        summary_lines.append("------------------------------------")
        
        return "\n".join(summary_lines)

    def log_summary(self):
        """Log the cost summary if enabled."""
        if LOG_COST_SUMMARY:
            summary = self.get_cost_summary()
            logger.info(summary)
            print(summary) # Also print to console for visibility
            
    # Methods for Chainlit integration (assuming ChainlitCallbackHandler uses this)
    def get_cost_info(self) -> str:
        # Returns a simple string for real-time display, maybe just total cost
        return f"Total Est. Cost: ${self.total_cost:.6f}"

    def update_cost_info(self, info: str):
        # Placeholder if Chainlit handler needs to push updates elsewhere
        pass 

    def record_cache_hit(self, model_name: str, input_tokens: int, output_tokens: int):
        """Records a cache hit using the token cost processor."""
        if not TRACK_TOKEN_USAGE:
            return
            
        logger.debug(f"Handler received cache hit: Model='{model_name}', Input={input_tokens}, Output={output_tokens}")
        self.update_cache_hit(model_name, input_tokens, output_tokens)

class TokenUsageCallbackHandler(BaseCallbackHandler):
    """LangChain Callback Handler to track token usage and calculate cost."""
    
    def __init__(self, token_cost_processor: TokenCostProcess):
        self.token_cost_processor = token_cost_processor
        # Initialize cache hit tracking - THESE ARE NOW HANDLED BY TokenCostProcess
        # self.cache_hits = {} 
        # self.total_saved_tokens_input = 0
        # self.total_saved_tokens_output = 0
        # self.total_saved_cost = 0.0
        self.model_roles: Dict[UUID, str] = {} # Stores role (primary/summarizer) per run_id

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], *, run_id: UUID, parent_run_id: Optional[UUID] = None, tags: Optional[List[str]] = None, metadata: Optional[Dict[str, Any]] = None, **kwargs: Any
    ) -> None:
        """Log LLM start and attempt to identify model role via tags."""
        # Determine model role from tags
        role = "unknown"
        if tags:
            if "primary" in tags:
                role = "primary"
            elif "summarizer" in tags:
                role = "summarizer"
            elif "next_step" in tags:
                role = "next_step"
            # Add more role checks if needed
            
        self.model_roles[run_id] = role
        logger.debug(f"LLM Start (run_id={run_id}, role={role}): Tags={tags}")
        
        # Original logic for debugging/logging
        # Extract model name using a safer approach
        inv_params = serialized.get('invocation_params', {})
        model_name = inv_params.get('model_name') or inv_params.get('model') or serialized.get('name') or serialized.get('id', ['unknown'])[-1]
        logger.debug(f"LLM Start (Model: {model_name}, Run ID: {run_id}, Parent Run ID: {parent_run_id})")

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, parent_run_id: Optional[UUID] = None, **kwargs: Any) -> None:
        """Process LLM end, extract tokens, calculate cost, and update usage."""
        if not TRACK_TOKEN_USAGE:
            return

        logger.debug(f"LLM End (run_id={run_id}): Received response object.")

        # --- Determine Model Name/Key using Role ---
        role = self.model_roles.get(run_id, "unknown")
        logger.debug(f"LLM End (run_id={run_id}): Retrieved role='{role}'")

        actual_model_name_from_response = "unknown"
        # Access response.llm_output directly
        if response.llm_output and isinstance(response.llm_output, dict):
            # Try to get the model name reported by the API (e.g., Google AI)
            actual_model_name_from_response = response.llm_output.get('model_name', "unknown")
        elif response.generations and response.generations[0] and response.generations[0][0].generation_info:
            # Fallback check in generation_info
            actual_model_name_from_response = response.generations[0][0].generation_info.get('model_name', "unknown")

        logger.debug(f"LLM End (run_id={run_id}): Actual model from response='{actual_model_name_from_response}'")

        # Determine the key for tracking based on role and config
        # Prefer configured names, fall back to actual response name
        if role == "primary":
            model_key = PRIMARY_MODEL_NAME or actual_model_name_from_response
        elif role == "summarizer":
            model_key = SUMMARIZER_MODEL or actual_model_name_from_response
        elif role == "next_step":
            # Always use the configured NEXT_STEP_MODEL for next_step role
            model_key = NEXT_STEP_MODEL
            logger.debug(f"LLM End (run_id={run_id}): Using NEXT_STEP_MODEL ('{NEXT_STEP_MODEL}') for key.")
        else:
            model_key = actual_model_name_from_response
            logger.warning(f"LLM End (run_id={run_id}): Role was unknown. Using actual model name '{model_key}' for tracking.")

        logger.debug(f"LLM End (run_id={run_id}): Determined model_key='{model_key}' for tracking.")
        # Add a clearer log message to help with debugging
        if 'gemini-2.5-flash' in model_key and role == 'next_step':
            logger.debug(f"Using Gemini 2.5 Flash pricing for next_step role (model: {model_key})")
        # -----------------------------------------

        # --- Simplified Token Extraction ---
        prompt_tokens = 0
        completion_tokens = 0
        cost = 0.0
        source = "unknown" # Track where tokens came from

        # Preferred method: usage_metadata directly from response.llm_output (Gemini, newer Langchain?)
        if response.llm_output and isinstance(response.llm_output, dict) and 'usage_metadata' in response.llm_output:
            usage_meta = response.llm_output['usage_metadata']
            if usage_meta and isinstance(usage_meta, dict):
                # Primary keys for Gemini (as seen in logs): 'input_tokens', 'output_tokens'
                # Fallback keys sometimes seen: 'prompt_token_count', 'candidates_token_count'
                prompt_tokens = usage_meta.get('input_tokens', usage_meta.get('prompt_token_count', 0))
                completion_tokens = usage_meta.get('output_tokens', usage_meta.get('candidates_token_count', usage_meta.get('completion_tokens', 0))) # Add completion_tokens fallback
                source = "response.llm_output.usage_metadata"
                logger.debug(f"LLM End (run_id={run_id}): Tokens from {source}: P={prompt_tokens}, C={completion_tokens}")

        # Fallback 1: token_usage directly from response.llm_output (Older LangChain/Anthropic?)
        elif response.llm_output and isinstance(response.llm_output, dict) and 'token_usage' in response.llm_output:
            usage = response.llm_output['token_usage']
            if usage and isinstance(usage, dict):
                prompt_tokens = usage.get('prompt_tokens', usage.get('input_tokens', 0))
                completion_tokens = usage.get('completion_tokens', usage.get('output_tokens', 0))
                source = "response.llm_output.token_usage"
                logger.debug(f"LLM End (run_id={run_id}): Tokens from {source}: P={prompt_tokens}, C={completion_tokens}")

        # Fallback 2: Iterating through generations (More complex structure, OpenAI/Anthropic?)
        elif response.generations:
            try:
                for gen_list in response.generations: # Iterate through lists of generations
                    if not gen_list: continue
                    for gen in gen_list: # Iterate through generations in the list
                         if not gen: continue
                         # Check generation_info
                         if gen.generation_info:
                             gen_info = gen.generation_info
                             if 'token_usage' in gen_info and isinstance(gen_info['token_usage'], dict):
                                 usage = gen_info['token_usage']
                                 prompt_tokens = usage.get('prompt_tokens', usage.get('input_tokens', 0))
                                 completion_tokens = usage.get('completion_tokens', usage.get('output_tokens', 0))
                                 source = "gen.generation_info.token_usage"
                                 logger.debug(f"LLM End (run_id={run_id}): Tokens from {source}: P={prompt_tokens}, C={completion_tokens}")
                                 break # Found tokens, exit inner loop
                             elif 'prompt_tokens' in gen_info and 'completion_tokens' in gen_info:
                                 prompt_tokens = gen_info.get('prompt_tokens', 0)
                                 completion_tokens = gen_info.get('completion_tokens', 0)
                                 source = "gen.generation_info keys"
                                 logger.debug(f"LLM End (run_id={run_id}): Tokens from {source}: P={prompt_tokens}, C={completion_tokens}")
                                 break # Found tokens, exit inner loop

                         # Check message usage_metadata (for AIMessage-like structures)
                         if hasattr(gen, 'message') and hasattr(gen.message, 'usage_metadata') and gen.message.usage_metadata:
                             usage_meta = gen.message.usage_metadata
                             if usage_meta and isinstance(usage_meta, dict):
                                 prompt_tokens = usage_meta.get('prompt_token_count', usage_meta.get('input_tokens', 0))
                                 completion_tokens = usage_meta.get('candidates_token_count', usage_meta.get('completion_tokens', usage_meta.get('output_tokens', 0)))
                                 source = "gen.message.usage_metadata"
                                 logger.debug(f"LLM End (run_id={run_id}): Tokens from {source}: P={prompt_tokens}, C={completion_tokens}")
                                 break # Found tokens, exit inner loop
                    if prompt_tokens > 0 or completion_tokens > 0:
                         break # Found tokens, exit outer loop
            except Exception as e:
                 logger.warning(f"LLM End (run_id={run_id}): Error parsing generations for tokens: {e}", exc_info=True)

        # Ensure tokens are integers
        try:
            prompt_tokens = int(prompt_tokens) if prompt_tokens is not None else 0
            completion_tokens = int(completion_tokens) if completion_tokens is not None else 0
        except (ValueError, TypeError):
            logger.warning(f"LLM End (run_id={run_id}): Could not convert extracted tokens to integers. P='{prompt_tokens}', C='{completion_tokens}'. Resetting to 0.")
            prompt_tokens = 0
            completion_tokens = 0
            source = "error_reset"

        # Calculate cost only if tokens were found
        if prompt_tokens > 0 or completion_tokens > 0:
            # Ensure token_cost_processor exists before calling methods
            if self.token_cost_processor:
                cost = _calculate_cost( # Call the standalone function directly
                    model_name=model_key, # Use the determined model key
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens
                )
                self.token_cost_processor.update_cost(cost)
                logger.info(f"LLM End ({source}): Model={model_key}, Prompt Tokens={prompt_tokens:,}, Completion Tokens={completion_tokens:,}, Cost=${cost:.6f}")
            else:
                logger.error(f"LLM End (run_id={run_id}): token_cost_processor not available to calculate cost for model '{model_key}'.")

        else:
             # Log detailed info if tokens still not found
             logger.warning(f"LLM End (run_id={run_id}): No token usage reported or extracted for model '{model_key}'. Source attempted: '{source}'. Role: '{role}'.")
             if response.llm_output:
                  logger.debug(f"LLM End (run_id={run_id}): llm_output content: {response.llm_output}")
             if response.generations:
                  logger.debug(f"LLM End (run_id={run_id}): generations content: {response.generations}")


        # Update usage with the determined model key
        if self.token_cost_processor:
            self.token_cost_processor.update_usage(
                model_name=model_key,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens
            )
        else:
             logger.error(f"LLM End (run_id={run_id}): token_cost_processor not available to update usage for model '{model_key}'.")

        # Clean up role mapping
        if run_id in self.model_roles:
            try:
                del self.model_roles[run_id]
                logger.debug(f"LLM End (run_id={run_id}): Cleaned up role mapping.")
            except KeyError:
                 logger.warning(f"LLM End (run_id={run_id}): Attempted to delete role mapping, but key was not found.")

    def on_chain_end(self, outputs: Dict[str, Any], *, run_id: UUID, parent_run_id: Optional[UUID] = None, tags: Optional[List[str]] = None, **kwargs: Any) -> None:
        """Log chain end, potentially useful for complex flows."""
        # Improved check for end of main research run
        if tags and ('research_workflow' in tags or 'final_summary' in tags): 
            if isinstance(outputs, dict) and ("final_answer" in outputs or "output" in outputs):
                logger.info(f"Chain end detected for run {run_id} (Tags: {tags}). Logging final token summary.")
                self.token_cost_processor.log_summary()
             
    def on_agent_finish(self, finish, *, run_id: UUID, parent_run_id: Optional[UUID] = None, tags: Optional[List[str]] = None, **kwargs: Any) -> Any:
        """Log the summary when the agent finishes."""
        logger.info(f"Agent finish detected for run {run_id}. Logging final token summary.")
        self.token_cost_processor.log_summary()

    def record_cache_hit(self, model_name: str, input_tokens: int, output_tokens: int):
        """Records a cache hit using the token cost processor."""
        if not TRACK_TOKEN_USAGE:
            return
            
        logger.debug(f"Handler received cache hit: Model='{model_name}', Input={input_tokens}, Output={output_tokens}")
        # Pass to the processor
        self.token_cost_processor.update_cache_hit(model_name, input_tokens, output_tokens) 

# --- Standalone Functions for Global Usage --- #

def get_global_token_summary() -> str:
    """Returns a formatted string summary of global token usage."""
    # Note: This uses the legacy global dict. Prefer TokenCostProcess.
    if not token_usage_global:
        return "No token usage tracked globally."
    
    summary_lines = ["--- Global Token Usage Summary ---"]
    total_cost = 0.0
    for model, usage in token_usage_global.items():
        prompt_tokens = usage.get('prompt', 0)
        completion_tokens = usage.get('completion', 0)
        model_cost = _calculate_cost(model, prompt_tokens, completion_tokens)
        total_cost += model_cost
        summary_lines.append(f"Model: {model}")
        summary_lines.append(f"  Prompt Tokens: {prompt_tokens:,}")
        summary_lines.append(f"  Completion Tokens: {completion_tokens:,}")
        summary_lines.append(f"  Total Tokens: {usage.get('total', 0):,}")
        summary_lines.append(f"  Estimated Cost: ${model_cost:.6f}")
    summary_lines.append("----------------------------------")
    summary_lines.append(f"Total Estimated Cost: ${total_cost:.6f}")
    summary_lines.append("----------------------------------")
    return "\\n".join(summary_lines)
    
def log_global_token_summary():
    """Logs the global token usage summary if enabled."""
    # Note: This uses the legacy global dict. Prefer TokenCostProcess.
    if LOG_COST_SUMMARY and TRACK_TOKEN_USAGE:
        summary = get_global_token_summary()
        logger.info(summary)
        print(summary) # Also print to console

# Context manager for token usage tracking
class TokenCallbackManager:
    """Context manager for token usage tracking.
    
    Example:
        ```python
        from src.token_callback import TokenCallbackManager
        
        # Track token usage for a specific LLM call
        with TokenCallbackManager() as cb:
            response = llm.invoke("Hello, world!")
            print(f"Total tokens: {cb.total_tokens}")
            print(f"Cost: ${cb.get_total_cost():.4f}")
        ```
    """
    
    def __init__(self):
        self.token_cost_process = TokenCostProcess()
        self.handler = TokenUsageCallbackHandler(self.token_cost_process)

    def __enter__(self):
        return self.handler
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass 
    
    def get_final_summary_message(self):
        """Return the final summary message for token usage (for Chainlit display)."""
        return self.handler.token_cost_processor.get_cost_info()

    def get_total_cost(self) -> float:
        """Calculate the total cost across all model types."""
        return self.handler.token_cost_processor.total_cost
    
    @property
    def model_usage(self) -> Dict[str, Dict[str, Any]]:
        """Get the model usage data directly.
        
        Returns:
            Dictionary mapping model names to usage statistics.
        """
        if hasattr(self.handler, 'token_cost_processor'):
            return self.handler.token_cost_processor.model_usage
        return {}
    
    def get_total_saved_cost(self) -> float:
        """
        Calculate the total saved cost from cache hits.

        Returns:
            float: The total saved cost.
        """
        return self.token_cost_process.total_saved_cost

    def get_cache_hits_data(self) -> dict:
        """
        Retrieve the cache hits data.

        Returns:
            dict: A dictionary containing cache hits data.
        """
        return {
            'total_queries': self.token_cost_process.total_queries,
            'cache_hits': self.token_cost_process.cache_hits,
            'cache_hits_percentage': self.token_cost_process.get_cache_hits_percentage()
        }
    
    def log_summary(self):
        """Log a summary of token usage and cost."""
        if not TRACK_TOKEN_USAGE or not LOG_COST_SUMMARY:
            return
        
        logger.info("===== Token Usage and Cost Summary =====")
        total_cost = 0.0
        
        for model_name, usage in self.handler.token_cost_processor.token_usage.items():
            # Only show models with usage
            if usage["prompt"] > 0 or usage["completion"] > 0:
                total_tokens = usage["prompt"] + usage["completion"]
                model_cost = _calculate_cost(model_name, usage["prompt"], usage["completion"])
                total_cost += model_cost
                
                logger.info(f"{model_name} Usage:")
                logger.info(f"  Tokens: {usage['prompt']} prompt, {usage['completion']} completion, {total_tokens} total")
                logger.info(f"  Cost: ${model_cost:.4f}")
        
        logger.info(f"Total usage across all models: {sum(usage['total'] for usage in self.handler.token_cost_processor.token_usage.values()):,} tokens, ${total_cost:.4f}")
        logger.info("=========================================")
    
    def __repr__(self) -> str:
        """String representation of token usage and cost."""
        base_info = (
            f"Tokens Used: {sum(usage['total'] for usage in self.handler.token_cost_processor.token_usage.values()):}\n"
            f"\tPrompt Tokens: {sum(usage['prompt'] for usage in self.handler.token_cost_processor.token_usage.values()):}\n"
            f"\tCompletion Tokens: {sum(usage['completion'] for usage in self.handler.token_cost_processor.token_usage.values()):}\n"
            f"Total Cost (USD): ${self.get_total_cost():.4f}"
        )
        
        return base_info

    def get_detailed_summary_string(self) -> str:
        """Generate a detailed Markdown string summary of token usage and cost per model."""
        if not TRACK_TOKEN_USAGE:
            return "Token tracking is disabled."
        
        summary_lines = []
        summary_lines.append("#### 📊 Token Usage & Cost Summary:")
        
        # Check if any models were used or saved tokens
        has_activity = any(
            usage["prompt"] > 0 or usage["completion"] > 0
            for usage in self.handler.token_cost_processor.token_usage.values()
        )

        has_cache_hits = hasattr(self.handler, 'cache_hits') and self.handler.cache_hits

        if not has_activity and not has_cache_hits:
            summary_lines.append("*No token usage recorded yet.*")
        else:
            # Create the consolidated table
            summary_lines.append("\n| Category | Input | Output | Total | Cost |")
            summary_lines.append("| :--- | ---: | ---: | ---: | ---: |")
            
            total_net_cost = 0.0
            total_net_tokens = 0
            
            # Process each model type explicitly
            for model_name, usage in self.handler.token_cost_processor.token_usage.items():
                # Actual (Net) Usage
                net_input = usage["prompt"]
                net_output = usage["completion"]
                net_total_tokens = net_input + net_output
                net_cost = _calculate_cost(model_name, net_input, net_output)

                # Add Actual Usage row if there was any net usage
                if net_total_tokens > 0:
                    # REMOVED: Special handling for pdf_processing
                    # cost_display = f"${net_cost:.4f}" if model_name != "pdf_processing" else "N/A"
                    # category_display = "PDF Text Extraction" if model_name == "pdf_processing" else f"**{model_name} Usage**"
                    # --- Simplified Row --- #
                    cost_display = f"${net_cost:.4f}"
                    category_display = f"**{model_name}**"
                    summary_lines.append(f"| {category_display} | `{net_input:,}` | `{net_output:,}` | `{net_total_tokens:,}` | {cost_display} |")
                    # --- End Simplified Row --- #
                    total_net_cost += net_cost
                    total_net_tokens += net_total_tokens
        
        # REMOVED redundant check, handled above
        # if not has_activity and not has_cache_hits:
        #     summary_lines.append("*No token usage recorded yet.*")
        # else:
        
        # Ensure totals row is only added if there was activity
        if has_activity or has_cache_hits: # Check if anything was processed
            # Add totals row
            summary_lines.append(f"| **Total** |  |  | `{total_net_tokens:,}` | **${total_net_cost:.4f}** |") # Simplified totals row

        # The logger call seems out of place here, summary string is returned
        # logger.info("\n".join(summary_lines))
        return "\n".join(summary_lines)