import logging
import os
from typing import Optional, List, Literal, Union, Dict, Any
from pathlib import Path
import time
import random
import asyncio

# LangChain component imports
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

# Import for HuggingFace models - make these conditional so they're only imported when needed
# HuggingFace imports will be done conditionally when provider == "local"

# Configuration imports
import config.settings

# Remove: from src.model_context import get_context_window, get_recommended_chunk_size
# Use config.settings.get_context_window and config.settings.get_recommended_chunk_size instead

logger = logging.getLogger(__name__)

ProviderType = Literal["claude", "gemini", "local"]

# Add a RetryingLLM class that wraps any LLM with retry functionality
class RetryingLLM(BaseChatModel):
    """A wrapper around any LLM that adds retry functionality.
    
    This class intercepts LLM calls and implements exponential backoff retry
    logic for transient errors.
    """
    
    def __init__(
        self,
        llm: BaseChatModel,
        max_retries: int = 3,
        initial_delay: float = 1.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        max_delay: float = 60.0,
        errors_to_retry: Optional[List[str]] = None
    ):
        """Initialize the RetryingLLM.
        
        Args:
            llm: The base LLM to wrap
            max_retries: Maximum number of retries before giving up
            initial_delay: Initial delay between retries in seconds
            exponential_base: Base for exponential backoff
            jitter: Whether to add random jitter to delay
            max_delay: Maximum delay between retries
            errors_to_retry: List of error message substrings to retry on
        """
        self.llm = llm
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.max_delay = max_delay
        
        # Default errors to retry on
        self.errors_to_retry = errors_to_retry or [
            # Standard rate limiting errors
            "Rate limit",
            "rate limit",
            "429",
            "too many requests",
            "Too many requests",
            
            # Connection and timeout errors
            "timeout",
            "Timeout",
            "connection",
            "Connection",
            
            # Server errors 
            "server error",
            "Server error",
            "503",
            "502",
            "500",
            "internal error",
            "Internal error",
            
            # Gemini-specific errors
            "contents.parts must not be empty",  # Gemini-specific error
            "GenerateContentRequest.contents",   # Gemini-specific error
            "Invalid argument provided to Gemini", # Gemini general error
            
            # General availability issues
            "temporarily unavailable",
            "Please try again",
            "try again later"
        ]
        
    @property
    def _llm_type(self) -> str:
        """Return the type of this LLM."""
        wrapped_type = getattr(self.llm, "_llm_type", "unknown")
        return f"Retrying{wrapped_type}"

    @property
    def model_name(self) -> str:
        """Return the model name of the wrapped LLM."""
        if hasattr(self.llm, "model_name"):
            return self.llm.model_name
        elif hasattr(self.llm, "model"):
            return self.llm.model
        return "unknown"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return identifying parameters of the wrapped LLM."""
        params = {"wrapped_llm": self.llm._identifying_params} if hasattr(self.llm, "_identifying_params") else {}
        params.update({
            "max_retries": self.max_retries,
            "initial_delay": self.initial_delay,
            "exponential_base": self.exponential_base,
            "jitter": self.jitter
        })
        return params
    
    def _should_retry(self, error: Exception) -> bool:
        """Determine if an error should trigger a retry.
        
        Args:
            error: The exception that was raised
            
        Returns:
            Whether to retry the operation
        """
        error_str = str(error)
        return any(err_type in error_str for err_type in self.errors_to_retry)
    
    def _get_retry_delay(self, attempt: int) -> float:
        """Get the delay before the next retry.
        
        Args:
            attempt: The retry attempt number (0-based)
            
        Returns:
            Delay in seconds
        """
        delay = min(
            self.initial_delay * (self.exponential_base ** attempt),
            self.max_delay
        )
        if self.jitter:
            delay = delay * (0.5 + random.random())
        return delay
    
    async def _agenerate(
        self, 
        messages: List[BaseMessage], 
        stop: Optional[List[str]] = None,
        run_manager = None,
        **kwargs
    ) -> ChatResult:
        """Wrap the LLM's _agenerate method with retry logic."""
        attempt = 0
        last_error = None
        
        while attempt <= self.max_retries:
            try:
                if attempt > 0:
                    logger.warning(f"Retry attempt {attempt}/{self.max_retries} for {self.model_name}...")
                    # Create a copy of kwargs and add cache bypass for retries
                    retry_kwargs = kwargs.copy()
                    # Add no_cache=True to the metadata to bypass caching on retries
                    retry_kwargs["metadata"] = retry_kwargs.get("metadata", {})
                    if isinstance(retry_kwargs["metadata"], dict):
                        retry_kwargs["metadata"]["no_cache"] = True
                    
                    # Call the wrapped LLM with cache bypass
                    if hasattr(self.llm, '_agenerate'):
                        return await self.llm._agenerate(
                            messages=messages, 
                            stop=stop,
                            run_manager=run_manager,
                            **retry_kwargs
                        )
                    else:
                        # Fallback for LLMs that don't implement _agenerate
                        logger.warning(f"LLM {self.model_name} doesn't implement _agenerate, using generate instead")
                        return await self.llm.agenerate([messages], stop=stop, run_manager=run_manager, **retry_kwargs)
                else:
                    # First attempt - use original kwargs
                    # Call the wrapped LLM
                    if hasattr(self.llm, '_agenerate'):
                        return await self.llm._agenerate(
                            messages=messages, 
                            stop=stop,
                            run_manager=run_manager,
                            **kwargs
                        )
                    else:
                        # Fallback for LLMs that don't implement _agenerate
                        logger.warning(f"LLM {self.model_name} doesn't implement _agenerate, using generate instead")
                        return await self.llm.agenerate([messages], stop=stop, run_manager=run_manager, **kwargs)
                
            except Exception as e:
                last_error = e
                
                # Check if we should retry
                if attempt < self.max_retries and self._should_retry(e):
                    delay = self._get_retry_delay(attempt)
                    logger.warning(
                        f"LLM call to {self.model_name} failed with error: {e}. "
                        f"Retrying in {delay:.2f}s (attempt {attempt+1}/{self.max_retries})"
                    )
                    # Use asyncio.sleep instead of time.sleep for async method
                    await asyncio.sleep(delay)
                    attempt += 1
                else:
                    logger.error(
                        f"LLM call to {self.model_name} failed: {e}. "
                        f"No more retries (attempt {attempt+1}/{self.max_retries})"
                    )
                    raise
        
        # We shouldn't get here, but if we do, raise the last error
        raise last_error

    def _generate(
        self, 
        messages: List[BaseMessage], 
        stop: Optional[List[str]] = None,
        run_manager = None,
        **kwargs
    ) -> ChatResult:
        """Wrap the LLM's _generate method with retry logic."""
        attempt = 0
        last_error = None
        
        while attempt <= self.max_retries:
            try:
                if attempt > 0:
                    logger.warning(f"Retry attempt {attempt}/{self.max_retries} for {self.model_name}...")
                    
                    # Create a copy of kwargs and add cache bypass for retries
                    retry_kwargs = kwargs.copy()
                    # Add no_cache=True to the metadata to bypass caching on retries
                    retry_kwargs["metadata"] = retry_kwargs.get("metadata", {})
                    if isinstance(retry_kwargs["metadata"], dict):
                        retry_kwargs["metadata"]["no_cache"] = True
                    
                    # Call the wrapped LLM with cache bypass
                    if hasattr(self.llm, '_generate'):
                        return self.llm._generate(
                            messages=messages, 
                            stop=stop,
                            run_manager=run_manager,
                            **retry_kwargs
                        )
                    else:
                        # Fallback for LLMs that don't implement _generate
                        logger.warning(f"LLM {self.model_name} doesn't implement _generate, using generate instead")
                        return self.llm.generate([messages], stop=stop, run_manager=run_manager, **retry_kwargs)
                else:
                    # First attempt - use original kwargs
                    # Call the wrapped LLM
                    if hasattr(self.llm, '_generate'):
                        return self.llm._generate(
                            messages=messages, 
                            stop=stop,
                            run_manager=run_manager,
                            **kwargs
                        )
                    else:
                        # Fallback for LLMs that don't implement _generate
                        logger.warning(f"LLM {self.model_name} doesn't implement _generate, using generate instead")
                        return self.llm.generate([messages], stop=stop, run_manager=run_manager, **kwargs)
                
            except Exception as e:
                last_error = e
                
                # Check if we should retry
                if attempt < self.max_retries and self._should_retry(e):
                    delay = self._get_retry_delay(attempt)
                    logger.warning(
                        f"LLM call to {self.model_name} failed with error: {e}. "
                        f"Retrying in {delay:.2f}s (attempt {attempt+1}/{self.max_retries})"
                    )
                    time.sleep(delay)
                    attempt += 1
                else:
                    logger.error(
                        f"LLM call to {self.model_name} failed: {e}. "
                        f"No more retries (attempt {attempt+1}/{self.max_retries})"
                    )
                    raise
        
        # We shouldn't get here, but if we do, raise the last error
        raise last_error

    # Make sure callbacks are properly passed through
    @property
    def callbacks(self):
        """Get the callbacks from the wrapped LLM."""
        return self.llm.callbacks if hasattr(self.llm, "callbacks") else None
    
    @callbacks.setter
    def callbacks(self, callbacks):
        """Set callbacks on the wrapped LLM."""
        if hasattr(self.llm, "callbacks"):
            self.llm.callbacks = callbacks
            
    # Forward verbose and other common properties
    @property
    def verbose(self):
        """Get the verbose setting from the wrapped LLM."""
        return self.llm.verbose if hasattr(self.llm, "verbose") else False
    
    @verbose.setter
    def verbose(self, verbose):
        """Set verbose on the wrapped LLM."""
        if hasattr(self.llm, "verbose"):
            self.llm.verbose = verbose

    # Forward tags
    @property
    def tags(self):
        """Get tags from the wrapped LLM."""
        return self.llm.tags if hasattr(self.llm, "tags") else None
    
    @tags.setter
    def tags(self, tags):
        """Set tags on the wrapped LLM."""
        if hasattr(self.llm, "tags"):
            self.llm.tags = tags
            
    # Forward metadata
    @property
    def metadata(self):
        """Get metadata from the wrapped LLM."""
        return self.llm.metadata if hasattr(self.llm, "metadata") else None
    
    @metadata.setter
    def metadata(self, metadata):
        """Set metadata on the wrapped LLM."""
        if hasattr(self.llm, "metadata"):
            self.llm.metadata = metadata
            
    # Required by BaseChatModel abstract class
    @property
    def client(self):
        """Get the client from the wrapped LLM."""
        return self.llm.client if hasattr(self.llm, "client") else None
    
    def get_num_tokens(self, text: str) -> int:
        """Get the number of tokens in the text."""
        if hasattr(self.llm, "get_num_tokens"):
            return self.llm.get_num_tokens(text)
        # Default tokenizer estimation as fallback (very rough)
        return len(text) // 4
    
    def get_num_tokens_from_messages(self, messages) -> int:
        """Get the number of tokens in the messages."""
        if hasattr(self.llm, "get_num_tokens_from_messages"):
            return self.llm.get_num_tokens_from_messages(messages)
        # Fallback: sum tokens from all message contents
        total = 0
        for message in messages:
            if hasattr(message, "content"):
                content = message.content
                if isinstance(content, str):
                    total += self.get_num_tokens(content)
        return total

def get_llm_client(
    provider: ProviderType,
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
    max_tokens: Optional[int] = None,
    is_summary_client: bool = False, # Flag for summarization model
    is_next_step_client: bool = False, # <<< ADDED FLAG
    is_local_client: bool = False, # Flag for local HuggingFace models
    content_size: Optional[int] = None, # Content size to determine if model is suitable
    callbacks: Optional[List[BaseCallbackHandler]] = None,  # Add back callbacks parameter
    use_retry_wrapper: bool = True,  # New parameter to control retry wrapper
    max_retries: int = 3  # New parameter for max retries
) -> BaseChatModel:
    """Factory function to create and return a LangChain chat model client.

    Args:
        provider: The LLM provider ("claude", "gemini", or "local").
        model_name: The specific model name to use. Defaults to config settings.
        api_key: The API key for the provider. Defaults to config settings or env vars.
        max_tokens: The maximum number of tokens for the response. Defaults based on provider/usage.
        is_summary_client: Flag to indicate if the client is for summarization (uses different token settings).
        is_next_step_client: Flag to indicate if the client is for the next step agent. # <<< ADDED DOC
        is_local_client: Flag to indicate if using a local HuggingFace model.
        content_size: Size of content to be processed (in tokens) - used to check if model is suitable.
        callbacks: Optional list of callback handlers to attach to the client.
        use_retry_wrapper: Whether to wrap the client with retry functionality.
        max_retries: Maximum number of retries for the retry wrapper.

    Returns:
        An instance of a LangChain BaseChatModel (e.g., ChatAnthropic, ChatGoogleGenerativeAI).

    Raises:
        ValueError: If the provider is unsupported or API key is missing.
        RuntimeError: If the client fails to initialize.
    """
    logger.info(f"Creating LLM client for provider: {provider}")

    # <<< MODIFIED TAG LOGIC >>>
    if is_next_step_client:
        tags = ["next_step"]
    elif is_summary_client:
        tags = ["summarizer"]
    else:
        tags = ["primary"]
    # <<< END MODIFIED TAG LOGIC >>>

    # Create the base LLM client
    llm_client = None

    if provider == "claude":
        # Determine parameters for Claude
        final_api_key = api_key or os.getenv("ANTHROPIC_API_KEY") or config.settings.ANTHROPIC_API_KEY
        final_model_name = model_name or config.settings.CLAUDE_MODEL_NAME
        final_max_tokens = max_tokens or 8092 # Default max tokens for Claude

        if not final_api_key:
            raise ValueError("ANTHROPIC_API_KEY is missing. Provide it via argument, config, or environment variable.")
        if not final_model_name:
             # Add a fallback default model if not configured
             final_model_name = "claude-3-haiku-20240307"
             logger.warning(f"Claude model name not specified, defaulting to {final_model_name}")

        try:
            llm_client = ChatAnthropic(
                anthropic_api_key=final_api_key,
                model_name=final_model_name,
                max_tokens=final_max_tokens,
                callbacks=callbacks,  # Add callbacks
                tags=tags # <<< USE UPDATED TAGS >>>
            )
            logger.info(f"Successfully created ChatAnthropic client for model: {final_model_name} with tags: {llm_client.tags}")
        except Exception as e:
            logger.error(f"Failed to initialize ChatAnthropic client: {e}", exc_info=True)
            raise RuntimeError(f"Failed to initialize ChatAnthropic client: {e}")

    elif provider == "gemini":
        # Determine parameters for Gemini
        final_api_key = api_key or os.getenv("GEMINI_API_KEY") or config.settings.GEMINI_API_KEY

        if not final_api_key:
            raise ValueError("GEMINI_API_KEY is missing. Provide it via argument, config, or environment variable.")

        if is_summary_client:
            if model_name:
                final_model_name = model_name
            else:
                final_model_name = config.settings.SUMMARIZER_MODEL
            final_max_tokens = max_tokens or config.settings.SUMMARY_MAX_TOKENS
            logger.info(f"Creating Gemini client for Summarization (Model: {final_model_name})")
        else:
            final_model_name = model_name or config.settings.GEMINI_MODEL_NAME
            final_max_tokens = max_tokens or 16384 # Default max tokens for Gemini main task
            logger.info(f"Creating Gemini client for Main Task (Model: {final_model_name})")
        if not final_model_name:
            # Add a fallback default model if not configured
            final_model_name = "gemini-1.5-flash-latest"
            logger.warning(f"Gemini model name not specified, defaulting to {final_model_name}")

        try:
            # <<< ADD DEBUG LOGGING >>>
            logger.debug(f"Attempting to create ChatGoogleGenerativeAI client with model='{final_model_name}', tags={tags}") # Log tags too
            llm_client = ChatGoogleGenerativeAI(
                google_api_key=final_api_key,
                model=final_model_name,
                max_output_tokens=final_max_tokens,
                convert_system_message_to_human=False,
                callbacks=callbacks,  # Add callbacks
                temperature=0.2,  # Setting a low temperature for more consistent output
                verbose=True,  # Enable verbose mode for better tracking
                metadata={"usage_metadata": True},  # Enable proper token usage tracking
                tags=tags, # <<< USE UPDATED TAGS >>>
                request_timeout=60.0,  # Increased timeout for reliability
                max_retries=6,  # Increase built-in retries
                streaming=False  # Disable streaming to avoid partial failures
            )
            logger.info(f"Successfully created ChatGoogleGenerativeAI client for model: {final_model_name} with tags: {llm_client.tags}")
        except Exception as e:
            logger.error(f"Failed to initialize ChatGoogleGenerativeAI client: {e}", exc_info=True)
            raise RuntimeError(f"Failed to initialize ChatGoogleGenerativeAI client: {e}")
            
    elif provider == "local":
        # --- REFACTOR for LlamaCpp ---
        # Conditionally import the required modules for local models
        try:
            # Only import these modules when actually using local models
            from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
            from langchain_community.llms import LlamaCpp
            from langchain_community.chat_models import ChatHuggingFace as CommunityChatHuggingFace
            logger.info("Successfully imported HuggingFace and LlamaCpp modules")
        except ImportError as ie:
            logger.error(f"Failed to import modules required for local models: {ie}", exc_info=True)
            raise RuntimeError(f"Local provider requires additional dependencies. Run: pip install langchain-huggingface langchain-community") from ie
            
        if is_summary_client:
            selected_model = model_name or config.settings.SUMMARIZER_MODEL
        else:
            # For primary reasoning model, use LOCAL_MODEL_NAME
            selected_model = model_name or config.settings.LOCAL_MODEL_NAME
            logger.info(f"Using local model for primary reasoning: {selected_model}")

        if not selected_model:
            raise ValueError("No local model specified. Please provide a model name or set LOCAL_MODEL_NAME in config.")

        logger.info(f"Attempting to load local GGUF model: {selected_model}")

        # Construct the full model path
        model_path = Path(config.settings.LOCAL_MODELS_DIR) / selected_model
        if not model_path.exists():
            # Check if it's just the filename vs directory name issue
            model_dir = Path(config.settings.LOCAL_MODELS_DIR)
            found = list(model_dir.glob(f"**/{selected_model}"))
            if found:
                 model_path = found[0]
                 logger.info(f"Found model file at: {model_path}")
            else:
                 raise ValueError(f"Model file {selected_model} not found in {config.settings.LOCAL_MODELS_DIR} or subdirectories. Please run setup.sh or check config.")
        
        # Ensure it's a file
        if not model_path.is_file():
             raise ValueError(f"Path {model_path} is not a file. Expected a GGUF model file.")

        logger.info(f"Loading local LlamaCpp model from: {model_path}")

        try:
            # Set default context window size based on model task
            default_n_ctx = 32768 
            if not is_summary_client:
                # For primary reasoning model, try to use larger context
                default_n_ctx = 65536  # 64K context window for reasoning model
                
            logger.info(f"Setting default n_ctx={default_n_ctx} for LlamaCpp initialization.")

            # Determine max tokens for generation
            if is_summary_client:
                final_max_tokens = max_tokens or config.settings.SUMMARY_MAX_TOKENS or 1024
            else:
                # For primary reasoning, allow longer responses
                final_max_tokens = max_tokens or 16384

            # Basic GPU layer offloading
            n_gpu_layers = config.settings.N_GPU_LAYERS # Default: -1 in settings

            # Create the LlamaCpp instance
            llm_client = LlamaCpp(
                model_path=str(model_path),
                n_ctx=default_n_ctx,  # Use the default n_ctx value here
                n_batch=512,  # Adjust based on VRAM/performance
                n_gpu_layers=n_gpu_layers, # Offload layers to GPU
                max_tokens=final_max_tokens, # Max tokens to generate
                temperature=0.2 if is_summary_client else 0.7, # Lower temp for summaries, higher for reasoning
                verbose=True, # Log Llama.cpp details
                callbacks=callbacks, # Pass callbacks
                # Add model-specific parameters based on task
                grammar_path=None,  # Can add tool grammar here for local models if needed
                tags=tags # <<< USE UPDATED TAGS >>>
            )
            logger.info(f"Successfully created LlamaCpp client for model: {selected_model} with tags: {llm_client.tags}")
            logger.info(f" LlamaCpp Params: n_ctx={llm_client.n_ctx}, n_gpu_layers={llm_client.n_gpu_layers}, max_tokens={llm_client.max_tokens}")
        except Exception as e:
            logger.error(f"Failed to initialize LlamaCpp client for {selected_model}: {e}", exc_info=True)
            # Provide more specific guidance if possible (e.g., build issues)
            error_msg = f"Failed to initialize LlamaCpp client for {selected_model}: {e}"
            if "cublas" in str(e).lower() or "metal" in str(e).lower() or "blas" in str(e).lower():
                 error_msg += "\\n -> This might indicate an issue with the llama-cpp-python build or GPU driver setup. Ensure CMake and necessary GPU SDKs were present during installation (see setup.sh)."
            raise RuntimeError(error_msg)

    # Wrap with retry functionality if requested
    if use_retry_wrapper and llm_client is not None:
        try:
            logger.info(f"Wrapping LLM client with retry functionality (max_retries={max_retries})")
            llm_client = RetryingLLM(llm_client, max_retries=max_retries)
        except Exception as e:
            logger.warning(f"Failed to initialize RetryingLLM wrapper: {e}. Using base LLM client without retry functionality.")
            # Continue with the unwrapped client
        
    return llm_client

def get_local_model_config(model_name: str) -> Dict[str, Any]:
    """Get configuration for specific local models.

    Ensures models targeted for lower RAM environments default to 4-bit loading.

    Returns:
        Dictionary with model-specific parameters:
        - tokenizer_params: parameters for AutoTokenizer.from_pretrained
        - model_params: parameters for AutoModelForCausalLM.from_pretrained
        - pipeline_params: parameters for HuggingFace pipeline
        - max_tokens: default max tokens for generation
    """
    # Basic configuration template
    base_config = {
        "tokenizer_params": {},
        "model_params": {},
        "pipeline_params": {},
        "max_tokens": 1024
    }

    config_dict = {} # Initialize empty dict

    # Model-specific configurations (Matching setup.sh names)
    # Tier 1 & 2 Models (Targeting <16GB RAM ideally)
    if model_name == "deepseek-r1-distill-llama-8b":
        config_dict = {
            "tokenizer_params": {"padding_side": "left", "truncation_side": "left"},
            "model_params": {"load_in_4bit": True, "trust_remote_code": True},
            "pipeline_params": {"do_sample": True, "top_k": 50, "top_p": 0.95, "repetition_penalty": 1.1},
            "max_tokens": 4096 # Increased default context
        }
    elif model_name == "yarn-mistral-7b-64k":
        config_dict = {
            "tokenizer_params": {"padding_side": "left"},
            "model_params": {"load_in_4bit": True},
            "pipeline_params": {"do_sample": True, "top_k": 50, "top_p": 0.95, "repetition_penalty": 1.1},
            "max_tokens": 4096 # Increased default context
        }
    elif model_name == "deepseek-r1-distill-qwen-7b":
        config_dict = {
            "tokenizer_params": {},
            "model_params": {"load_in_4bit": True, "trust_remote_code": True},
            "pipeline_params": {},
            "max_tokens": 4096 # Increased default context
        }
    elif model_name == "qwen-2.5-7b":
        config_dict = {
            "tokenizer_params": {},
            "model_params": {"load_in_4bit": True, "trust_remote_code": True},
            "pipeline_params": {},
            "max_tokens": 4096 # Increased default context
        }

    # Tier 2 & 3 Models (Targeting >=16GB RAM, 4-bit still recommended for <32GB)
    elif model_name == "deepseek-r1-distill-qwen-14b":
        config_dict = {
            "tokenizer_params": {},
            "model_params": {"load_in_4bit": True, "trust_remote_code": True},
            "pipeline_params": {},
            "max_tokens": 4096
        }
    elif model_name == "qwen-2.5-14b":
        config_dict = {
            "tokenizer_params": {},
            "model_params": {"load_in_4bit": True, "trust_remote_code": True},
            "pipeline_params": {},
            "max_tokens": 4096
        }

    # Tier 3 Models (Targeting >=16GB/32GB+ RAM, 4-bit recommended unless >64GB)
    elif model_name == "deepseek-r1-distill-qwen-32b":
        config_dict = {
            "tokenizer_params": {},
            "model_params": {"load_in_4bit": True, "trust_remote_code": True},
            "pipeline_params": {},
            "max_tokens": 4096
        }
    elif model_name == "qwen-2.5-32b":
        config_dict = {
            "tokenizer_params": {},
            "model_params": {"load_in_4bit": True, "trust_remote_code": True},
            "pipeline_params": {},
            "max_tokens": 4096
        }

    else:
        # Default configuration for any other unknown models
        config_dict = base_config
        logger.warning(f"No specific configuration for model {model_name}, using default values (may not load correctly)")

    # Merge with base config (allowing override)
    final_config = {**base_config, **config_dict}
    # Ensure critical model_params aren't accidentally overwritten by base if empty
    if "model_params" not in final_config:
         final_config["model_params"] = {}
    if "load_in_4bit" in config_dict.get("model_params", {}) and "load_in_4bit" not in final_config["model_params"]:
         final_config["model_params"]["load_in_4bit"] = config_dict["model_params"]["load_in_4bit"]
    if "trust_remote_code" in config_dict.get("model_params", {}) and "trust_remote_code" not in final_config["model_params"]:
        final_config["model_params"]["trust_remote_code"] = config_dict["model_params"]["trust_remote_code"]


    logger.info(f"Using final config for {model_name}: {final_config}") # Log the final config
    return final_config 