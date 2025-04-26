import os
import logging
from pathlib import Path
import yaml
from dotenv import load_dotenv

# Set up logger
logger = logging.getLogger(__name__)

# Define config path
CONFIG_DIR = Path(__file__).parent.absolute()
USER_CONFIG_PATH = Path.cwd() / "config.yaml"

# Default configuration values to use as fallbacks
DEFAULT_CONFIG = {
    # --- Model Configuration --- #
    "PRIMARY_MODEL_TYPE": "gemini",  # Options: "gemini", "claude", "local"
    "CLAUDE_MODEL_NAME": "claude-3-7-sonnet-20250219",
    "GEMINI_MODEL_NAME": "gemini-2.5-pro-exp-03-25",
    "LOCAL_MODEL_NAME": "mistral-7b-instruct-v0.2.Q4_K_M.gguf",  # Default local model
    "LOCAL_MODEL_QUANT_LEVEL": "Q4_K_M",  # Default quantization level
    "SUMMARIZER_MODEL": "gemini-2.0-flash",  # Default summarization model setting
    "SUMMARY_MAX_TOKENS": 1200,
    "FINAL_SUMMARY_MAX_TOKENS": 16000, # Max tokens for the final report
    # --- Next Step Model Configuration --- #
    "NEXT_STEP_MODEL": "gemini-2.5-flash-preview-04-17",  # Default next step model
    "NEXT_STEP_THINKING_DEFAULT": False,      # Default: do not use thinking mode unless planner requests
    
    # --- Local Model Configuration --- #
    "LOCAL_MODELS_DIR": "models",  # Directory for storing local models
    "N_GPU_LAYERS": -1, # Number of layers to offload to GPU (-1 = try all)

    # --- Content Management Settings --- #
    "USE_PROGRESSIVE_LOADING": True,
    "MAX_CONTENT_PREVIEW_TOKENS": 1000,
    "CHUNK_SIZE": 0,  # Default to 0 (no chunking) for gemini-2.0-flash
    "CHUNK_OVERLAP": 400,  # Default chunk overlap

    # --- Cache Configuration --- #
    "ENABLE_ADVANCED_CACHE": True,  # Enable the normalizing cache for better hit rates
    "CACHE_DB_PATH": ".langchain.db",  # Path to SQLite database for caching
    "CACHE_SCHEMA": "cache",  # Schema name for cache tables

    # --- CAPTCHA Settings --- #
    "USE_CAPTCHA_SOLVER": True,
    "CAPTCHA_SOLVER_TIMEOUT": 2000,

    # --- Agent Configuration --- #
    "ENABLE_THINKING": False, # Default for thinking feature
    "THINKING_BUDGET": 4000,   # Default budget if thinking enabled
    "MIN_REGULAR_WEB_PAGES": 1,
    "MAX_REGULAR_WEB_PAGES": 2,

    # --- Playwright/Browser Settings --- #
    "BROWSER_NAVIGATION_TIMEOUT": 15000,
    "STORAGE_STATE_PATH": "browser_state.json",

    # --- Logging --- #
    "LOG_LEVEL": "INFO",

    # --- Token Usage and Cost Tracking --- #
    "TRACK_TOKEN_USAGE": True,
    "CLAUDE_COST_PER_1K_INPUT_TOKENS": 0.008,
    "CLAUDE_COST_PER_1K_OUTPUT_TOKENS": 0.024,
    "GEMINI_COST_PER_1K_INPUT_TOKENS": 0.0,
    "GEMINI_COST_PER_1K_OUTPUT_TOKENS": 0.0,
    "GEMINI_FLASH_COST_PER_1K_INPUT": 0.00010,
    "GEMINI_FLASH_COST_PER_1K_OUTPUT": 0.00040,
    "GEMINI_25_FLASH_PREVIEW_COST_PER_1K_INPUT": 0.00015,
    "GEMINI_25_FLASH_PREVIEW_COST_PER_1K_OUTPUT": 0.00060,
    "LOG_COST_SUMMARY": True,
    
    # --- Defaults for potentially missing keys --- #
    "MEMORY_KEY": "history",
    "CONVERSATION_MEMORY_MAX_TOKENS": 3000,
    "TOOLS_CONFIG": {"web_browser": {"enabled": True}}, # Updated to be a dict for tools
    "MIN_POSTS_PER_SEARCH": 1,
    "MAX_POSTS_PER_SEARCH": 2,
    "MAX_RESULTS_PER_SEARCH_PAGE": 10,
    "MAX_CONTINUED_CONVERSATION_TOKENS": 32768
}

def load_config():
    """Load configuration from config.yaml in working directory.
    If keys are missing, fall back to default values.
    
    API keys are always loaded from environment variables.
    """
    # Start with default config values
    config = DEFAULT_CONFIG.copy()
    
    # Load user config if exists
    if USER_CONFIG_PATH.exists():
        try:
            with open(USER_CONFIG_PATH, "r") as f:
                user_config = yaml.safe_load(f)
            if user_config:
                # Merge user config with defaults, overwriting defaults
                for key, value in user_config.items():
                    config[key] = value # Update or add user's key
                logger.info(f"Loaded config from {USER_CONFIG_PATH}")
                # --- PATCH: Flatten 'tools' key if present --- #
                if 'tools' in user_config and isinstance(user_config['tools'], dict):
                    config['TOOLS_CONFIG'] = user_config['tools']
                    logger.info(f"Flattened 'tools' key from config.yaml into TOOLS_CONFIG: {list(config['TOOLS_CONFIG'].keys())}")
        except Exception as e:
            logger.error(f"Error loading config: {e}")
    else:
        logger.warning(f"No config.yaml found at {USER_CONFIG_PATH}. Using default values.")
    
    # Show important config settings
    print(f"Settings loaded:")
    print(f"- Primary Model Type: {config['PRIMARY_MODEL_TYPE']}")
    print(f"- Claude Model: {config['CLAUDE_MODEL_NAME']}")
    print(f"- Gemini Model: {config['GEMINI_MODEL_NAME']}")
    print(f"- Local Model: {config.get('LOCAL_MODEL_NAME', 'Not configured')}")
    print(f"- Summarizer Model: {config['SUMMARIZER_MODEL']}")
    print(f"- Search Results per Page: {config.get('MAX_RESULTS_PER_SEARCH_PAGE', 'N/A')} (min results: {config.get('MIN_REGULAR_WEB_PAGES', 'N/A')}, max results: {config.get('MAX_REGULAR_WEB_PAGES', 'N/A')})")

    return config

# Load the configuration
config = load_config()

# Export all config values as module variables
for key, value in config.items():
    globals()[key] = value

# --- Global Variables (Derived) --- #
# Only keep derived variables or those not in DEFAULT_CONFIG
GEMINI_MODEL = config.get('GEMINI_MODEL_NAME')
CLAUDE_MODEL = config.get('CLAUDE_MODEL_NAME')

# Derive LLM_PROVIDER and PRIMARY_MODEL_NAME based on PRIMARY_MODEL_TYPE
PRIMARY_MODEL_TYPE = config.get('PRIMARY_MODEL_TYPE', 'gemini')
if PRIMARY_MODEL_TYPE == 'local':
    PRIMARY_MODEL_NAME = config.get('LOCAL_MODEL_NAME')
    LLM_PROVIDER = 'local'  # Set for backwards compatibility
elif PRIMARY_MODEL_TYPE == 'claude':
    PRIMARY_MODEL_NAME = config.get('CLAUDE_MODEL_NAME')
    LLM_PROVIDER = 'claude'  # Set for backwards compatibility
else:  # default to gemini
    PRIMARY_MODEL_NAME = config.get('GEMINI_MODEL_NAME')
    LLM_PROVIDER = 'gemini'  # Set for backwards compatibility

# Use a VALID default if SUMMARIZER_MODEL is missing from config.yaml
SUMMARIZER_MODEL = config.get('SUMMARIZER_MODEL', 'gemini-2.0-flash') 
ENABLE_THINKING = config.get('ENABLE_THINKING', False)
THINKING_BUDGET = config.get('THINKING_BUDGET', 4000)
N_GPU_LAYERS = config.get('N_GPU_LAYERS', -1)
# --- Next Step Model (Agentic Decision) --- #
NEXT_STEP_MODEL = config.get('NEXT_STEP_MODEL', 'gemini-2.5-flash-preview-04-17')
NEXT_STEP_THINKING_DEFAULT = config.get('NEXT_STEP_THINKING_DEFAULT', False)

# Load API keys from environment (for security)
load_dotenv()  # Also load from .env file for API keys
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Local models configuration
# Derive USE_LOCAL_SUMMARIZER_MODEL based on the SUMMARIZER_MODEL setting
USE_LOCAL_SUMMARIZER_MODEL = not (SUMMARIZER_MODEL.startswith('gemini-') and 'flash' in SUMMARIZER_MODEL)
LOCAL_MODELS_DIR = Path(config.get('LOCAL_MODELS_DIR', 'models'))

# Check if the selected summarizer model is available locally if using local models
def check_summarizer_model_available():
    """Check if the selected summarizer model is available locally."""
    global SUMMARIZER_MODEL_AVAILABLE
    
    # Check if the model exists
    model_path = LOCAL_MODELS_DIR / SUMMARIZER_MODEL
    SUMMARIZER_MODEL_AVAILABLE = model_path.exists()
    
    # Log availability
    if USE_LOCAL_SUMMARIZER_MODEL: # Use the derived variable
        if SUMMARIZER_MODEL_AVAILABLE:
            logger.info(f"Local summarizer model {SUMMARIZER_MODEL} is available.")
        else:
            logger.warning(f"Local summarizer model {SUMMARIZER_MODEL} is not available. Falling back to cloud model for summarization.")
            # Optional: Force SUMMARIZER_MODEL to gemini-2.0-flash if local not found?
            # globals()['SUMMARIZER_MODEL'] = 'gemini-2.0-flash'
            # globals()['USE_LOCAL_SUMMARIZER_MODEL'] = False
    
    return SUMMARIZER_MODEL_AVAILABLE

# Initialize model availability
SUMMARIZER_MODEL_AVAILABLE = False

# Only check for local model if it's determined to be a local model
if USE_LOCAL_SUMMARIZER_MODEL:
    SUMMARIZER_MODEL_AVAILABLE = check_summarizer_model_available()
else:
    logger.info(f"Using cloud-based summarizer model: {SUMMARIZER_MODEL}")

# Critical checks
if not ANTHROPIC_API_KEY:
    # Allow running without Anthropic if Gemini is the provider
    if LLM_PROVIDER != 'gemini':
        raise ValueError("CRITICAL ERROR: ANTHROPIC_API_KEY environment variable not set (required for Claude provider).")
    else:
        logger.warning("ANTHROPIC_API_KEY not set, but using Gemini provider. Claude models will be unavailable.")

# Check for Gemini API Key if using Gemini provider or using Gemini Flash for summarization
if (LLM_PROVIDER == 'gemini' or not USE_LOCAL_SUMMARIZER_MODEL) and not GEMINI_API_KEY:
    logger.warning("GEMINI_API_KEY not set but Gemini provider or cloud summarization is enabled. Some features may not work properly.")

# Log token tracking settings if enabled
if config.get('TRACK_TOKEN_USAGE', False):
    claude_in = config.get('CLAUDE_COST_PER_1K_INPUT_TOKENS', 0.0)
    claude_out = config.get('CLAUDE_COST_PER_1K_OUTPUT_TOKENS', 0.0)
    gemini_in = config.get('GEMINI_COST_PER_1K_INPUT_TOKENS', 0.0)
    gemini_out = config.get('GEMINI_COST_PER_1K_OUTPUT_TOKENS', 0.0)
    print(f"Token tracking enabled: Claude (${claude_in:.5f}/${claude_out:.5f} per 1K i/o tokens), Gemini (${gemini_in:.5f}/${gemini_out:.5f} per 1K i/o tokens)")

# Log summarization model configuration
logger.info(f"Configuration loaded: LLM Provider={LLM_PROVIDER}, Primary Model={PRIMARY_MODEL_NAME}, Summary Model={SUMMARIZER_MODEL}, Use Local Summarizer={USE_LOCAL_SUMMARIZER_MODEL}")
logger.info(f"Tools Config: {TOOLS_CONFIG}")

# Ensure MIN_POSTS_PER_SEARCH and MAX_POSTS_PER_SEARCH are available as globals
MIN_POSTS_PER_SEARCH = config.get('MIN_POSTS_PER_SEARCH', 1)
MAX_POSTS_PER_SEARCH = config.get('MAX_POSTS_PER_SEARCH', 2)

# --- Content Condensation Settings --- #
CONDENSE_FREQUENCY = 1  # Condense after each new content addition

# --- Available Standard Tools Mapping --- #
# Define tools as string identifiers to avoid circular imports
AVAILABLE_TOOLS = {
    "web_browser": "src.browser.PlaywrightBrowserTool",
    "reddit_search": "src.tools.reddit_search_tool.RedditSearchTool",
    "reddit_extract_post": "src.tools.reddit_search_tool.RedditExtractPostTool",
}
logger.info(f"Defined AVAILABLE_TOOLS: {list(AVAILABLE_TOOLS.keys())}")

# === Debugging and Development ===
DEBUG_MODE = os.getenv("DEBUG_MODE", "False").lower() == "true"

# --- Further global settings --- #