import asyncio
import logging
import json
import random # <-- Added import
from typing import Any, Dict, List, Type, Optional, Callable, Awaitable, Union, TypeVar
import io # For handling bytes data with pymupdf
import os # Added import
import re # <-- Moved import here
import time
from datetime import datetime
import tempfile
import urllib.parse
import httpx # <<< Added import

# --- Tiktoken setup --- 
import tiktoken
TIKTOKEN_ENCODING = "cl100k_base"
# Cache the encoder globally for efficiency
try:
    encoding = tiktoken.get_encoding(TIKTOKEN_ENCODING)
except Exception as e:
     logger.warning(f"Failed to get tiktoken encoding '{TIKTOKEN_ENCODING}', falling back to p50k_base. Error: {e}")
     TIKTOKEN_ENCODING = "p50k_base"
     encoding = tiktoken.get_encoding(TIKTOKEN_ENCODING)
def get_token_count_for_text(text: str) -> int:
    """Uses tiktoken to count tokens for a given text."""
    if not text:
        return 0
    try:
        return len(encoding.encode(text))
    except Exception as e:
        logger.warning(f"Tiktoken encoding failed: {e}. Falling back to character count estimate.")
        return len(text) // 4 # Fallback estimate
# --- End Tiktoken setup ---

# Set up logger
logger = logging.getLogger(__name__)

from playwright.async_api import async_playwright, Page, Browser, Playwright, Response, BrowserContext, TimeoutError, Error # Import BrowserContext
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from readability import Document # Using readability-lxml
import markdownify # For converting HTML to Markdown
import pymupdf  # PyMuPDF's import name is now pymupdf, not fitz
import aiofiles
from bs4 import BeautifulSoup
# from src.content_manager import ContentManager # Remove direct import

# Import settings directly
from config.settings import (
    BROWSER_NAVIGATION_TIMEOUT, 
    USE_PROGRESSIVE_LOADING,
    USE_CAPTCHA_SOLVER,
    TRACK_TOKEN_USAGE,
    TOTAL_EXTRACTION_TIMEOUT
)

# Import BaseCallbackHandler for type hinting
from langchain_core.callbacks import BaseCallbackHandler
from src.chainlit_callbacks import ChainlitCallbackHandler # Import directly

# Check for Recognizer availability without initializing anything
try:
    from recognizer.agents.playwright import AsyncChallenger
    RECOGNIZER_AVAILABLE = True
    logger.info("Recognizer CAPTCHA solver package is available")
except ImportError:
    RECOGNIZER_AVAILABLE = False
    logger.warning("Recognizer package not found. Automatic CAPTCHA solving will not be available.")

# Disable CAPTCHA solver if requested in settings
if not USE_CAPTCHA_SOLVER:
    logger.info("Automatic CAPTCHA solving is disabled via configuration")

def extract_content_from_html(html: str) -> dict:
    """Extracts the main content from HTML using readability and converts to Markdown."""
    try:
        doc = Document(html)
        title = doc.title()
        content_html = doc.summary(html_partial=True)
        # Convert to Markdown, maybe simplify structure slightly
        content_md = markdownify.markdownify(content_html, heading_style="ATX")

        # Removed section/toc extraction

        return {
            "title": title,
            "full_content": f"# {title}\n\n{content_md}",
            # "toc": toc, # Removed
            # "sections": sections # Removed
        }
    except Exception as e:
        logger.warning(f"Readability/Markdownify failed: {e}", exc_info=True)
        # Fallback: return error as structured response
        return {
            "title": "Error during extraction",
            "full_content": "Error during HTML content extraction.",
            # "toc": ["Error"], # Removed
            # "sections": {"Error": "Error during HTML content extraction."} # Removed
        }

# Replace old _estimate_tokens with tiktoken wrapper
_estimate_tokens = get_token_count_for_text

def extract_content_from_pdf(pdf_bytes: bytes) -> dict:
    """Extracts text content from PDF bytes using PyMuPDF."""
    try:
        text_content = ""
        token_count = 0
        has_images = False
        image_count = 0
        text_blocks_count = 0
        
        logger.info("Starting PDF content extraction")
        start_time = time.time()
        
        with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
            # Get PDF metadata
            pdf_size_kb = len(pdf_bytes) / 1024
            total_pages = len(doc)
            logger.info(f"Processing PDF: {total_pages} pages, {pdf_size_kb:.1f} KB")
            
            # Extract text by page
            page_texts = []
            for page_num in range(total_pages):
                page = doc.load_page(page_num)
                
                # Check for images on the page
                page_images = page.get_images(full=True)
                if len(page_images) > 0:
                    has_images = True
                    image_count += len(page_images)
                
                # Get text blocks to determine if it's a text-based or image-based PDF
                blocks = page.get_text("blocks")
                text_blocks_count += len(blocks)
                
                # Extract page text
                page_text = page.get_text()
                page_texts.append(page_text)
                text_content += page_text
                
                # Add page break marker between pages
                if page_num < total_pages - 1:
                    text_content += "\n\n--- Page Break ---\n\n"
                
                # Log every 10 pages for large documents
                if page_num > 0 and page_num % 10 == 0:
                    # Estimate running total for logging purposes only
                    current_token_estimate = get_token_count_for_text(text_content)
                    logger.debug(f"Extracted {page_num}/{total_pages} PDF pages (~{current_token_estimate} tokens)")

        # Basic cleanup (optional)
        text_content = ' \n'.join([line.strip() for line in text_content.splitlines() if line.strip()])
        
        # Detect if PDF is likely scanned/image-based with little text
        is_mostly_images = False
        if has_images and (
            text_blocks_count < total_pages * 3 or  # Few text blocks per page
            len(text_content.strip()) < 100 * total_pages  # Very little text per page
        ):
            is_mostly_images = True
            logger.warning(f"PDF appears to be primarily image-based: {image_count} images, {text_blocks_count} text blocks, {len(text_content)} chars in {total_pages} pages")
        
        # Final token count for the cleaned content using tiktoken
        final_token_count = get_token_count_for_text(text_content)
        
        # Log extraction time
        extraction_time = time.time() - start_time
        logger.info(f"PDF extraction complete: {total_pages} pages, {final_token_count} tokens in {extraction_time:.2f}s")

        # If PDF is empty or mostly images, return appropriate message
        if not text_content.strip():
            return {
                "title": "PDF Document (Empty)",
                "full_content": "(PDF content appears empty or could not be extracted)",
                "toc": ["Empty Document"],
                "sections": {"Empty Document": "(PDF content appears empty or could not be extracted)"}
            }
        elif is_mostly_images:
            return {
                "title": "PDF Document (Primarily Images)",
                "full_content": f"# PDF Content (Primarily Image-Based)\n\nThis PDF appears to be primarily image-based or scanned. Limited text extracted:\n\n{text_content}",
                "toc": ["Image-Based PDF"],
                "sections": {"Image-Based PDF": f"This PDF contains {image_count} images and appears to be scanned or primarily graphics-based. Limited text could be extracted."}
            }

        return {
            "title": "PDF Document",
            "full_content": f"# PDF Content\n\n{text_content}",
            "toc": ["PDF Content"],
            "sections": {"PDF Content": text_content}
        }
    except Exception as e:
        logger.error(f"PyMuPDF failed to extract text from PDF: {e}", exc_info=True)
        return {
            "title": "Error during PDF extraction",
            "full_content": f"Error during PDF content extraction: {str(e)}",
            "toc": ["Error"],
            "sections": {"Error": f"Error during PDF content extraction: {str(e)}"}
        }

async def handle_captcha(page: Page, chainlit_callback: Optional[ChainlitCallbackHandler] = None):
    """Handles CAPTCHA detection and solving, using Chainlit UI for manual intervention.
    
    Args:
        page: The Playwright Page object.
        chainlit_callback: An optional ChainlitCallbackHandler instance for UI interaction.
    
    Returns:
        bool: True if CAPTCHA was handled (automatically or manually), 
              False if no CAPTCHA was detected or if context destroyed error occurred.
    """
    current_url = page.url # Get URL early for context
    try:
        # Basic detection - check for common CAPTCHA elements
        captcha_detected = False
        selectors = [
            'iframe[src*="recaptcha"]',
            'iframe[src*="hcaptcha"]',
            'div.g-recaptcha',
            'div.recaptcha', 
            'div[class*="captcha"]',
            '#captcha',
            '#recaptcha',
            'form[action*="captcha"]'
        ]
        
        # Check for visible CAPTCHA elements
        for selector in selectors:
            elements = await page.query_selector_all(selector)
            for element in elements:
                # Skip if not visible
                if not await element.is_visible():
                    continue
                    
                # Skip known background elements
                if selector == 'iframe[src*="recaptcha"]':
                    iframe_src = await element.get_attribute('src') or ""
                    if iframe_src == "https://www.google.com/recaptcha/api2/aframe" or "enterprise/anchor" in iframe_src:
                        continue
                
                # Found a likely CAPTCHA
                captcha_detected = True
                break
                
            if captcha_detected:
                break
        
        # If not found by elements, check page title
        if not captcha_detected:
            page_title = await page.title()
            for term in ['captcha', 'verify', 'verification', 'security check']:
                if term in page_title.lower():
                    captcha_detected = True
                    break
        
        # Only proceed if we've detected a CAPTCHA
        if not captcha_detected:
            return False
            
        # Log the detection
        logger.info(f"CAPTCHA detected on page: {current_url}")
        
        # If we have a valid challenger, try automatic solving
        if USE_CAPTCHA_SOLVER and RECOGNIZER_AVAILABLE and hasattr(page, 'browser_tool'):
            browser_tool = page.browser_tool
            if hasattr(browser_tool, 'captcha_challenger') and browser_tool.captcha_challenger:
                try:
                    logger.info("Attempting to solve CAPTCHA automatically with Recognizer")
                    # --- Send status to Chainlit BEFORE solving ---
                    if chainlit_callback:
                        # Use cl.Message directly if handler doesn't have a specific method
                        await cl.Message(content="🤖 Attempting automatic CAPTCHA solve...", author="System").send()
                    else:
                         print("\n [🤖] Attempting to solve CAPTCHA automatically...")
                    # --- End Chainlit status ---
                    
                    solved = await browser_tool.captcha_challenger.solve_recaptcha()
                    
                    if solved:
                        logger.info("CAPTCHA automatically solved successfully")
                        # --- Send success to Chainlit ---
                        if chainlit_callback:
                             await cl.Message(content="✅ CAPTCHA automatically solved!", author="System").send()
                        else:
                            print("\n [✅] CAPTCHA automatically solved!")
                        # --- End Chainlit status ---
                        return True # Solved automatically
                    else:
                        logger.warning("Automatic CAPTCHA solving failed or returned false. Falling back to manual.")
                except Exception as e:
                    logger.error(f"Error during automatic CAPTCHA solving: {e}. Falling back to manual.", exc_info=True)
        
        # Fall back to manual intervention using Chainlit if available
        logger.info("Falling back to manual CAPTCHA intervention.")
        if chainlit_callback:
            try:
                await chainlit_callback.ask_for_captcha_completion(captcha_page_url=current_url)
                return True # User confirmed completion in Chainlit
            except TimeoutError:
                 logger.error("User did not respond to Chainlit CAPTCHA prompt within timeout.")
                 return False # Treat timeout as failure to handle
            except Exception as chainlit_e:
                 logger.error(f"Error during Chainlit CAPTCHA prompt: {chainlit_e}. Falling back to console.", exc_info=True)
                 # Fall through to console input as last resort
        
        # --- Console Fallback (if no chainlit_callback or if it failed) ---
        print(f"\n [🚫] CAPTCHA detected - manual intervention needed (URL: {current_url})")
        print(" Please solve the CAPTCHA in the separate browser window.")
        print(" Then, press Enter in THIS CONSOLE window when you are done...")
        try:
             # Use asyncio.to_thread to run input() without blocking the event loop
             await asyncio.to_thread(input, " Press Enter to continue...")
             logger.info("User pressed Enter in console after manual CAPTCHA solve.")
             return True # User confirmed completion in console
        except Exception as input_e:
             logger.error(f"Error waiting for console input for CAPTCHA: {input_e}")
             return False # Failed to get confirmation
        # --- End Console Fallback ---
            
    except Error as ple:
        # Specifically catch Playwright errors
        if "Execution context was destroyed" in str(ple):
            logger.debug(f"Caught 'Execution context destroyed' error during CAPTCHA check (URL: {current_url}). Assuming navigation occurred and no CAPTCHA needed. Error: {ple}")
            return False # Indicate no manual intervention needed
        else:
            # Handle other Playwright errors
            logger.error(f"Playwright error during CAPTCHA handling (URL: {current_url}): {ple}", exc_info=True)
            # Ask for manual check via Chainlit if possible
            if chainlit_callback:
                try:
                    await chainlit_callback.ask_for_captcha_completion(captcha_page_url=current_url)
                    return True # User confirmed completion in Chainlit
                except Exception as chainlit_e:
                    logger.error(f"Error during Chainlit prompt after Playwright error: {chainlit_e}. Falling back to console.", exc_info=True)
            
            # Console Fallback
            print(f"\n [⚠️] Playwright error checking for CAPTCHA, please verify manually (URL: {current_url}).")
            print(" Press Enter in THIS CONSOLE window if there was a CAPTCHA and you solved it, OR if there was no CAPTCHA...")
            try:
                 await asyncio.to_thread(input, " Press Enter to continue...")
                 return True
            except Exception as input_e:
                 logger.error(f"Error waiting for console input after Playwright error: {input_e}")
                 return False
            
    except Exception as e:
        # Catch any other generic exceptions
        logger.error(f"Generic error in CAPTCHA handling (URL: {current_url}): {e}", exc_info=True)
        # Ask for manual check via Chainlit if possible
        if chainlit_callback:
            try:
                await chainlit_callback.ask_for_captcha_completion(captcha_page_url=current_url)
                return True # User confirmed completion in Chainlit
            except Exception as chainlit_e:
                logger.error(f"Error during Chainlit prompt after generic error: {chainlit_e}. Falling back to console.", exc_info=True)
                
        # Console Fallback
        print(f"\n [⚠️] Error checking for CAPTCHA, please verify manually (URL: {current_url})")
        print(" Press Enter in THIS CONSOLE window if there was a CAPTCHA and you solved it, OR if there was no CAPTCHA...")
        try:
             await asyncio.to_thread(input, " Press Enter to continue...")
             return True
        except Exception as input_e:
             logger.error(f"Error waiting for console input after generic error: {input_e}")
             return False

# --- Tool Input Schemas --- #

class NavigateInput(BaseModel):
    url: str = Field(..., description="The URL to navigate to.")

class NavigateAndExtractInput(BaseModel):
    url: str = Field(..., description="The URL to navigate to and extract content from.")

class ExtractContentInput(BaseModel):
    pass

class SearchInput(BaseModel):
    query: str = Field(..., description="The search query.")

class SearchPaginationInput(BaseModel):
    query: str = Field(..., description="The search query to continue with.")
    page: int = Field(default=2, description="The page number of search results to retrieve.")

class WebBrowserInput(BaseModel):
    """Schema for the web browser tool input."""
    action: str = Field(
        ..., 
        description="The action to perform.", 
        enum=["navigate", "extract_content", "extract", "search", "navigate_and_extract", "search_next_page"]
    )
    url: Optional[str] = Field(None, description="URL to navigate to (for navigate, extract, and navigate_and_extract actions)")
    query: Optional[str] = Field(None, description="Search query (for search and search_next_page actions)")
    page: Optional[int] = Field(None, description="Page number for search pagination (for search_next_page action)")

# --- Playwright Browser Tool --- #

class PlaywrightBrowserTool(BaseTool):
    """A LangChain-compatible browser tool using Playwright for web interactions."""
    
    name: str = "web_browser"
    description: str = (
        "A comprehensive web browser tool to interact with web pages and gather information. "
        "This tool can perform several actions including:\n"
        "1. 'search': Search the web for information on a topic\n"
        "2. 'navigate_and_extract': Navigate to a URL and extract its main content\n"
        "3. 'extract': Alias for 'navigate_and_extract'\n"
        "4. 'search_next_page': Get the next page of search results for the previous search\n\n"
        "When extracting content, it automatically handles different content types including HTML and PDF documents, "
        "converting them to readable text. The tool detects CAPTCHAs and "
        "provides a way for the user to solve them when needed."
    )
    args_schema: Type[BaseModel] = WebBrowserInput
    return_direct: bool = False

    # Annotate internal state attributes to avoid Pydantic conflicts
    playwright: Optional[Playwright] = None
    browser: Optional[Browser] = None
    context: Optional[BrowserContext] = None
    is_running: bool = False
    last_response: Optional[Response] = None
    last_extracted_content: Optional[dict] = None
    last_search_query: Optional[str] = None
    search_results_cache: Dict[str, List[Dict[str, str]]] = {}
    captcha_challenger: Optional[Any] = None
    # Add field for Chainlit callback handler
    chainlit_callback: Optional[Any] = None # Use Any for now to avoid circular import
    
    # --- Added attributes for ContentManager and Callbacks ---
    # Use string literal for type hint to break circular dependency
    content_manager: Optional['ContentManager'] = None 
    callbacks: Optional[List[BaseCallbackHandler]] = None
    # -------------------------------------------------------

    # --- Context state tracking ---
    context_is_closed: bool = True
    # ---------------------------------

    def __init__(self, content_manager: Optional['ContentManager'] = None, callbacks: Optional[List[BaseCallbackHandler]] = None, chainlit_callback: Optional[Any] = None, **kwargs): # Added **kwargs
        """Initialize the PlaywrightBrowserTool.
        
        Args:
            content_manager: Optional ContentManager for optimized content handling
            callbacks: Optional list of general callbacks for content preview etc.
            chainlit_callback: Optional specific ChainlitCallbackHandler for UI interactions.
        """
        print(">>> PlaywrightBrowserTool from src/browser.py __init__ called!")
        logger.info(">>> PlaywrightBrowserTool from src/browser.py __init__ called!")
        # Initialize Pydantic fields first using super().__init__
        # Pass only kwargs expected by BaseTool or its parent Pydantic BaseModel
        # Filter kwargs to avoid passing unexpected arguments like 'content_manager'
        base_kwargs = {k: v for k, v in kwargs.items() if k in self.model_fields}
        super().__init__(**base_kwargs)
        
        # Manually set internal state attributes (as they are excluded from Pydantic model_fields or handled separately)
        self.playwright = None
        self.browser = None
        self.context = None
        self.is_running = False
        self.last_response = None
        self.last_extracted_content = None
        self.last_search_query = None
        self.search_results_cache = {}
        self.captcha_challenger = None # Initialize here

        # --- Store provided arguments ---
        self.content_manager = content_manager 
        self.callbacks = callbacks or [] 
        self.chainlit_callback = chainlit_callback
        # ---------------------------------

        # --- Context state tracking ---
        self.context_is_closed = True
        # ---------------------------------

        # Initialize CAPTCHA Challenger if available
        if RECOGNIZER_AVAILABLE and USE_CAPTCHA_SOLVER:
            logger.debug("Initializing CAPTCHA Challenger (will be attached to page later)")
        else:
            logger.debug("CAPTCHA Challenger not initialized (Recognizer unavailable or disabled)")

    # --- Need to declare fields in the model for Pydantic v2 ---
    class Config:
        arbitrary_types_allowed = True

    # Explicitly declare model fields if using Pydantic v2 style within BaseTool
    # Mark internal state and custom init args with exclude=True
    # Use string literal for type hint here as well
    content_manager: Optional['ContentManager'] = Field(None, exclude=True) 
    callbacks: Optional[List[BaseCallbackHandler]] = Field(default_factory=list, exclude=True)
    chainlit_callback: Optional[Any] = Field(None, exclude=True)
    # Internal state fields (also need exclusion if not handled by BaseTool properly)
    playwright: Optional[Playwright] = Field(None, exclude=True)
    browser: Optional[Browser] = Field(None, exclude=True)
    context: Optional[BrowserContext] = Field(None, exclude=True)
    is_running: bool = Field(False, exclude=True)
    last_response: Optional[Response] = Field(None, exclude=True)
    last_extracted_content: Optional[dict] = Field(None, exclude=True)
    last_search_query: Optional[str] = Field(None, exclude=True)
    search_results_cache: Dict[str, List[Dict[str, str]]] = Field(default_factory=dict, exclude=True)
    captcha_challenger: Optional[Any] = Field(None, exclude=True)
    # --- End Pydantic field declarations ---

    async def _ensure_browser_running(self):
        """Initializes Playwright, browser, and context if not already running by using the BrowserManager singleton."""
        from src.browser_manager import browser_manager  # Import here to avoid circular imports
        browser_valid = False
        if self.is_running and self.browser:
            try:
                _ = self.browser.contexts
                browser_valid = True
                logger.debug("Browser is already running and connected.")
            except Exception as e:
                logger.warning(f"Browser appears to be disconnected: {e}. Will reinitialize.")
                self.is_running = False
                self.browser = None
                self.playwright = None
                self.context = None
                self.context_is_closed = True
        if not self.is_running or not browser_valid:
            logger.info("Initializing browser through BrowserManager...")
            try:
                self.browser = await browser_manager.initialize_browser()
                self.playwright = browser_manager.playwright
                self.is_running = True
                logger.info("Browser setup complete using shared browser instance.")
            except Exception as e:
                logger.error(f"Failed during _ensure_browser_running: {e}", exc_info=True)
                await self.clean_up()
                raise RuntimeError(f"Failed to initialize Playwright browser: {e}")
        # Ensure context is initialized and valid
        if not self.context or self.context_is_closed:
            logger.info("Creating new browser context for tab management...")
            self.context = await self.browser.new_context()
            self.context_is_closed = False
            # Attach event handler to set flag when context is closed
            def _on_context_close():
                logger.info("Browser context closed event received. Marking context as closed.")
                self.context_is_closed = True
            self.context.on("close", lambda _: _on_context_close())
            logger.info("Browser context created.")

    async def _human_like_scroll(self, page: Page):
        """Simulates human-like scrolling on the page, with network idle wait and a hard timeout."""
        logger.info("Performing human-like scrolling to trigger lazy-loaded content")
        try:
            page_height = await page.evaluate("document.body.scrollHeight")
            logger.info(f"Starting human-like scroll of page (height: {page_height}px)")
            # Pick a random percentage between 80 and 90
            scroll_percent = random.uniform(0.80, 0.90)
            target_scroll = int(page_height * scroll_percent)
            logger.info(f"Will scroll to approximately {int(scroll_percent*100)}% of the page ({target_scroll}px)")
            scroll_steps = 2
            start_time = asyncio.get_event_loop().time()
            for i in range(scroll_steps):
                # Each step goes a fraction of the way to the target_scroll
                scroll_position = int((i + 1) * target_scroll / scroll_steps)
                scroll_duration = random.uniform(0.5, 1.5)
                overshoot_position = scroll_position + random.randint(-100, 100)
                await page.evaluate(f"window.scrollTo({{ top: {overshoot_position}, behavior: 'smooth' }})")
                await asyncio.sleep(scroll_duration)
                if overshoot_position != scroll_position:
                    await page.evaluate(f"window.scrollTo({{ top: {scroll_position}, behavior: 'instant' }})")
                    await asyncio.sleep(0.2)
                pause_duration = random.uniform(0.25, 1.75)
                logger.debug(f"Scroll {i+1}/{scroll_steps} to position {scroll_position}px, pausing for {pause_duration:.2f}s")
                await asyncio.sleep(pause_duration)
                if asyncio.get_event_loop().time() - start_time > 18:
                    logger.warning("Human-like scrolling exceeded 18s, skipping remaining scrolls to allow for network idle wait.")
                    break
            # Scroll to the final target position (not all the way to the bottom)
            await page.evaluate(f"window.scrollTo({{ top: {target_scroll}, behavior: 'instant' }})")
            elapsed = asyncio.get_event_loop().time() - start_time
            max_wait = max(0.5, 20.0 - elapsed)
            try:
                logger.info(f"Waiting for network idle after scrolling (up to {max_wait:.2f}s)...")
                await asyncio.wait_for(page.wait_for_load_state('networkidle'), timeout=max_wait)
            except asyncio.TimeoutError:
                logger.warning("Network idle wait after scrolling timed out. Proceeding with extraction.")
            except Exception as e:
                logger.warning(f"Unexpected error during network idle wait: {e}")
            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed < 20.0:
                await asyncio.sleep(min(1.0, 20.0 - elapsed))
            logger.info("Finished human-like scroll and network idle wait.")
        except Error as e:
            if "Execution context was destroyed" in str(e):
                logger.warning(f"Warning during human-like scrolling: Execution context was destroyed (likely navigation). Proceeding with extraction.")
            else:
                logger.error(f"Error during human-like scrolling: {e}", exc_info=False)
        except asyncio.TimeoutError:
            logger.warning("Human-like scrolling timed out after 20s. Proceeding with extraction.")
        except Exception as e:
            logger.error(f"Unexpected error during human-like scrolling: {e}", exc_info=True)

    async def _extract_content(self, url: str) -> dict:
        """Extracts content from the specified URL."""
        await self._ensure_browser_running()
        if not self.browser or not self.context:
            logger.error("Browser or context is not available.")
            return {"title": "Error", "full_content": "Browser or context not available."}
        page = await self.context.new_page()
        current_url = url
        response = None
        
        try:
            # Navigate to the URL
            logger.info(f"Navigating to {url} for content extraction")
            response = await page.goto(url, wait_until="domcontentloaded", timeout=BROWSER_NAVIGATION_TIMEOUT * 1000)
            logger.info(f"Timeout: {BROWSER_NAVIGATION_TIMEOUT * 1000} seconds")
            if response and not response.ok:
                status = response.status
                error_msg = f"Navigation failed with status {status}. Cannot extract content."
                logger.error(error_msg)
                return {"title": f"Error {status}", "full_content": error_msg}
                
            # Check for CAPTCHA
            if await handle_captcha(page, self.chainlit_callback):
                logger.info("CAPTCHA handled. Proceeding with extraction.")
            
            # Get updated URL after potential redirects
            current_url = page.url
            title = await page.title()
            
            # Determine content type
            content_type = None
            if response:
                content_type = response.headers.get('content-type', '').lower()
                logger.info(f"Determined content type from headers: {content_type}")
            else:
                # Fallback detection based on URL
                if current_url.lower().endswith('.pdf') or '/pdf' in current_url.lower():
                    content_type = 'application/pdf'
                    logger.info(f"Assuming content type is PDF based on URL: {current_url}")
                else:
                    content_type = 'text/html'
                    logger.info(f"Assuming content type is HTML (fallback): {current_url}")
            
            # Perform human-like scrolling for better content loading
            await self._human_like_scroll(page)
            
            # Extract content based on type
            if content_type and 'application/pdf' in content_type:
                logger.info("Detected PDF content, downloading raw bytes...")
                pdf_bytes = None
                try:
                    # Download raw PDF bytes using httpx
                    async with httpx.AsyncClient(follow_redirects=True, timeout=30.0) as client:
                        headers = {
                            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                            'Accept': 'application/pdf,*/*',
                            'Accept-Language': 'en-US,en;q=0.9',
                            'Connection': 'keep-alive'
                        }
                        response = await client.get(current_url, headers=headers)
                        response.raise_for_status()
                        pdf_bytes = response.content
                        logger.info(f"Successfully downloaded {len(pdf_bytes)} bytes of PDF data from {current_url}")
                        
                    if pdf_bytes:
                        extracted_data = extract_content_from_pdf(pdf_bytes)
                        extracted_data['source_url'] = current_url
                    else:
                        logger.error("PDF download failed or returned empty bytes.")
                        extracted_data = {"title": "Error during PDF download", "full_content": "Failed to download PDF bytes."}

                except httpx.HTTPStatusError as http_err:
                    logger.error(f"HTTP error while downloading PDF from {current_url}: {http_err}", exc_info=True)
                    extracted_data = {"title": f"Error {http_err.response.status_code}", "full_content": f"HTTP error {http_err.response.status_code} while downloading PDF: {http_err}"}
                except Exception as download_err:
                    logger.error(f"Failed to download or process PDF from {current_url}: {download_err}", exc_info=True)
                    extracted_data = {"title": "Error during PDF download/processing", "full_content": f"Failed to download or process PDF: {download_err}"}
            
            elif content_type and ('text/html' in content_type or 'text/plain' in content_type or 'application/xhtml+xml' in content_type):
                logger.info("Detected HTML or Text content, extracting...")
                html_content = await page.content()
                extracted_data = extract_content_from_html(html_content)
                extracted_data['source_url'] = current_url
                
            elif content_type and 'application/json' in content_type:
                logger.info("Detected JSON content.")
                json_content = await page.evaluate("() => document.body.innerText")
                try:
                    # Try to pretty-print the JSON
                    json_obj = json.loads(json_content)
                    formatted_json = json.dumps(json_obj, indent=2)
                    extracted_data = {
                        "title": f"JSON Response from {current_url}",
                        "full_content": f"```json\n{formatted_json}\n```",
                        "source_url": current_url
                    }
                except:
                    # If parsing fails, return the raw text
                    extracted_data = {
                        "title": f"JSON Response from {current_url}",
                        "full_content": json_content,
                        "source_url": current_url
                    }
                
            elif content_type and 'text/plain' in content_type:
                logger.info("Detected Plain Text content.")
                text_content = await page.evaluate("() => document.body.innerText")
                extracted_data = {
                    "title": f"Plain Text from {current_url}",
                    "full_content": text_content,
                    "source_url": current_url
                }
                
            else:
                logger.warning(f"Unsupported content type '{content_type}' for URL {current_url}. Attempting generic text extraction.")
                try:
                    # Fallback: try getting the text content directly
                    body_text = await page.locator('body').text_content(timeout=5000)
                    if body_text:
                         extracted_data = {
                             "title": title or "Unknown Title", 
                             "full_content": body_text,
                             "source_url": current_url
                         }
                    else:
                         extracted_data = {
                            "title": title, 
                            "full_content": f"Unsupported content type: {content_type}. Could not extract text.", 
                            "source_url": current_url
                         }
                except Exception as generic_extract_err:
                    logger.error(f"Generic text extraction failed: {generic_extract_err}")
                    extracted_data = {
                        "title": title, 
                        "full_content": f"Unsupported content type: {content_type}. Extraction failed: {generic_extract_err}", 
                        "source_url": current_url
                    }

            # Store and return the extracted content
            self.last_extracted_content = extracted_data
            logger.info(f"Content extracted successfully from {current_url}. Title: {extracted_data.get('title', 'N/A')}")
            return extracted_data
            
        except Error as e:
            # Catch Playwright errors during content access
            if "Execution context was destroyed" in str(e):
                logger.error(f"Extraction failed for {current_url}: Context destroyed before content could be read. {e}", exc_info=False)
                return {"title": "Extraction Error", "full_content": f"Error: Extraction failed for {current_url}, page navigated away before content could be read.", "source_url": current_url}
            else:
                logger.error(f"Playwright error during content extraction for {current_url}: {e}", exc_info=True)
                return {"title": "Extraction Error", "full_content": f"Error: Playwright error during extraction: {e}", "source_url": current_url}
        except Exception as e:
            logger.error(f"Error during content extraction from {current_url}: {e}", exc_info=True)
            return {"title": "Error", "full_content": f"Error during content extraction: {e}", "source_url": current_url}
        finally:
            # Keep page open - don't close
            pass

    # --- Search Query Cleaning for Google ---
    def clean_search_query(self, query: str) -> str:
        """
        Clean and post-process a Google search query:
        - Remove leading/trailing quotes if the whole query is quoted and more than 2 words
        - Move site: operators outside of quotes if present
        - Remove double spaces
        - Avoid wrapping the whole query in quotes
        """
        q = query.strip()
        # Remove leading/trailing quotes if the whole query is quoted and more than 2 words
        if q.startswith('"') and q.endswith('"'):
            inner = q[1:-1].strip()
            if len(inner.split()) > 2:
                q = inner
        # Move site: operator outside of quotes if present
        site_match = re.search(r'site:[^\s"]+', q)
        if site_match:
            site_part = site_match.group(0)
            # Remove from inside quotes if present
            q = re.sub(r'"?\b' + re.escape(site_part) + r'\b"?', '', q).strip()
            # Remove any double spaces
            q = ' '.join(q.split())
            # Prepend site: operator at the start
            q = f"{site_part} {q}".strip()
        # Remove double spaces
        q = ' '.join(q.split())
        # Remove leading/trailing quotes again if still present
        if q.startswith('"') and q.endswith('"') and len(q.split()) > 2:
            q = q[1:-1].strip()
        return q

    async def _search(self, query: str, num_results: int = 20) -> Union[str, List[Dict[str, str]]]:
        """Performs a Google search using direct page interaction.
        
        This approach is more reliable against anti-bot measures:
        1. Navigates to Google homepage
        2. Enters search query through direct interaction 
        3. Parses results as an API might not be available
        
        Args:
            query: The search query
            num_results: Maximum number of results to return
            
        Returns:
            List of dictionaries containing search results (title, link, snippet) 
            OR an error string.
        """
        # Debug message - log more information about the search
        print(f"\n🔍 SEARCH DEBUG: About to search for '{query}'")
        print(f"- Requesting {num_results} results")
        print(f"- Using Google homepage interaction method")

        # Check if results are in cache (using renamed attribute)
        if query in self.search_results_cache:
            logging.info(f"Using cached search results for: '{query}'")
            results = self.search_results_cache[query]
            # Limit results from cache as well
            final_cached_results = results[:num_results]
            # <<< RETURN STRUCTURED DATA DIRECTLY FROM CACHE >>>
            logger.debug(f"Returning {len(final_cached_results)} cached results as list of dicts.")
            return final_cached_results 

        logger.info(f"Performing Google search for: '{query}' via homepage interaction.")
        
        # Ensure browser is running
        await self._ensure_browser_running()
        if not self.browser or not self.context:
            logger.error("Browser or context is not available.")
            return "Error: Browser or context is not available."
            
        # Create a new page for this search operation
        page = await self.context.new_page()
        
        try:
            # --- Step 1: Navigate to Google Homepage --- #
            homepage_url = "https://www.google.com/"
            logger.info(f"Navigating to Google homepage: {homepage_url}")
            
            # Navigate to Google homepage
            response = await page.goto(homepage_url, wait_until="domcontentloaded", timeout=BROWSER_NAVIGATION_TIMEOUT * 1000)
            if not response or not response.ok:
                status = response.status if response else "unknown"
                logger.error(f"Failed to navigate to Google homepage: Status {status}")
                return f"Error navigating to Google homepage: Status {status}"

            # --- Step 2: Find Search Box, Type Query, and Submit --- #
            try:
                logger.info("Finding search input and typing query...")
                # Common selector for the search textarea
                search_box_selector = 'textarea[name="q"]'
                search_box = page.locator(search_box_selector)
                await search_box.wait_for(state="visible", timeout=10000)

                # Clean the query for Google best practices
                cleaned_query = self.clean_search_query(query)
                await search_box.fill(cleaned_query)
                await asyncio.sleep(0.5) # Brief pause after typing
                await search_box.press("Enter")
                logger.info("Search submitted.")

            except Exception as e:
                logger.error(f"Error interacting with Google search box: {e}", exc_info=True)
                return f"Error performing search interaction: {str(e)}"

            # --- Step 3: Wait for Results and Parse --- #
            logger.info("Waiting for search results page URL...")
            try:
                # Wait for the URL to change to the search results URL pattern
                await page.wait_for_url("**/search?**", timeout=15000)
                logger.info("Search results page URL detected. Allowing time for content load...")
                # Add a fixed delay to allow results rendering after URL change
                await asyncio.sleep(5) # Increased delay - adjust if needed

                # Check if the URL indicates we might be on a CAPTCHA or verification page
                current_url = page.url
                page_title = await page.title()
                
                # Only check for CAPTCHAs if the page looks suspicious
                if (
                    any(keyword in current_url.lower() for keyword in ['captcha', 'verify', 'security', 'challenge']) or
                    any(keyword in page_title.lower() for keyword in ['captcha', 'verify', 'verification', 'security check', 'confirm'])
                ):
                    logger.info(f"Search results page looks suspicious, checking for CAPTCHAs: {page_title}")
                    await handle_captcha(page, self.chainlit_callback)
                
                # Also check for CAPTCHAs if we can't find search results
                has_results = await page.query_selector('div#search')
                if not has_results:
                    logger.warning("Search results container not found, checking for CAPTCHAs")
                    await handle_captcha(page, self.chainlit_callback)

            except Exception as e:
                 logger.warning(f"Did not detect search results URL or timed out waiting. Page might be unusual or loading slow. Attempting parse anyway. Error: {e}")
                 # Add a fallback delay even if URL didn't change as expected
                 await asyncio.sleep(3)

            # --- Add enhanced debugging --- #
            print(f"\n🔍 SEARCH RESULTS DEBUG: Extracting results from page for '{query}'")
            
            # Try to get the main content container as suggested
            main_content = await page.query_selector('div[role="main"]')
            if main_content:
                print(f"✅ Found main content container with role='main'")
                
                # Debug: Get a count of all links in the main content
                links_in_main = await main_content.query_selector_all('a')
                print(f"- Found {len(links_in_main)} links in the main content container")
                
            else:
                print(f"❌ Could not find main content container with role='main'")
                # If main content isn't found, cannot proceed with parsing
                return "Error: Could not find main content container (div[role='main']) to parse search results."
                
            # --- Parse search results using selectors --- #
            try:
                parsed_results_data = []
                print("Parsing Google results using new robust approach...")

                # Find the main content container first
                main_container = await page.query_selector('#main')
                if not main_container:
                    main_container = await page.query_selector('div[role="main"]')
                
                if not main_container:
                    return "Error: Could not find main search results container (#main or div[role='main']) to parse search results."

                # 1. Find all h3 elements in the main container (titles of search results)
                h3_elements = await main_container.query_selector_all('h3')
                print(f"- Found {len(h3_elements)} h3 elements (titles) in the main container.")

                for h3 in h3_elements:
                    try:
                        # 2. Extract the title text
                        title = await h3.inner_text()
                        
                        # 3. Find the closest ancestor <a> (not just parent)
                        link = await h3.evaluate('''
                            node => {
                                let el = node;
                                for (let i = 0; i < 5; i++) {
                                    if (!el) break;
                                    if (el.tagName && el.tagName.toLowerCase() === 'a') return el.href;
                                    el = el.parentElement;
                                }
                                // Fallback: use closest('a')
                                let a = node.closest('a');
                                return a ? a.href : null;
                            }
                        ''')
                        if not link:
                            continue  # Skip if no link found
                        
                        # 4. Find the containing element with the description text
                        snippet = None
                        a_element = await h3.evaluate_handle('node => node.closest("a")')
                        if a_element:
                            current_element = a_element
                            last_valid_element = a_element
                            for _ in range(5):
                                parent = await current_element.evaluate('node => node.parentElement')
                                if not parent:
                                    break
                                # Only traverse up through element nodes
                                is_element = await page.evaluate('node => node && node.nodeType === 1', parent)
                                if not is_element:
                                    break
                                h3_count = await page.evaluate('node => node.querySelectorAll && node.querySelectorAll("h3").length', parent)
                                if h3_count and h3_count > 1:
                                    # Use last_valid_element for snippet extraction
                                    break
                                last_valid_element = parent
                                current_element = parent
                            # Now extract snippet from last_valid_element
                            snippet = await page.evaluate('''
                                node => {
                                    if (!node || node.nodeType !== 1 || !node.childNodes) return '';
                                    const textContent = Array.from(node.childNodes)
                                        .filter(child => child.nodeType === 3)
                                        .map(child => child.textContent.trim())
                                        .join(' ');
                                    const childElements = node.querySelectorAll ? Array.from(node.querySelectorAll('*:not(h3):not(a)')) : [];
                                    const childText = childElements
                                        .map(el => el.textContent.trim())
                                        .join(' ');
                                    return (textContent + ' ' + childText).trim();
                                }
                            ''', last_valid_element)
                        if not snippet:
                            snippet = "(No description available)"
                        elif len(snippet) > 300:
                            snippet = snippet[:300] + "..."
                        if title and link:
                            parsed_results_data.append({
                                'title': title.strip(),
                                'link': link.strip(),
                                'snippet': snippet.strip()
                            })
                            print(f"[DEBUG] Appended result: {title.strip()} | {link.strip()}")
                            if len(parsed_results_data) >= num_results:
                                break
                    except Exception as item_error:
                        logger.warning(f"Error parsing individual search result: {item_error}")
                        continue  # Skip this result and continue with the next one

                if not parsed_results_data:
                    logger.warning("No search results found with new approach, falling back to old method.")
                    # Fall back to the original selector approach
                    search_div = await page.query_selector('div#search')
                    if search_div:
                        result_blocks = await search_div.query_selector_all(
                            "div[jscontroller][lang][jsaction][data-hveid][data-ved], div.g, div.Gx5Zad, div.tF2Cxc"
                        )
                        print(f"- Fallback: Found {len(result_blocks)} result blocks with traditional selectors.")
                        for block in result_blocks:
                            # Title from h3
                            title = None
                            try:
                                title_el = await block.query_selector('h3')
                                if title_el:
                                    title = await title_el.inner_text()
                            except Exception:
                                title = None

                            # Link from a:has(> h3)
                            link = None
                            try:
                                # Playwright does not support :has, so do it manually
                                a_tags = await block.query_selector_all('a')
                                for a_tag in a_tags:
                                    h3_child = await a_tag.query_selector('h3')
                                    if h3_child:
                                        link = await a_tag.get_attribute('href')
                                        break
                            except Exception:
                                link = None

                            # Description from [data-sncf='1']
                            snippet = None
                            try:
                                desc_el = await block.query_selector("[data-sncf='1']")
                                if desc_el:
                                    snippet = await desc_el.inner_text()
                            except Exception:
                                snippet = None

                            # Only add if title and link are present
                            if title and link:
                                parsed_results_data.append({
                                    'title': title.strip(),
                                    'link': link.strip(),
                                    'snippet': snippet.strip() if snippet else None
                                })
                                if len(parsed_results_data) >= num_results:
                                    break
                # After all parsing attempts, handle caching and return
                if not parsed_results_data:
                    print("[DEBUG] No results found after fallback, returning empty list.")
                    return []
                self.search_results_cache[query] = parsed_results_data
                print(f"[DEBUG] Returning {len(parsed_results_data)} results.")
                return parsed_results_data
            except Exception as e:
                logger.error(f"Error during parsing: {e}", exc_info=True)
                print(f"[DEBUG] Exception in _search: {e}")
                return []
        except Exception as e:
            logger.error(f"Unexpected error during search: {e}", exc_info=True)
            return f"Error: {str(e)}"
        finally:
            # Keep page open - don't close
            pass

    async def _navigate_and_extract(self, url: str) -> str:
        """Navigates to a URL and extracts the main content."""
        
        # Ensure browser is running
        await self._ensure_browser_running()
        if not self.browser or not self.context:
            logger.error("Browser or context is not available.")
            return "Error: Browser or context is not available."
            
        # Create a new page for this navigation and extraction
        page = await self.context.new_page()
        content_type = 'text/html'  # Default assumption
        
        try:
            # Navigate to the URL
            logger.info(f"Navigating to {url} for extraction")
            response = await page.goto(url, wait_until="domcontentloaded", timeout=BROWSER_NAVIGATION_TIMEOUT * 1000)
            
            # Check response
            if response and not response.ok:
                status = response.status
                logger.warning(f"Navigation failed with status {status} for URL: {url}")
                return f"Error: Navigation failed with status {status}"
            
            # Check for CAPTCHA after navigation
            if await handle_captcha(page, self.chainlit_callback):
                logger.info("CAPTCHA handled (or manual intervention needed). Proceeding with extraction.")
                
            # Determine content type
            if response:
                try:
                    content_type = response.headers.get("content-type", "").lower()
                    logger.info(f"Determined content type for {url}: {content_type}")
                except Exception as e:
                    logger.warning(f"Could not get headers from response for {url}: {e}. Assuming HTML.")
                    content_type = 'text/html'
            
            # Scroll for better content loading
            if USE_PROGRESSIVE_LOADING:
                await self._human_like_scroll(page)
                
            # Extract content based on type
            extracted_data = None
            try:
                if 'application/pdf' in content_type:
                    logger.info("Detected PDF content, extracting text...")
                    # Download PDF content using httpx
                    async with httpx.AsyncClient(follow_redirects=True, timeout=30.0) as client:
                        headers = {
                            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                            'Accept': 'application/pdf,*/*',
                            'Accept-Language': 'en-US,en;q=0.9',
                            'Connection': 'keep-alive'
                        }
                        pdf_response = await client.get(url, headers=headers)
                        pdf_response.raise_for_status()
                        pdf_bytes = pdf_response.content
                        
                        if pdf_bytes:
                            extracted_data = extract_content_from_pdf(pdf_bytes)
                        else:
                            return f"Error: Could not retrieve PDF content for {url} due to empty response."
                
                elif 'text/html' in content_type or 'application/xhtml+xml' in content_type:
                    logger.info("Detected HTML content, extracting main content...")
                    html_content = await page.content()
                    extracted_data = extract_content_from_html(html_content)
                
                elif 'application/json' in content_type:
                    logger.info("Detected JSON content.")
                    json_content = await page.evaluate("() => document.body.innerText")
                    try:
                        # Try to pretty-print the JSON
                        json_obj = json.loads(json_content)
                        formatted_json = json.dumps(json_obj, indent=2)
                        extracted_data = {
                            "title": f"JSON Response from {url}",
                            "full_content": f"```json\n{formatted_json}\n```"
                        }
                    except:
                        # If parsing fails, return the raw text
                        extracted_data = {
                            "title": f"JSON Response from {url}",
                            "full_content": json_content
                        }
                    
                elif 'text/plain' in content_type:
                    logger.info("Detected Plain Text content.")
                    text_content = await page.evaluate("() => document.body.innerText")
                    extracted_data = {
                        "title": f"Plain Text from {url}",
                        "full_content": text_content
                    }
                    
                else:
                    logger.warning(f"Unsupported content type '{content_type}' for URL: {url}")
                    
                    # Try generic extraction
                    try:
                        body_text = await page.locator('body').text_content(timeout=5000)
                        title = await page.title()
                        
                        if body_text:
                            extracted_data = {
                                "title": title or "Unknown Content",
                                "full_content": body_text
                            }
                        else:
                            return f"Error: Unsupported content type '{content_type}' and could not extract text"
                    except Exception as e:
                        return f"Error: Unsupported content type '{content_type}' and extraction failed: {e}"
            
            except Error as e:
                # Catch Playwright errors during content access
                if "Execution context was destroyed" in str(e):
                    logger.error(f"Extraction failed for {url}: Context destroyed before content could be read. {e}", exc_info=False)
                    return f"Error: Extraction failed for {url}, page navigated away before content could be read."
                else:
                    logger.error(f"Playwright error during content extraction for {url}: {e}", exc_info=True)
                    return f"Error: Playwright error during extraction: {e}"
            except Exception as e:
                logger.error(f"Unexpected error during content extraction for {url}: {e}", exc_info=True)
                return f"Error: Unexpected error during extraction: {e}"
            
            # Store and return the extracted content
            if extracted_data:
                self.last_extracted_content = extracted_data
                logger.info(f"Content extracted successfully from {url}. Title: {extracted_data.get('title')}")
                # Return only the full content string
                return extracted_data.get("full_content", "Error: Extracted data was empty.") 
            else:
                logger.warning(f"Extraction yielded no data for {url} (Content-Type: {content_type})")
                return f"Error: Could not extract content from {url} (Type: {content_type})"
                
        except Exception as e:
            logger.error(f"Error during navigate and extract for {url}: {e}", exc_info=True)
            return f"Error: {str(e)}"
        finally:
            # Keep page open - don't close
            pass

    async def _search_next_page(self, query: str, page_num: int = 2) -> str:
        """Gets the next page of search results for a query."""
        if not self.last_search_query:
            self.last_search_query = query
        elif query != self.last_search_query:
            logger.info(f"Search query changed: {self.last_search_query} -> {query}")
            self.last_search_query = query
        
        # Ensure browser is running
        await self._ensure_browser_running()
        if not self.browser or not self.context:
            logger.error("Browser or context is not available.")
            return "Error: Browser or context is not available."
            
        # Create a new page for this search pagination
        page = await self.context.new_page()
        
        try:
            logger.info(f"Searching for next page {page_num} of results for: '{query}'")
            
            # Construct a direct URL to the specific search results page
            encoded_query = query.replace(' ', '+')
            start_param = (page_num - 1) * 10  # Google uses 'start' parameter for pagination
            search_url = f"https://www.google.com/search?q={encoded_query}&start={start_param}"
            
            logger.info(f"Navigating directly to search results page {page_num} using URL: {search_url}")
            response = await page.goto(search_url, wait_until="domcontentloaded", timeout=BROWSER_NAVIGATION_TIMEOUT * 1000)
            
            # Check response status
            if response and not response.ok:
                status = response.status
                logger.warning(f"Navigation to search page {page_num} failed with status {status}")
                return f"Error: Navigation to search page {page_num} failed with status {status}"
            
            # Check for CAPTCHA after navigation
            if await handle_captcha(page, self.chainlit_callback):
                logger.info(f"CAPTCHA handled on search page {page_num}. Proceeding with extraction.")
            
            # Wait for the results to stabilize
            await asyncio.sleep(2)
            await self._human_like_scroll(page)
            
            # Extract search results
            results = []
            
            # Try multiple result selectors as Google's HTML structure can vary
            selectors = [
                'div.g', 
                'div.Gx5Zad', 
                'div.tF2Cxc',
                'div[data-sokoban-container]'
            ]
            
            # Try each selector until we find results
            result_elements = []
            for selector in selectors:
                result_elements = await page.query_selector_all(selector)
                if result_elements and len(result_elements) > 0:
                    logger.info(f"Found {len(result_elements)} search results using selector '{selector}'")
                    break
            
            if not result_elements or len(result_elements) == 0:
                logger.warning("No search results found with standard selectors, trying alternative approach")
                # Fallback to looking for links with substantial text content
                all_links = await page.query_selector_all('a')
                parsed_results_data = []
                
                for link_element in all_links:
                    href = await link_element.get_attribute('href')
                    if not href or not href.startswith('http'): continue
                    if any(skip_part in href for skip_part in [
                        'google.com/search?', 'google.com/maps', 'google.com/preferences',
                        'google.com/account', 'google.com/advanced_search', 'google.com/alerts',
                        'support.google.com', 'policies.google.com', 'accounts.google.com'
                    ]): continue
                    if "&adurl=" in href or await link_element.query_selector('[data-text-ad="1"]'): continue

                    title = None
                    h3_title = await link_element.query_selector('h3')
                    if h3_title:
                        title_text = await h3_title.inner_text()
                        if title_text: title = title_text.strip()
                    if not title:
                        link_text = await link_element.inner_text()
                        if link_text and len(link_text) > 15 and not link_text.startswith('http'):
                            title = link_text.strip()
                    if not title: continue

                    snippet = "(Snippet not found)"
                    try:
                        container = await link_element.query_selector('xpath=./ancestor::div[string-length(normalize-space(.)) > 50][1]')
                        if container:
                            container_text = await container.inner_text()
                            snippet_text = container_text.replace(title, "").strip()
                            if len(snippet_text) > 10:
                                snippet = snippet_text[:300] + ("..." if len(snippet_text) > 300 else "")
                    except Exception as snip_e:
                        logger.debug(f"Failed to find snippet for link {href}: {snip_e}")

                    if not any(d['link'] == href for d in parsed_results_data):
                        parsed_results_data.append({"title": title, "link": href, "snippet": snippet})
                    if len(parsed_results_data) >= 20: break
                
                if parsed_results_data:
                    for i, data in enumerate(parsed_results_data):
                        results.append(f"{i+1}. {data['title']}\nURL: {data['link']}\nSnippet: {data['snippet']}\n")
            else:
                # Process regular search results
                for i, result in enumerate(result_elements):
                    if i >= 20:
                        break
                        
                    try:
                        title_element = await result.query_selector('h3')
                        link_element = await result.query_selector('a')
                        snippet_element = await result.query_selector('div.VwiC3b, div.lEBKkf, div[role="textbox"]')
                        
                        title = await title_element.inner_text() if title_element else "No title"
                        url = await link_element.get_attribute('href') if link_element else "No URL"
                        snippet = await snippet_element.inner_text() if snippet_element else "No snippet"
                        
                        results.append(f"{i+1}. {title}\nURL: {url}\nSnippet: {snippet}\n")
                    except Exception as e:
                        logger.warning(f"Error extracting search result {i}: {e}")
            
            if not results:
                return f"No search results found on page {page_num} for query: {query}"
                
            return f"Search results for '{query}' (Page {page_num}):\n\n" + "\n".join(results)
        except Exception as e:
            logger.error(f"Error in search_next_page: {e}", exc_info=True)
            return f"Error performing search: {str(e)}"
        finally:
            # Keep page open - don't close
            pass

    async def arun(
        self,
        tool_input: Dict[str, Any],
        callbacks: Optional[List[BaseCallbackHandler]] = None # Add callbacks param
    ) -> Union[str, List[Dict[str, str]]]: # <<< UPDATED RETURN TYPE
        """Asynchronously run the tool, preparing the input based on the action type.
           This is the main entrypoint for tool execution. Returns string or list of dicts.
        """
        print(f">>> PlaywrightBrowserTool.arun() CALLED with input: {tool_input}")
        logger.info(f">>> PlaywrightBrowserTool.arun() CALLED with input: {tool_input}")
        
        # Initialize input_data
        input_data = {}

        if isinstance(tool_input, str):
            # Handle legacy string input format by assuming it's a simple search
            try:
                input_data = json.loads(tool_input)
            except json.JSONDecodeError:
                # Basic string is treated as a search query
                input_data = {"action": "search", "query": tool_input}
        elif isinstance(tool_input, dict): # Ensure tool_input is a dict
            input_data = tool_input
        else:
             # Handle unexpected input type
             err_msg = f"Error: Unexpected tool input type {type(tool_input)}. Expected dict or JSON string."
             logger.error(err_msg)
             return err_msg # Return error string immediately
            
        # Standardize the format
        action = input_data.get("action", "")
        
        # Add alias for "extract" to "navigate_and_extract"
        if action == "extract":
            logger.info("Converting 'extract' action alias to 'navigate_and_extract'")
            action = "navigate_and_extract"
            
        # Handle cases where action might be inferred
        if not action and "url" in input_data:
            # Handle direct URL input for navigation/extraction
            action = "navigate_and_extract"
        elif not action and ("search_query" in input_data or "query" in input_data):
            # Handle direct search query input
            if "page" in input_data:
                action = "search_next_page"
            else:
                action = "search"
        
        try:
            if action == "navigate_and_extract":
                url = input_data.get("url")
                if not url:
                    return "Error: URL is required for navigation"
                # --- GLOBAL TIMEOUT PATCH ---
                import asyncio
                try:
                    return await asyncio.wait_for(self._navigate_and_extract(url), timeout=TOTAL_EXTRACTION_TIMEOUT)
                except asyncio.TimeoutError:
                    logger.error(f"Extraction timed out after {TOTAL_EXTRACTION_TIMEOUT} seconds for URL: {url}")
                    # Return any content that was extracted before the timeout, if available
                    if self.last_extracted_content and self.last_extracted_content.get("full_content"):
                        return self.last_extracted_content["full_content"]
                    return ""
                # --- END PATCH ---
                
            elif action == "search":
                query = input_data.get("query") or input_data.get("search_query")
                if not query:
                    return "Error: Search query is required"
                num_results = input_data.get("num_results", 20)
                # _search can return str or List[Dict]
                search_result = await self._search(query, num_results) 
                # Return the result directly (str or list)
                return search_result 
                
            elif action == "search_next_page":
                query = input_data.get("query") or input_data.get("search_query")
                page = input_data.get("page", 2)
                if not query:
                    return "Error: Search query is required for pagination"
                # _search_next_page returns str
                return await self._search_next_page(query, page)
            
            elif action == "extract_content":
                url = input_data.get("url")
                if not url:
                    return "Error: URL is required for extraction"
                # --- GLOBAL TIMEOUT PATCH ---
                import asyncio
                try:
                    return await asyncio.wait_for(self._extract_content(url), timeout=TOTAL_EXTRACTION_TIMEOUT)
                except asyncio.TimeoutError:
                    logger.error(f"Extraction timed out after {TOTAL_EXTRACTION_TIMEOUT} seconds for URL: {url}")
                    # Return any content that was extracted before the timeout, if available
                    if self.last_extracted_content and self.last_extracted_content.get("full_content"):
                        return self.last_extracted_content["full_content"]
                    return ""
                # --- END PATCH ---
            
            else:
                return f"Error: Unknown action '{action}'. Supported actions: navigate_and_extract, extract (alias for navigate_and_extract), search, search_next_page"
                
        except Exception as e:
            logger.error(f"Error executing browser action '{action}': {e}", exc_info=True)
            # Ensure error return is a string
            return f"Error executing browser action: {e}"

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("Use arun for asynchronous Playwright operations.")

    async def clean_up(self):
        """Cleanup method now only cleans up the shared browser if this is the last tool requiring it."""
        logger.info("PlaywrightBrowserTool clean_up called")
        self.is_running = False
        self.browser = None
        self.playwright = None
        self.context = None
        self.context_is_closed = True
        # We could potentially check if any other tools are using the browser
        # and call browser_manager.clean_up() if not, but that would require 
        # additional tracking mechanisms. 