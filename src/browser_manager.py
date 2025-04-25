import logging
import asyncio
from typing import Optional, Dict
from playwright.async_api import async_playwright, Browser, Playwright

logger = logging.getLogger(__name__)

class BrowserManager:
    """
    A singleton class to manage a single Playwright browser instance across the application.
    Provides browser access to different tools while ensuring only one instance is running.
    """
    _instance = None
    _lock = asyncio.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(BrowserManager, cls).__new__(cls)
            cls._instance.playwright = None
            cls._instance.browser = None
            cls._instance.is_running = False
            cls._instance._cleanup_in_progress = False
            cls._instance._browser_options = {
                "headless": False,
                "channel": "chrome",
                "args": ["--disable-blink-features=AutomationControlled", "--start-maximized", "--mute-audio"]
            }
        return cls._instance
    
    @property
    def is_initialized(self) -> bool:
        """
        Check if the browser is initialized and running.
        """
        return self.is_running and self.browser is not None
    
    async def initialize_browser(self) -> Browser:
        """
        Initialize the browser if not already running.
        Returns the browser instance.
        """
        async with self._lock:
            if self._cleanup_in_progress:
                # If browser is being cleaned up, wait until it's done
                while self._cleanup_in_progress:
                    logger.info("Waiting for browser cleanup to complete...")
                    await asyncio.sleep(0.5)
            
            if not self.is_running:
                logger.info("Initializing shared Playwright browser instance...")
                try:
                    self.playwright = await async_playwright().start()
                    self.browser = await self.playwright.chromium.launch(**self._browser_options)
                    self.is_running = True
                    logger.info("Shared browser instance initialized successfully")
                except Exception as e:
                    logger.error(f"Failed to initialize browser: {e}", exc_info=True)
                    self.is_running = False
                    self.browser = None
                    self.playwright = None
                    raise RuntimeError(f"Failed to initialize Playwright browser: {e}")
            
            return self.browser
    
    async def clean_up(self):
        """
        Clean up the browser instance when no longer needed.
        """
        async with self._lock:
            if self.is_running and not self._cleanup_in_progress:
                self._cleanup_in_progress = True
                logger.info("Cleaning up shared browser instance...")
                try:
                    if self.browser:
                        # Check if there are any open pages
                        try:
                            contexts = self.browser.contexts
                            all_pages = []
                            for context in contexts:
                                pages = context.pages
                                all_pages.extend(pages)
                            
                            if all_pages:
                                # If pages are still open, don't close the browser
                                page_count = len(all_pages)
                                logger.warning(f"Browser cleanup aborted: {page_count} pages still active. Browser will be kept running.")
                                self._cleanup_in_progress = False
                                return
                        except Exception as e:
                            logger.warning(f"Failed to check for open pages: {e}")
                            # Continue with cleanup even if check fails
                            
                        # No active pages found, proceed with cleanup
                        await self.browser.close()
                        self.browser = None
                        logger.info("Browser closed successfully")
                    
                    if self.playwright:
                        await self.playwright.stop()
                        self.playwright = None
                        logger.info("Playwright stopped successfully")
                    
                    self.is_running = False
                except Exception as e:
                    logger.error(f"Error during browser cleanup: {e}", exc_info=True)
                finally:
                    self._cleanup_in_progress = False
    
    def get_browser(self) -> Optional[Browser]:
        """
        Get the current browser instance if available.
        
        Returns:
            The browser instance or None if not initialized.
        """
        if self.is_running and self.browser:
            return self.browser
        return None
    
    def set_browser_options(self, options: Dict):
        """
        Set browser launch options. Must be called before initialize_browser.
        
        Args:
            options: Dictionary of options to pass to playwright.chromium.launch()
        """
        if not self.is_running:
            self._browser_options.update(options)
            logger.info(f"Browser options updated: {self._browser_options}")
        else:
            logger.warning("Cannot update browser options while browser is running")

    async def close_all_pages(self):
        """
        Close all open pages but keep the browser running.
        This can be used periodically to clean up without closing the browser.
        """
        if not self.is_running or not self.browser:
            logger.warning("No browser running, cannot close pages")
            return
            
        try:
            page_count = 0
            contexts = self.browser.contexts
            for context in contexts:
                pages = context.pages
                for page in pages:
                    try:
                        await page.close()
                        page_count += 1
                    except Exception as e:
                        logger.warning(f"Error closing page: {e}")
            
            if page_count > 0:
                logger.info(f"Closed {page_count} pages while keeping browser running")
        except Exception as e:
            logger.error(f"Error during page cleanup: {e}", exc_info=True)

# Singleton instance
browser_manager = BrowserManager() 