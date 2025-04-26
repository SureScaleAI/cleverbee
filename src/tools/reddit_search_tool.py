import asyncio
from typing import Optional, Dict, Any, Type
from pydantic import BaseModel, Field
from src.browser import PlaywrightBrowserTool
from playwright.async_api import Page, Error as PlaywrightError
import logging
import re
import math
import json
from src.utils import strip_class_attributes, clean_reddit_html
import urllib.parse

logger = logging.getLogger(__name__)

class RedditSearchInput(BaseModel):
    query: str = Field(..., description="The search query for Reddit.")
    extract_result_index: Optional[int] = Field(None, description="Optional: If provided (e.g., 1 for first result), extract the post and comments from the specified search result link instead of just returning the list.")

class RedditSearchTool(PlaywrightBrowserTool):
    name: str = "reddit_search"
    description: str = (
        "Searches Reddit using Playwright. Goes to reddit.com, types the query, "
        "and returns a list of post snippets/links. Optionally, if 'extract_result_index' is provided (e.g., 1), "
        "it extracts the content (post/comments HTML) of that result using the same browser session. "
        "Advanced: Query can use boolean operators (AND, OR, NOT - case sensitive), grouping (), and field searches (e.g., `title:\"some phrase\"`). Don't get too complex."
    )
    args_schema: Type[BaseModel] = RedditSearchInput
    return_direct: bool = False

    def _parse_comment_count(self, text: str) -> int:
        """Parse comment count text like '123', '1.2k', '5,000' into an integer."""
        text = text.lower().strip()
        num_str = re.sub(r'[^\d.]', '', text) # Keep only digits and decimal point
        
        multiplier = 1
        if 'k' in text:
            multiplier = 1000
           
        try:
            value = float(num_str) * multiplier
            return math.floor(value) # Use floor to handle potential float artifacts
        except ValueError:
            logger.warning(f"[RedditSearch] Could not parse '{text}' into a number.")
            return 0 # Return 0 if parsing fails

    async def _apply_filter(self, page: Page, button_selector: str, menu_selector: str, value: str, url_param: str, url_value: str) -> bool:
        try:
            btn = await page.query_selector(button_selector)
            if not btn:
                logger.warning(f"[RedditSearch] Filter button '{button_selector}' not found on {page.url}")
                return False
            visible = await btn.is_visible()
            logger.info(f"[RedditSearch] Filter button '{button_selector}' found. Visible: {visible}")
            await btn.click()
            logger.info(f"[RedditSearch] Clicked button '{button_selector}'.")

            # --- New Link Finding Logic ---
            param_pattern_exact = f"{url_param}={url_value}"
            link_selector_query = f'a[href*="?{param_pattern_exact}"], a[href*="&{param_pattern_exact}"]'
            logger.info(f"[RedditSearch] Creating locator for link matching selector: '{link_selector_query}'")

            # *** Create a Locator for the link instead of finding handle immediately ***
            target_link_locator = page.locator(link_selector_query).first # Use .first to target one

            # Optional: Wait briefly for the locator to potentially find the element
            try:
                 await target_link_locator.wait_for(state="attached", timeout=5000)
                 logger.info(f"[RedditSearch] Locator found an attached element for '{link_selector_query}'.")
            except Exception as e:
                 logger.warning(f"[RedditSearch] Locator could not find attached element for '{link_selector_query}' within timeout: {e}")
                 return False

            # --- Link potentially exists, proceed to click within expect_navigation ---
            logger.info(f"[RedditSearch] Proceeding to click locator: '{link_selector_query}'.")

            # Click the locator and wait for navigation
            try:
                logger.info(f"[RedditSearch] Starting expect_navigation for locator click...")
                async with page.expect_navigation(timeout=15000):
                    logger.info(f"[RedditSearch] Executing locator.click(force=True)...")
                    # *** Click the LOCATOR, not the handle ***
                    await target_link_locator.click(force=True)
                    logger.info(f"[RedditSearch] locator.click(force=True) finished.")
                logger.info(f"[RedditSearch] expect_navigation finished.")
            except Exception as e:
                logger.error(f"[RedditSearch] Exception during locator.click() or navigation wait: {e}", exc_info=True)
                return False

            # Wait for URL to contain the correct filter parameter
            wait_pattern = f"**/*{url_param}={url_value}*"
            logger.info(f"[RedditSearch] Waiting for URL pattern: '{wait_pattern}'")
            try:
                await page.wait_for_url(wait_pattern, timeout=15000)
                logger.info(f"[RedditSearch] URL contains '{url_param}={url_value}'. Current URL: {page.url}")
            except Exception as e:
                logger.warning(f"[RedditSearch] Timed out waiting for URL pattern '{wait_pattern}': {e}")
                return False

            return True # Filter applied successfully

        except Exception as e:
            logger.error(f"[RedditSearch] Exception in _apply_filter: {e}", exc_info=True) # Log full traceback
            return False

    async def _search(self, query: str, extract_result_index: Optional[int] = None) -> tuple[str | dict, Optional[str]]:
        await self._ensure_browser_running()
        homepage_url = "https://www.reddit.com/"
        
        # --- Clean the query ---
        original_query = query
        cleaned_query = query
        
        # Define problematic terms to remove/filter
        # These terms can lead to poor search results when included in Reddit search
        problematic_terms = ['reddit', 'discussion']
        
        # 1. Process quoted phrases to extract meaningful keywords
        # Examples:
        # - "mcp development discussion" -> "Kambo safety"
        # - "reddit data surfer" -> "plant medicine"
        def replace_quoted_phrase(match):
            phrase = match.group(1)
            # If the phrase contains problematic terms, extract meaningful words
            if any(re.search(rf'\b{term}\b', phrase, re.IGNORECASE) for term in problematic_terms):
                # Split the phrase into words
                words = re.findall(r'\b\w+\b', phrase)
                # Filter out problematic words
                filtered_words = [word for word in words if not any(
                    re.match(rf'\b{term}\b', word, re.IGNORECASE) for term in problematic_terms)]
                # If we have meaningful words left, join them and return
                if filtered_words:
                    return ' '.join(filtered_words)
                return ''
            # Not a problematic phrase, return as is
            return f'"{phrase}"'
        
        # Process quoted phrases first
        cleaned_query = re.sub(r'"([^"]+)"', replace_quoted_phrase, cleaned_query)
        
        # 2. Remove special site search patterns
        # Example: "site:reddit.com Kambo" -> "Kambo"
        cleaned_query = re.sub(r'\(?\s*site\s*:\s*reddit\.com\s*\)?', '', cleaned_query, flags=re.IGNORECASE)
        
        # 3. Remove standalone problematic terms
        # Examples:
        # - "best reddit posts" -> "best posts"
        # - "Kambo safety discussion" -> "Kambo safety"
        for term in problematic_terms:
            cleaned_query = re.sub(rf'\b{term}\b', '', cleaned_query, flags=re.IGNORECASE)
        
        # 4. Clean up remaining artifacts
        # Remove any dangling boolean operators (OR, AND, NOT) at the start/end 
        cleaned_query = re.sub(r'\b(OR|AND|NOT)\b\s*$', '', cleaned_query, flags=re.IGNORECASE)
        cleaned_query = re.sub(r'^\s*\b(OR|AND|NOT)\b', '', cleaned_query, flags=re.IGNORECASE)
        # Remove repeated spaces and stray parentheses
        cleaned_query = re.sub(r'\s+', ' ', cleaned_query)
        cleaned_query = re.sub(r'\(\s*\)', '', cleaned_query)
        cleaned_query = cleaned_query.strip()
        
        if cleaned_query != original_query:
            logger.info(f"[RedditSearch] Cleaned query from '{original_query}' to '{cleaned_query}'")
            query = cleaned_query # Use the cleaned query
        # -----------------------

        # Create a new page for this search operation
        page = await self.browser.new_page()
        try:
            # Navigate to Reddit
            await page.goto(homepage_url, wait_until="domcontentloaded", timeout=30000)
            logger.info(f"Successfully navigated to Reddit homepage")
            
            # Wait for the page to stabilize
            await asyncio.sleep(1)
            
            # Try multiple selectors for the search input (Reddit's UI might vary)
            search_selectors = [
                "input[placeholder*='Search']", 
                "input[type='search']",
                "input[aria-label*='search']",
                "input[data-testid='search-input']"
            ]
            
            # Helper to safely count locators with retry on execution context errors
            async def safe_count(locators, retries=3, delay=0.5):
                for attempt in range(retries):
                    try:
                        return await locators.count()
                    except Exception as e:
                        if "Execution context was destroyed" in str(e) and attempt < retries - 1:
                            await asyncio.sleep(delay)
                            continue
                        raise

            # Wait for any search input to become visible after navigation (up to 10s)
            search_ready = False
            for selector in search_selectors:
                try:
                    locator = page.locator(selector)
                    await locator.wait_for(state="visible", timeout=10000)
                    search_ready = True
                    break
                except Exception:
                    continue
            if not search_ready:
                logger.warning("No search input became visible after navigation, proceeding anyway.")

            search_input = None
            for selector in search_selectors:
                locators = page.locator(selector)
                count = await safe_count(locators)
                for i in range(count):
                    locator = locators.nth(i)
                    try:
                        if await locator.is_visible():
                            search_input = locator
                            logger.info(f"Found visible search input using locator: {selector} (index {i})")
                            break
                    except Exception as e:
                        logger.debug(f"Locator {selector} (index {i}) visibility check failed: {e}")
                        continue
                if search_input:
                    break
            
            if not search_input:
                # If search input not found, try to navigate directly to the search URL
                search_url = f"https://www.reddit.com/search/?q={urllib.parse.quote(query)}"
                logger.info(f"Search input not found, navigating directly to search URL: {search_url}")
                await page.goto(search_url, wait_until="domcontentloaded", timeout=30000)
                # Wait for any search input to become visible after navigation (up to 10s)
                search_ready = False
                for selector in search_selectors:
                    try:
                        locator = page.locator(selector)
                        await locator.wait_for(state="visible", timeout=10000)
                        search_ready = True
                        break
                    except Exception:
                        continue
                if not search_ready:
                    logger.warning("No search input became visible after search URL navigation, proceeding anyway.")
                # After navigation, try to find the search input again (in case we want to interact further)
                for selector in search_selectors:
                    locators = page.locator(selector)
                    count = await safe_count(locators)
                    for i in range(count):
                        locator = locators.nth(i)
                        try:
                            if await locator.is_visible():
                                search_input = locator
                                logger.info(f"Found visible search input after navigation using locator: {selector} (index {i})")
                                break
                        except Exception as e:
                            logger.debug(f"Locator {selector} after navigation (index {i}) visibility check failed: {e}")
                            continue
                    if search_input:
                        break
                # If still not found, just proceed (search URL should show results)
            else:
                # Found the search input, use it
                await search_input.fill(query)
                await asyncio.sleep(0.5)
                await search_input.press("Enter")
            
            logger.info("[RedditSearch] Waiting 5 seconds for search results page to settle...")
            await asyncio.sleep(5)  # Give extra time for UI to settle
            logger.info("[RedditSearch] Wait finished. Proceeding with post extraction.")
            
            # --- Final URL and Status --- #
            url = page.url

            # --- Extract posts using structural locators --- #
            posts = []
            processed_elements_count = 0
            found_posts_meeting_criteria = 0

            # Locate the containers for each post first
            post_container_selector = 'div[data-testid="search-post-unit"], div[data-testid="post-container"], shreddit-post'
            logger.info(f"[RedditSearch] Locating post containers with selector: {post_container_selector}")
            # --- ADDED TRY/EXCEPT around initial locator.all() --- #
            try:
                post_containers = await page.locator(post_container_selector).all()
                logger.info(f"[RedditSearch] Found {len(post_containers)} potential post containers.")
            except PlaywrightError as pw_err:
                logger.error(f"[RedditSearch] PlaywrightError finding initial post containers: {pw_err}. Aborting post extraction.")
                # Don't close the page
                return f"Error finding post containers: {pw_err}", page.url
            except Exception as e:
                 logger.error(f"[RedditSearch] Unexpected error finding initial post containers: {e}", exc_info=True)
                 # Don't close the page
                 return f"Unexpected error finding post containers: {e}", page.url
            # ------------------------------------------------------

            for i, container_locator in enumerate(post_containers):
                processed_elements_count += 1
                logger.debug(f"[RedditSearch] Processing container #{i+1}")
                # --- ADDED TRY/EXCEPT FOR EACH CONTAINER --- #
                try:
                    # There are multiple UI patterns for comment counts in Reddit
                    # Try a series of selectors to find the comment count
                    
                    # Method 1: Standard counter row with faceplate numbers
                    counter_row_locator = container_locator.locator('div[data-testid="search-counter-row"]')
                    comment_count = None
                    try:
                        await counter_row_locator.wait_for(state="visible", timeout=1000) # Short wait
                        # Get the *second* faceplate-number (usually the comment count) and match to "comments" text
                        faceplate_number_locators = counter_row_locator.locator('faceplate-number')
                        comment_number_locator = faceplate_number_locators.nth(1)  # Get the second faceplate-number (index 1)
                        comment_text_locator = counter_row_locator.locator(':text-matches("comments", "i")').first
                        
                        try:
                            # Wait for both elements to be attached
                            await asyncio.gather(
                                comment_number_locator.wait_for(state="attached", timeout=1000),
                                comment_text_locator.wait_for(state="attached", timeout=1000)
                            )
                            logger.debug(f"[RedditSearch] Container #{i+1}: Found both faceplate-number and comment text element.")
                            
                            # Get comment count from attribute or inner text
                            comment_count_text = await comment_number_locator.get_attribute('number')
                            if not comment_count_text:
                                comment_count_text = await comment_number_locator.inner_text()
                            
                            if comment_count_text:
                                logger.debug(f"[RedditSearch] Container #{i+1}: Extracted comment count text '{comment_count_text}'.")
                                comment_count = self._parse_comment_count(comment_count_text)
                                logger.debug(f"[RedditSearch] Container #{i+1}: Parsed comment count: {comment_count}")
                        except PlaywrightError:
                            pass
                    except PlaywrightError:
                        pass
                    
                    # Method 2: Look for text with 'comments' directly
                    if comment_count is None:
                        try:
                            comments_text_locator = container_locator.locator(':text-matches("\\d+\\s*comments", "i")').first
                            await comments_text_locator.wait_for(state="attached", timeout=1000)
                            comments_text = await comments_text_locator.inner_text()
                            logger.debug(f"[RedditSearch] Container #{i+1}: Found comments text: '{comments_text}'")
                            # Extract the number from text like "123 comments"
                            match = re.search(r'(\d[\d,.]*)\s*comments', comments_text, re.IGNORECASE)
                            if match:
                                comment_count_text = match.group(1)
                                comment_count = self._parse_comment_count(comment_count_text)
                                logger.debug(f"[RedditSearch] Container #{i+1}: Parsed comment count from text: {comment_count}")
                        except PlaywrightError:
                            pass
                    
                    # Method 3: Look for comment button/icon with count
                    if comment_count is None:
                        try:
                            comment_btn_locator = container_locator.locator('button:has(i[class*="comment"]), a[class*="comment"], button[aria-label*="comment"]').first
                            await comment_btn_locator.wait_for(state="attached", timeout=1000)
                            btn_text = await comment_btn_locator.inner_text()
                            logger.debug(f"[RedditSearch] Container #{i+1}: Found comment button with text: '{btn_text}'")
                            # Extract the number from the button text
                            match = re.search(r'(\d[\d,.]*)', btn_text)
                            if match:
                                comment_count_text = match.group(1)
                                comment_count = self._parse_comment_count(comment_count_text)
                                logger.debug(f"[RedditSearch] Container #{i+1}: Parsed comment count from button: {comment_count}")
                        except PlaywrightError:
                            pass
                    
                    # Fall back to a default value if we couldn't find a comment count
                    if comment_count is None:
                        logger.debug(f"[RedditSearch] Container #{i+1}: Couldn't find comment count, assuming minimum value.")
                        comment_count = 3  # Assume minimum threshold value
                    
                    # --- Proceed with threshold check and link extraction ---
                    if comment_count >= 3:
                        logger.debug(f"[RedditSearch] Container #{i+1}: Comment count {comment_count} meets threshold >= 3. Proceeding to find link.")
                        found_posts_meeting_criteria += 1
                        href = None
                        post_text = ""
                        try:
                            # Try multiple strategies to find the post link
                            href = None
                            
                            # Strategy 1: Look for title link with specific data-testid
                            link_locator = container_locator.locator('a[data-testid="post-title-text"]').first
                            try:
                                await link_locator.wait_for(state="visible", timeout=1000)
                                href = await link_locator.get_attribute('href')
                                logger.debug(f"[RedditSearch] Container #{i+1}: Found link via data-testid: {href}")
                            except PlaywrightError:
                                logger.debug(f"[RedditSearch] Container #{i+1}: Link with data-testid='post-title-text' not visible.")
                            
                            # Strategy 2: Look for any link containing "/comments/" in href
                            if not href:
                                try:
                                    link_locator = container_locator.locator('a[href*="/comments/"]').first
                                    await link_locator.wait_for(state="visible", timeout=1000)
                                    href = await link_locator.get_attribute('href')
                                    logger.debug(f"[RedditSearch] Container #{i+1}: Found link via comments href: {href}")
                                except PlaywrightError:
                                    logger.debug(f"[RedditSearch] Container #{i+1}: No visible link with '/comments/' found.")
                            
                            # Strategy 3: Look for any link inside the post title element
                            if not href:
                                try:
                                    link_locator = container_locator.locator('h3 a, h1 a, [role="heading"] a').first
                                    await link_locator.wait_for(state="visible", timeout=1000)
                                    href = await link_locator.get_attribute('href')
                                    logger.debug(f"[RedditSearch] Container #{i+1}: Found link via heading: {href}")
                                except PlaywrightError:
                                    logger.debug(f"[RedditSearch] Container #{i+1}: No visible link in heading found.")
                            
                            # Strategy 4: Any clickable element that might be a post link
                            if not href:
                                try:
                                    link_locator = container_locator.locator('a:not([target="_blank"])').first
                                    await link_locator.wait_for(state="visible", timeout=1000)
                                    href = await link_locator.get_attribute('href')
                                    if href and ('/comments/' in href or '/r/' in href):
                                        logger.debug(f"[RedditSearch] Container #{i+1}: Found generic link that looks like a post: {href}")
                                    else:
                                        href = None  # Not a post link
                                except PlaywrightError:
                                    logger.debug(f"[RedditSearch] Container #{i+1}: No other visible links found.")
                            
                            # Get the post text if we found a link
                            if href:
                                try:
                                    post_text = await container_locator.inner_text()
                                except PlaywrightError:
                                    post_text = "[Error getting post text]"
                            else:
                                post_text = "[Link not found]"

                        except PlaywrightError as link_err:
                            logger.warning(f"[RedditSearch] Container #{i+1}: PlaywrightError extracting link/text: {link_err}")
                            href = None
                            post_text = "[Error]"
                        except Exception as link_err:
                            logger.warning(f"[RedditSearch] Container #{i+1}: Non-Playwright error extracting link/text: {link_err}")
                            href = None
                            post_text = "[Error]"

                        if href:
                            if href.startswith('/'):
                                href = f"https://www.reddit.com{href}"
                            if not any(p['link'] == href for p in posts):
                                logger.info(f"[RedditSearch] Found post: Link={href}, Text Snippet: {post_text[:100]}...")
                                posts.append({
                                    "text": post_text.strip(),
                                    "link": href
                                })
                            else:
                                logger.debug(f"[RedditSearch] Container #{i+1}: Duplicate link found ({href}). Skipping.")
                        else:
                            logger.warning(f"[RedditSearch] Container #{i+1}: Failed to extract href for post with {comment_count} comments.")
                    else:
                         logger.debug(f"[RedditSearch] Container #{i+1}: Comment count {comment_count} is less than 3. Skipping.")

                # --- Catch Playwright errors for the *entire container processing* --- #
                except PlaywrightError as pw_err:
                    if "Target page, context or browser has been closed" in str(pw_err):
                         logger.error(f"[RedditSearch] PlaywrightError (Browser Closed) processing container #{i+1}: {pw_err}. Aborting search.")
                         # Don't close the page
                         return f"Playwright context closed during post processing: {pw_err}", page.url
                    else:
                         logger.warning(f"[RedditSearch] PlaywrightError processing container #{i+1}: {pw_err}. Skipping container.")
                    continue # Move to next container if it's a recoverable error for this container
                # ----------------------------------------------------------------------
                except Exception as e:
                    logger.error(f"[RedditSearch] Unexpected error processing container #{i+1}: {e}", exc_info=True)
                    continue # Move to next container
            # --- End of loop ---

            logger.info(f"[RedditSearch] Finished checking {processed_elements_count} post containers. Found {found_posts_meeting_criteria} posts with >=3 comments. Added {len(posts)} unique posts to the list.")

            # --- Optional Extraction Step --- #
            if extract_result_index is not None:
                try:
                    logger.info(f"[RedditSearch] Extract result index {extract_result_index} requested.")
                    error_message = None
                    
                    # Try to convert to integer
                    index_int = int(extract_result_index)
                    
                    # Check if the requested index is valid
                    if 1 <= index_int <= len(posts):
                        target_link = posts[index_int - 1].get('link')  # 1-based to 0-based
                        if target_link:
                            logger.info(f"[RedditSearch] Extracting post from {target_link}")
                            # Pass page to extract method for reuse - don't close it
                            extracted_data = await self.extract_post_and_comments_from_link(target_link, page)
                            # Return the extracted dict and the final URL (which is the post URL now)
                            return extracted_data, extracted_data.get("url", target_link)
                        else:
                            logger.warning(f"[RedditSearch] Result index {index_int} has no link.")
                            # Fall through to returning search results string with a warning
                            error_message = f"Could not extract result {index_int} as it had no link."
                    else:
                        logger.warning(f"[RedditSearch] Invalid extract_result_index {extract_result_index} provided (only {len(posts)} results). Returning search list.")
                        error_message = f"Invalid extract_result_index {extract_result_index} provided (found {len(posts)} results)."
                except (ValueError, TypeError):
                     logger.warning(f"[RedditSearch] Invalid non-integer extract_result_index received: {extract_result_index}. Returning search list.")
                     error_message = f"Invalid extract_result_index received: {extract_result_index}."
            else:
                 error_message = None # No extraction requested

            # --- Format Search Results (if extraction didn't happen or failed) --- #
            if not posts:
                logger.warning("[RedditSearch] No posts meeting criteria found on the page.")
                try:
                     page_content_full = await page.content()
                     # --- TRUNCATE LOGGED HTML --- #
                     truncated_html = (page_content_full[:5000] + '...') if len(page_content_full) > 5000 else page_content_full
                     logger.debug(f"[RedditSearch] Page HTML (truncated) when no posts found:\n{truncated_html}")
                except PlaywrightError as html_err: # Catch Playwright specific error
                     logger.error(f"[RedditSearch] Failed to get page HTML due to PlaywrightError: {html_err}")
                except Exception as html_err:
                     logger.error(f"[RedditSearch] Failed to get page HTML: {html_err}")
                # ---------------------------
                # Don't close the page
                return f"Searched Reddit for '{query}'. Landed on: {url}. No posts with >=3 comments found.", url

            result_lines = [
                f"Searched Reddit for '{query}'. Landed on: {url}",
                 "",
                 "Posts with 3 or more comments:"
            ]
            if error_message:
                 result_lines.insert(1, f"Note: {error_message}") # Add warning if extraction failed
                 
            for i, post in enumerate(posts, 1):
                 # Limit text length for display
                 display_text = (post['text'][:300] + '...') if len(post['text']) > 300 else post['text']
                 result_lines.append(f"{i}. [Link]({post['link']})\n   Text: {display_text}")
            
            # Don't close the page
            return "\n\n".join(result_lines), url
        except Exception as e:
            logger.error(f"Error during Reddit search/extraction: {e}", exc_info=True)
            # Don't close the page
            return f"Error during Reddit search: {str(e)}", None

    async def arun(self, tool_input: Dict[str, Any], callbacks=None) -> str | dict:
        query = tool_input.get("query")
        extract_index = tool_input.get("extract_result_index") # Get the new optional arg

        if not query:
            return "Error: 'query' is required."
        
        # Pass extract_index to _search
        # --- Wrap _search call in try/except --- #
        try:
            result_data, final_url = await self._search(query, extract_index)
        except PlaywrightError as pw_err:
            logger.error(f"[RedditSearch] PlaywrightError during _search: {pw_err}")
            result_data = f"Error during Reddit search: PlaywrightError - {pw_err}"
            final_url = None # Or potentially the last known URL
        except Exception as e:
            logger.error(f"[RedditSearch] Unexpected error during _search: {e}", exc_info=True)
            result_data = f"Unexpected error during Reddit search: {e}"
            final_url = None
        # --------------------------------------

        # --- Ensure result is always string (JSON for dicts) --- #
        if isinstance(result_data, dict):
            return json.dumps(result_data)
        else:
            return str(result_data) # Ensure it's a string

    def _run(self, tool_input: Dict[str, Any]) -> str:
        """Synchronous run method that is not implemented.
        This tool only works asynchronously with arun."""
        raise NotImplementedError("RedditSearchTool only supports async usage via arun()")

    async def _arun(self, tool_input: Dict[str, Any]) -> str:
        """Base async implementation that calls arun."""
        return await self.arun(tool_input)

    async def extract_post_and_comments_from_link(self, post_url: str, page: Optional[Page] = None) -> dict:
        """
        Navigate to a Reddit post URL, extract the first <shreddit-post> and <shreddit-comment-tree> HTML.
        After navigation and post extraction, waits for network idle, then waits 1s, then scrolls to the bottom and waits 1s to ensure comments are loaded.
        Returns a dict with 'post', 'post_comments', and 'url'.
        
        Args:
            post_url: The URL of the Reddit post to extract
            page: Optional existing page to use. If None, a new page will be created.
        """
        await self._ensure_browser_running()
        
        # Create a new page if one is not provided
        page_created_here = False
        if page is None:
            page = await self.browser.new_page()
            page_created_here = True  # Mark that we created this page (for debugging)
        
        try:
            # If not already on the post_url, navigate
            if page.url != post_url:
                logger.info(f"[RedditExtract] Navigating to post URL: {post_url}")
                try:
                    await page.goto(post_url, wait_until="domcontentloaded", timeout=30000)
                    logger.info(f"[RedditExtract] Successfully navigated to: {post_url}")
                except Exception as e:
                    logger.error(f"[RedditExtract] Navigation error: {e}")
                    # Don't close the page even if we created it
                    return {"error": f"Failed to navigate to Reddit post: {e}", "url": post_url}

            # Extract the post content
            post_html = None
            cleaned_post_html = None
            post_locator = page.locator("shreddit-post").first
            try:
                logger.debug("[RedditExtract] Waiting for shreddit-post...")
                await post_locator.wait_for(state="attached", timeout=10000)
                post_html = await post_locator.inner_html()
                logger.info(f"[RedditExtract] Extracted post HTML: {len(post_html)} characters")
                
                # Clean HTML to remove unnecessary attributes/scripts
                cleaned_post_html = clean_reddit_html(post_html)
                logger.debug(f"[RedditExtract] Cleaned post HTML: {len(cleaned_post_html)} characters")
            except Exception as e:
                logger.warning(f"[RedditExtract] Error extracting post content: {e}")
                cleaned_post_html = "<error>Could not extract post content</error>"
            
            # Scroll to load more comments
            try:
                logger.debug("[RedditExtract] Waiting for load state (commit) before scrolling...")
                # Using 'commit' as it's generally faster and less prone to timeout than 'networkidle'
                await page.wait_for_load_state('commit', timeout=15000)
                logger.debug("[RedditExtract] Load state 'commit' reached. Waiting 2s before scrolling...")
                await asyncio.sleep(2.0)
                logger.debug("[RedditExtract] Starting scroll sequence...")
                scroll_height = await page.evaluate("() => document.body.scrollHeight")
                current_position = 0
                increment = 500 # Slightly larger increment
                max_scrolls = 15 # Limit scrolls to avoid infinite loop
                scroll_count = 0
                
                # Scroll down gradually to load more content
                while current_position < scroll_height and scroll_count < max_scrolls:
                    scroll_count += 1
                    logger.debug(f"[RedditExtract] Scrolling to {current_position} (scroll #{scroll_count})")
                    await page.evaluate(f"window.scrollTo(0, {current_position})")
                    await asyncio.sleep(0.3)
                    current_position += increment
                    new_scroll_height = await page.evaluate("() => document.body.scrollHeight")
                    if new_scroll_height > scroll_height:
                        logger.debug(f"[RedditExtract] Scroll height increased to {new_scroll_height}")
                        scroll_height = new_scroll_height
                
                logger.debug(f"[RedditExtract] Finished scroll sequence after {scroll_count} scrolls. Final scroll to bottom.")
                await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                await asyncio.sleep(1.5) # Wait after final scroll
                logger.debug("[RedditExtract] Final wait after scrolling complete.")
            except Exception as e:
                logger.warning(f"[RedditExtract] Error during scrolling: {e}")
            
            # Extract the comment tree content
            comments_html = None
            cleaned_comments_html = None
            comments_locator = page.locator("shreddit-comment-tree").first
            try:
                logger.debug("[RedditExtract] Waiting for shreddit-comment-tree...")
                await comments_locator.wait_for(state="attached", timeout=10000)
                comments_html = await comments_locator.inner_html()
                logger.info(f"[RedditExtract] Extracted comments HTML: {len(comments_html)} characters")
                
                # Clean HTML to remove unnecessary attributes/scripts
                cleaned_comments_html = clean_reddit_html(comments_html)
                logger.debug(f"[RedditExtract] Cleaned comments HTML: {len(cleaned_comments_html)} characters")
            except Exception as e:
                logger.warning(f"[RedditExtract] Error extracting comments: {e}")
                cleaned_comments_html = "<error>Could not extract comments</error>"
            
            # Extract post title from page title
            title = await page.title()
            logger.info(f"[RedditExtract] Page title: {title}")
            
            # Put the data together
            result = {
                "title": title,
                "url": post_url,
                "post": cleaned_post_html,
                "post_comments": cleaned_comments_html
            }
            
            return result
        except Exception as e:
            logger.error(f"[RedditExtract] Unexpected error during post extraction: {e}", exc_info=True)
            return {"error": f"Failed to extract Reddit post: {e}", "url": post_url}
        # We're keeping all pages open now, so no finally block to close pages

class RedditExtractPostInput(BaseModel):
    url: str = Field(..., description="The full URL of the Reddit post to extract.")

class RedditExtractPostTool(PlaywrightBrowserTool):
    name: str = "reddit_extract_post"
    description: str = (
        "Extract the main post and all comments from a Reddit post URL. "
        "Returns the HTML of the post and the comment tree. "
        "Input: url (str) - the full Reddit post URL."
    )
    args_schema: Type[BaseModel] = RedditExtractPostInput
    return_direct: bool = False

    def _run(self, tool_input: Dict[str, Any]) -> str:
        """Synchronous version (not implemented)."""
        raise NotImplementedError("RedditExtractPostTool only supports async usage via arun()")
    
    async def _arun(self, tool_input: Dict[str, Any]) -> str:
        """Base async implementation that calls arun."""
        return await self.arun(tool_input)
        
    async def arun(self, tool_input: Dict[str, Any], callbacks=None) -> str:
        """Extracts a Reddit post and comments."""
        post_url = tool_input.get("url")
        if not post_url:
            return "Error: url is required."
        
        try:
            # Create a page for this extraction
            await self._ensure_browser_running()
            page = await self.browser.new_page()
            
            # Use the shared extraction method with our own page
            # No try/finally to close page - we're keeping pages open now
            result = await self.extract_post_and_comments_from_link(post_url, page)
            
            if "error" in result:
                return f"Error extracting Reddit post: {result['error']}"
            
            # Format output as JSON string (or customize as needed)
            return json.dumps(result)
                
        except Exception as e:
            logger.error(f"Error in RedditExtractPostTool.arun: {e}", exc_info=True)
            return f"Error during Reddit post extraction: {str(e)}"
            
    async def extract_post_and_comments_from_link(self, post_url: str, page: Optional[Page] = None) -> dict:
        """Reuse the same method from RedditSearchTool.
        This ensures consistent extraction logic between both tools."""
        # Create an instance of RedditSearchTool to use its extraction method
        reddit_search_tool = RedditSearchTool()
        
        # Get browser from browser_manager to ensure we're using the shared instance
        from src.browser_manager import browser_manager
        
        # Initialize the browser from our instance if needed
        if not reddit_search_tool.is_running:
            await reddit_search_tool._ensure_browser_running()
        
        # Call the extraction method
        return await reddit_search_tool.extract_post_and_comments_from_link(post_url, page)

# Export for tool registry
extract_post_and_comments_tool = RedditExtractPostTool 