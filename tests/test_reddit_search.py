import os
import pytest
import asyncio
from src.tools.reddit_search_tool import RedditSearchTool
import logging
import re

pytestmark = pytest.mark.asyncio

@pytest.mark.skipif(os.environ.get("PLAYWRIGHT_SKIP_BROWSER"), reason="Playwright browser tests skipped by env var.")
async def test_reddit_search_full(tmp_path):
    # *** Force logger level for this test ***
    logging.getLogger('src.tools.reddit_search_tool').setLevel(logging.DEBUG)
    tool = RedditSearchTool()
    # Perform a single search with non-default filters
    print("[TEST] Searching Reddit with filters: Top, Past month...")
    status_string, final_url = await tool._search("python", sort="Top", date_range="Past month")
    print("[DEBUG] Result (status):", status_string)
    print("[DEBUG] Final URL:", final_url)
    
    # Assertions based on the final URL
    assert final_url is not None
    assert "reddit.com/search" in final_url
    assert "sort=top" in final_url
    assert "t=month" in final_url
    
    # Test post extraction
    assert "Posts with 3 or more comments:" in status_string
    # Check if we've found at least one post
    assert "[Link](" in status_string
    
    # Test formatting of extracted posts
    lines = status_string.split('\n\n')
    assert len(lines) >= 3  # Header, blank line, posts list starting line, at least one post
    
    # Parse out posts from the status string for more detailed testing
    post_section_start = status_string.find("Posts with 3 or more comments:")
    if post_section_start != -1:
        post_section = status_string[post_section_start:]
        # Count the number of extracted posts (each post starts with a number followed by period)
        post_count = len(re.findall(r'\d+\. \[Link\]', post_section))
        print(f"[DEBUG] Number of extracted posts: {post_count}")
        assert post_count > 0, "No posts were extracted"

    # Clean up browser
    if hasattr(tool, 'clean_up'):
        await tool.clean_up()

@pytest.mark.skipif(os.environ.get("PLAYWRIGHT_SKIP_BROWSER"), reason="Playwright browser tests skipped by env var.")
async def test_reddit_search_missing_query():
    tool = RedditSearchTool()
    # The arun method handles the query check
    result = await tool.arun({})
    assert "Error" in result
    # No browser should be launched if query is missing
    assert tool.is_running is False 

@pytest.mark.skipif(os.environ.get("PLAYWRIGHT_SKIP_BROWSER"), reason="Playwright browser tests skipped by env var.")
async def test_reddit_extract_post_and_comments():
    logging.getLogger('src.tools.reddit_search_tool').setLevel(logging.DEBUG)
    tool = RedditSearchTool()
    post_url = "https://www.reddit.com/r/videos/comments/5gafop/rick_astley_never_gonna_give_you_up_sped_up_every/"
    print(f"[TEST] Extracting post and comments from: {post_url}")
    result = await tool.extract_post_and_comments_from_link(post_url)
    print("[DEBUG] Extraction result:", result)

    assert isinstance(result, dict)
    assert result.get("url") == post_url
    assert result.get("post"), "Post HTML should not be empty"
    assert result.get("post_comments"), "Post comments HTML should not be empty"
    # Check that class attributes are stripped
    assert "class=" not in result["post"], "Class attributes should be stripped from post HTML"
    assert "class=" not in result["post_comments"], "Class attributes should be stripped from comments HTML"

    # Check for a specific known comment in the HTML
    expected_comment = "Ugh. I thought I was smarter than that."
    # Normalize whitespace in the HTML
    normalized_comments = re.sub(r"\s+", " ", result["post_comments"]).strip()
    normalized_expected = re.sub(r"\s+", " ", expected_comment).strip()
    assert normalized_expected in normalized_comments, f"Expected comment not found in post_comments HTML (whitespace-insensitive): '{expected_comment}'"

    # Clean up browser
    if hasattr(tool, 'clean_up'):
        await tool.clean_up() 