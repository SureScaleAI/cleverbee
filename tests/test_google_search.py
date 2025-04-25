import os
import pytest
import logging
from src.browser import PlaywrightBrowserTool
from src.content_manager import ContentManager

PlaywrightBrowserTool.model_rebuild()

pytestmark = pytest.mark.asyncio

@pytest.mark.skipif(os.environ.get("PLAYWRIGHT_SKIP_BROWSER"), reason="Playwright browser tests skipped by env var.")
async def test_google_search_real():
    logging.getLogger('src.browser').setLevel(logging.DEBUG)
    tool = PlaywrightBrowserTool()
    query = "python programming"
    print(f"[TEST] Performing Google search for: {query}")
    results = await tool._search(query, num_results=5)
    print("[DEBUG] Results:", results)

    # Assert the result is a list of dicts
    assert isinstance(results, list), f"Expected list, got {type(results)}"
    assert len(results) >= 5, f"Expected at least 5 results, got {len(results)}"
    for result in results:
        assert isinstance(result, dict), f"Each result should be a dict, got {type(result)}"
        assert 'title' in result and result['title'], "Result missing non-empty title"
        assert 'link' in result and result['link'], "Result missing non-empty link"
        assert 'snippet' in result, "Result missing snippet"
    print("[TEST] First result:", results[0])
    # Clean up browser
    if hasattr(tool, 'clean_up'):
        await tool.clean_up()

@pytest.mark.skipif(os.environ.get("PLAYWRIGHT_SKIP_BROWSER"), reason="Playwright browser tests skipped by env var.")
async def test_google_search_youtube_query():
    logging.getLogger('src.browser').setLevel(logging.DEBUG)
    tool = PlaywrightBrowserTool()
    query = '"meaning of life" site:youtube.com'
    print(f"[TEST] Performing Google search for: {query}")
    results = await tool._search(query, num_results=5)
    print("[DEBUG] Results:", results)

    # Assert the result is a list of dicts
    assert isinstance(results, list), f"Expected list, got {type(results)}"
    assert len(results) >= 5, f"Expected at least 5 results, got {len(results)}"
    for result in results:
        assert isinstance(result, dict), f"Each result should be a dict, got {type(result)}"
        assert 'title' in result and result['title'], "Result missing non-empty title"
        assert 'link' in result and result['link'], "Result missing non-empty link"
        assert 'snippet' in result, "Result missing snippet"
    print("[TEST] First result:", results[0])
    # Clean up browser
    if hasattr(tool, 'clean_up'):
        await tool.clean_up()