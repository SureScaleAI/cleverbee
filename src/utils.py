import logging
import lxml.html
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# Placeholder for utility functions if needed later
# For example, text cleaning, specific data extraction helpers, etc.

def example_utility_function(text: str) -> str:
    """Example utility function."""
    logger.debug("Running example utility function.")
    return text.strip()

def strip_class_attributes(html: str) -> str:
    """Remove all class attributes from an HTML fragment using lxml."""
    try:
        doc = lxml.html.fromstring(html)
        for tag in doc.xpath('//*[@class]'):
            tag.attrib.pop('class', None)
        return lxml.html.tostring(doc, encoding='unicode')
    except Exception as e:
        logger.warning(f"Failed to strip class attributes: {e}")
        return html 

def clean_reddit_html(html: str) -> str:
    """
    Clean Reddit HTML by removing unwanted tags (svg, input, button, script, style, img, etc.),
    custom elements, and all attributes except a strict whitelist (href, id, datetime, title).
    Returns cleaned HTML as a string.
    """
    try:
        soup = BeautifulSoup(html, "lxml")
        # Tags to remove entirely
        remove_tags = [
            "svg", "input", "button", "script", "style", "img", "icon-load", "faceplate-tracker",
            "shreddit-async-loader", "shreddit-post-overflow-menu", "shreddit-join-button",
            "shreddit-status-icons", "rpl-tooltip", "faceplate-screen-reader-content",
            "faceplate-partial", "award-button", "shreddit-comment-share-button",
            "shreddit-overflow-menu", "shreddit-comment-action-row", "shreddit-comment-badges",
            "shreddit-comment-author-modifier-icon", "community-status-tooltip", "community-status",
            "faceplate-progress", "shreddit-comment-tree-ad", "cbau-trigger", "icon-load",
            "shreddit-post-flair", "shreddit-post-overflow-menu", "shreddit-distinguished-post-tags",
            "shreddit-async-loader", "shreddit-status-icons", "shreddit-join-button",
            "faceplate-hovercard", "faceplate-number", "faceplate-timeago", "faceplate-perfmark",
            "shreddit-comment-tree-ad", "shreddit-comment-action-row", "shreddit-overflow-menu",
            "shreddit-comment-badges", "shreddit-comment-author-modifier-icon", "community-status-tooltip",
            "community-status", "faceplate-progress", "shreddit-comment-tree-ad", "cbau-trigger",
            "icon-load", "shreddit-post-flair", "shreddit-post-overflow-menu", "shreddit-distinguished-post-tags"
        ]
        for tag in remove_tags:
            for t in soup.find_all(tag):
                t.decompose()
        # Remove all custom elements (tags with a dash in the name, except shreddit-post and shreddit-comment)
        for t in soup.find_all(lambda tag: '-' in tag.name and tag.name not in ["shreddit-post", "shreddit-comment", "shreddit-comment-tree"]):
            t.decompose()
        # Remove all attributes except a stricter whitelist
        whitelist = {"href", "id", "datetime", "title"}
        for tag in soup.find_all(True):
            attrs = dict(tag.attrs)
            for attr in attrs:
                if attr not in whitelist:
                    del tag.attrs[attr]
        # Remove empty divs/spans
        for tag in soup.find_all(["div", "span"]):
            if not tag.text.strip() and not tag.find(True):
                tag.decompose()
        # Return cleaned HTML
        return str(soup)
    except Exception as e:
        logger.warning(f"Failed to clean Reddit HTML: {e}")
        return html 

def get_tool_metadata_for_prompt(mcp_json_path: str = "mcp.json") -> str:
    """
    Load tool metadata from mcp.json and format as a markdown string for prompt injection.
    Returns a string with a table of tool name, description, minToolCalls, and maxToolCalls.
    """
    import json, os
    full_path = os.path.join(os.getcwd(), mcp_json_path)
    if not os.path.exists(full_path):
        return "No tool metadata found (mcp.json missing)."
    with open(full_path, 'r') as f:
        data = json.load(f)
    mcp_servers = data.get("mcpServers", {})
    if not mcp_servers:
        return "No tool metadata found (no mcpServers key)."
    lines = [
        "| Tool Name | Description | Min Calls | Max Calls |",
        "|-----------|-------------|-----------|-----------|"
    ]
    for name, cfg in mcp_servers.items():
        desc = cfg.get("description", "No description")
        min_calls = cfg.get("minToolCalls", "-")
        max_calls = cfg.get("maxToolCalls", "-")
        lines.append(f"| {name} | {desc} | {min_calls} | {max_calls} |")
    return "\n".join(lines) 