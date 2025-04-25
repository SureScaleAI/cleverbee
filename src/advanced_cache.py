"""
Advanced caching implementation for LangChain that normalizes prompts before caching.

This module provides a custom cache implementation that normalizes volatile parts of prompts
(like timestamps, UUIDs, agent scratchpads) to improve cache hit rates.
"""

import hashlib
import json
import logging
import re
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

from langchain.schema import Generation
from langchain_community.cache import SQLiteCache

logger = logging.getLogger(__name__)


@dataclass
class CacheStats:
    """Statistics for the cache."""
    hits: int = 0
    misses: int = 0
    normalizations: int = 0
    skipped: int = 0


class ContentNormalizer:
    """Normalizes content for caching purposes by removing volatile parts."""

    def __init__(self):
        # Patterns to identify and normalize volatile parts
        self.patterns = [
            # Dates and times in various formats
            (r"\b\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z\b", "[TIMESTAMP]"),
            (r"\b\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z\b", "[TIMESTAMP]"),
            (r"\b\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\b", "[TIMESTAMP]"),
            (r"current date: \d{4}-\d{2}-\d{2}\b", "current date: [DATE]"),
            (r"current_date\s*[:=]\s*[\'\"]?\d{4}-\d{2}-\d{2}[\'\"]?", "current_date: [DATE]"),
            
            # UUIDs
            (r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", "[UUID]"),
            
            # Conversation history markers (more specific first)
            (r"AI: .*?(?=\nHuman:|$)", "AI: [AI_RESPONSE]"),
            (r"Human: .*?(?=\nAI:|$)", "Human: [HUMAN_MESSAGE]"),
            
            # Tool executions and scratchpad
            (r"Action: .*?(?=\nObservation:|$)", "Action: [ACTION]"),
            (r"Observation: .*?(?=\nAction:|Thought:|$)", "Observation: [OBSERVATION]"),
            
            # Agent scratchpad: Refactored pattern to avoid variable-width look-behind
            (r'(agent_scratchpad\s*[:=]\s*[\"\']{0,1})[^\"\'\n]+?', r'\1[SCRATCHPAD]'),
            
            # Research specific patterns
            (r"Research History \(Newest first\):.*?(?=Condensed Research Content|$)", "Research History: [HISTORY]"),
            (r"Condensed Research Content.*?(?=Task:|$)", "Condensed Research Content: [CONTENT]"),
        ]
        
        # Compile patterns for efficiency
        self.compiled_patterns = [(re.compile(pattern, re.DOTALL | re.MULTILINE), replacement) 
                                 for pattern, replacement in self.patterns]
        
        # Sections to completely remove (identify by markers)
        self.sections_to_remove = [
            (r"--- BEGIN CONDENSED CONTENT ---.*?--- END CONDENSED CONTENT ---", "[CONDENSED_CONTENT]"),
            (r"Condensed Research Content \(Summaries & Key Info\):.*?Task:", "Condensed Research Content: [CONTENT]\n\nTask:"),
        ]
        self.compiled_sections = [(re.compile(pattern, re.DOTALL), replacement) 
                                 for pattern, replacement in self.sections_to_remove]

    def normalize_prompt(self, prompt: str) -> str:
        """
        Normalize a prompt by replacing volatile parts with constants.
        
        Args:
            prompt: The prompt string to normalize
            
        Returns:
            The normalized prompt string
        """
        if not prompt:
            return prompt
            
        # Apply section removals first (these are larger chunks)
        result = prompt
        for pattern, replacement in self.compiled_sections:
            result = pattern.sub(replacement, result)
            
        # Then apply pattern replacements
        for pattern, replacement in self.compiled_patterns:
            result = pattern.sub(replacement, result)
            
        return result


class NormalizingCache(SQLiteCache):
    """Cache implementation that normalizes prompts for better cache hit rates."""
    
    def __init__(self, database_path: str = ".langchain.db"):
        # Removed cache_schema from super init call due to TypeError
        super().__init__(database_path=database_path) 
        self.normalizer = ContentNormalizer()
        self.stats = CacheStats()
        self._original_lookup = super().lookup
        self._original_update = super().update

    def should_normalize(self, prompt: str) -> bool:
        """Determine if this prompt should be normalized."""
        # Skip normalization for very short prompts (likely not agent prompts)
        if len(prompt) < 100:
            return False
            
        # Skip normalization if it doesn't contain key terms indicating agent prompts
        agent_indicators = [
            "scratchpad", "agent_", "history", "memory", "conversation",
            "research", "task", "plan", "tools", "iterate", "execute"
        ]
        return any(indicator in prompt.lower() for indicator in agent_indicators)
    
    def _prompt_hash_and_preview(self, prompt: str) -> str:
        h = hashlib.sha256(prompt.encode()).hexdigest()[:8]
        preview = prompt[:80].replace('\n', ' ')
        return f"hash={h}, preview=\"{preview}\""

    def lookup(self, prompt: str, llm_string: str) -> Optional[List[Generation]]:
        """
        Look up an LLM prompt in the cache with normalization.
        
        Args:
            prompt: The prompt to look up
            llm_string: The LLM identifier
            
        Returns:
            A list of generations if found, None otherwise
        """
        log_id = self._prompt_hash_and_preview(prompt)
        logger.info(f"[CACHE] Lookup: model={llm_string}, {log_id}")
        original_result = self._original_lookup(prompt, llm_string)
        if original_result is not None:
            self.stats.hits += 1
            logger.info(f"[CACHE] HIT: model={llm_string}, {log_id} (original)")
            return original_result
        if self.should_normalize(prompt):
            try:
                normalized_prompt = self.normalizer.normalize_prompt(prompt)
                self.stats.normalizations += 1
                logger.info(f"[CACHE] NORMALIZATION attempted: {log_id}, original_len={len(prompt)}, normalized_len={len(normalized_prompt)}")
                if abs(len(normalized_prompt) - len(prompt)) < 20:
                    self.stats.skipped += 1
                    self.stats.misses += 1
                    logger.info(f"[CACHE] NORMALIZATION skipped: {log_id}, not enough change")
                    return None
                normalized_log_id = self._prompt_hash_and_preview(normalized_prompt)
                normalized_result = self._original_lookup(normalized_prompt, llm_string)
                if normalized_result is not None:
                    self.stats.hits += 1
                    logger.info(f"[CACHE] HIT: model={llm_string}, {normalized_log_id} (normalized)")
                    return normalized_result
            except Exception as e:
                logger.warning(f"[CACHE] Error during prompt normalization: {e}")
        self.stats.misses += 1
        logger.info(f"[CACHE] MISS: model={llm_string}, {log_id}")
        return None
        
    def update(self, prompt: str, llm_string: str, return_val: List[Generation]) -> None:
        """
        Update the cache with a new prompt and result.
        Store both the original and normalized versions to maximize future hits.
        
        Args:
            prompt: The prompt to store
            llm_string: The LLM identifier
            return_val: The generations to store
        """
        log_id = self._prompt_hash_and_preview(prompt)
        logger.info(f"[CACHE] UPDATE: model={llm_string}, {log_id}")
        self._original_update(prompt, llm_string, return_val)
        if self.should_normalize(prompt):
            try:
                normalized_prompt = self.normalizer.normalize_prompt(prompt)
                if abs(len(normalized_prompt) - len(prompt)) >= 20:
                    normalized_log_id = self._prompt_hash_and_preview(normalized_prompt)
                    self._original_update(normalized_prompt, llm_string, return_val)
                    logger.info(f"[CACHE] UPDATE: model={llm_string}, {normalized_log_id} (normalized)")
            except Exception as e:
                logger.warning(f"[CACHE] Error updating cache with normalized prompt: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        hit_rate = 0
        if (self.stats.hits + self.stats.misses) > 0:
            hit_rate = self.stats.hits / (self.stats.hits + self.stats.misses)
            
        return {
            "hits": self.stats.hits,
            "misses": self.stats.misses,
            "normalizations": self.stats.normalizations,
            "skipped": self.stats.skipped,
            "hit_rate": hit_rate
        }
    
    def print_stats(self) -> str:
        """Get a formatted string with cache statistics."""
        stats = self.get_stats()
        return (f"Cache Stats: {stats['hits']} hits, {stats['misses']} misses, "
                f"Hit Rate: {stats['hit_rate']:.2%}, "
                f"Normalizations: {stats['normalizations']}, "
                f"Skipped: {stats['skipped']}")


def initialize_normalizing_cache(db_path: str = ".langchain.db") -> NormalizingCache:
    """
    Initialize and return a normalizing cache instance.
    
    Args:
        db_path: Path to the SQLite database file
        
    Returns:
        A NormalizingCache instance
    """
    cache = NormalizingCache(database_path=db_path)
    return cache 