import logging
import asyncio
from typing import Dict, Any, List, Optional, Tuple, Set
import hashlib
import re
import uuid
from datetime import datetime

# LangChain components
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain_core.language_models.chat_models import BaseChatModel # Updated import
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.callbacks import BaseCallbackHandler # For passing callbacks
from langchain_core.prompts import PromptTemplate # Needed for custom prompt

# Project specific imports
from config.settings import (
    USE_PROGRESSIVE_LOADING,
    MAX_CONTENT_PREVIEW_TOKENS,
    CHUNK_SIZE, # Add config setting
    CHUNK_OVERLAP, # Add config setting
    USE_LOCAL_SUMMARIZER_MODEL, # For checking if local models are enabled
    SUMMARIZER_MODEL,  # New unified summarizer model setting
    SUMMARY_MAX_TOKENS,
    LOCAL_MODELS_DIR # To check for model file existence
)
from config.prompts import CONDENSE_PROMPT, COMBINE_PROMPT # <<< IMPORT BOTH PROMPTS

# Import the factory function for creating LLM clients
from src.llm_clients.factory import get_llm_client

# --- Import Tiktoken helper --- 
from src.browser import get_token_count_for_text
# Replace old estimate with tiktoken
_estimate_token_count = get_token_count_for_text
# --------------------------

# Add universal content extraction helper for MCP tools
from typing import Any, Optional

def extract_mcp_content_universal(tool_output: Any) -> Optional[str]:
    """Extract text from MCP tool output universally."""
    if tool_output is None:
        return None
    if isinstance(tool_output, str):
        return tool_output
    # Attempt JSON serialization
    try:
        import json
        return json.dumps(tool_output)
    except Exception:
        try:
            return str(tool_output)
        except Exception:
            return None

logger = logging.getLogger(__name__)

class ContentItem:
    """Represents a single content item with source tracking metadata."""
    
    def __init__(self, content: str, source_url: str, source_type: str, title: str = None, metadata: Dict[str, Any] = None):
        """Initialize a ContentItem.
        
        Args:
            content: The actual content text
            source_url: Complete URL where the content was sourced from
            source_type: Type of source (e.g., "web", "reddit", "pubmed")
            title: Optional title for the content 
            metadata: Additional metadata about the content
        """
        self.content = content
        self.source_url = source_url
        self.source_type = source_type
        self.title = title or "Unknown Title"
        self.metadata = metadata or {}
        self.timestamp = datetime.now()
        self.content_id = self._generate_content_id(source_url)
        self.documents = []  # Will hold chunked documents
        
    def _generate_content_id(self, url: str) -> str:
        """Generate a short hash identifier for a URL."""
        return hashlib.md5(url.encode()).hexdigest()[:8]
    
    def create_documents(self, splitter, chunking_enabled: bool = True) -> List[Document]:
        """Create LangChain documents from this content item.
        
        Args:
            splitter: RecursiveCharacterTextSplitter instance to use
            chunking_enabled: Whether to chunk the content
            
        Returns:
            List of Document objects
        """
        metadata = {
            "source": self.source_url,
            "title": self.title,
            "content_id": self.content_id,
            "source_type": self.source_type,
            "estimated_tokens": _estimate_token_count(self.content)
        }
        
        # Add any additional metadata
        metadata.update(self.metadata)
        
        try:
            if chunking_enabled:
                self.documents = splitter.create_documents([self.content], metadatas=[metadata])
            else:
                self.documents = [Document(page_content=self.content, metadata=metadata)]
                
            return self.documents
        except Exception as e:
            logger.error(f"Error processing document for {self.source_url}: {e}", exc_info=True)
            self.documents = [Document(page_content=f"[Error processing content: {e}]", metadata=metadata)]
            return self.documents

class ContentManager:
    """Manages web content using LangChain Documents, TextSplitters, and summarization chains.

    This class provides content management capabilities:
    1. Store full content as chunked LangChain Documents.
    2. Generate and cache summaries using LangChain summarization chains.
    3. Progressive loading (preview based on summary).
    4. Retrieval of content chunks.
    """

    def __init__(
        self,
        primary_llm: BaseChatModel, # Updated type hint
        summarization_llm: Optional[BaseChatModel] = None, # Optional: Pre-loaded fallback
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP
    ):
        """Initialize the ContentManager.

        Args:
            primary_llm: The main LangChain BaseChatModel instance.
            summarization_llm: Optional pre-initialized lightweight LangChain BaseChatModel instance (e.g., Gemini Flash for fallback).
            chunk_size: The target size for text chunks.
            chunk_overlap: The overlap between text chunks.
        """
        self.primary_llm = primary_llm
        self.summarization_llm = summarization_llm
        self.use_chunking = chunk_size != 0
        if not self.use_chunking:
             logger.info("Content chunking is disabled (CHUNK_SIZE=0)")
             chunk_size = 100000 # Use large default if chunking disabled

        # Storage
        self.documents: Dict[str, List[Document]] = {}
        self.summaries: Dict[str, str] = {}
        self.content_hash_map: Dict[str, str] = {}
        
        # New content tracking with improved source attribution
        self.content_items: Dict[str, ContentItem] = {}

        # Splitter
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
        logger.info(f"Initialized RecursiveCharacterTextSplitter with chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")

        # Settings
        self.use_progressive_loading = USE_PROGRESSIVE_LOADING
        self.max_preview_tokens = MAX_CONTENT_PREVIEW_TOKENS
        self.summary_max_tokens = SUMMARY_MAX_TOKENS
        self.use_local_summarizer_model = USE_LOCAL_SUMMARIZER_MODEL
        self.local_models_dir = LOCAL_MODELS_DIR
        self.summarizer_model = SUMMARIZER_MODEL
        
        # --- BEGIN Define Detailed Summary Prompts ---
        self.detailed_map_prompt_template = PromptTemplate(
            input_variables=["text"],
            template="""
            You are an assistant tasked with summarizing chunks of text from a larger document (like a webpage or discussion thread). Focus on extracting key information objectively.

            **Instructions:**
            1.  **Identify Key Points:** Extract the main arguments, facts, opinions, or distinct themes presented in this specific chunk.
            2.  **List Details:** Use bullet points to list these key points concisely.
            3.  **Preserve Specifics:** Keep names, numbers, dates, and **especially any URLs** mentioned in this chunk.
            4.  **Direct Extraction:** Primarily extract information directly present in the text. Avoid deep interpretation or adding outside knowledge.
            5.  **Context:** Remember this is just one part of a larger document.

            Text Chunk:
            "{text}"

            CONCISE SUMMARY POINTS (with specific details and URLs preserved):
            """
        )

        self.detailed_combine_prompt_template = PromptTemplate(
            input_variables=["text"],
            template="""
            You are an assistant synthesizing key points extracted from different chunks of a document (e.g., a webpage or discussion thread) about a specific topic. Your goal is to create a single, structured, and detailed summary.

            **Instructions:**
            1.  **Synthesize:** Combine the provided summaries from different text chunks.
            2.  **Structure:** Organize the final output logically. Use markdown headings (`###`) for distinct themes or sections if identifiable. Use bullet points (`*` or `-`) for individual points, examples, or arguments within those themes.
            3.  **Detail:** Include specific details, examples, names, numbers, and viewpoints mentioned in the summaries. Do *not* over-simplify.
            4.  **Objectivity:** Present the information factually based on the provided text points.
            5.  **URL Preservation (CRITICAL):** Ensure **ALL URLs** present in the chunk summaries are included in the final output, typically listed at the end under a "### Source Links" heading.
            6.  **Completeness:** Cover the main topics discussed across the summaries.
            7.  **Link Formatting (CRITICAL):** When creating the "### Source Links" section, ensure all relative Reddit links (those starting with `/r/`, `/u/`, or `/comments/`) are converted to full URLs by prepending `https://www.reddit.com`.

            Summaries from Document Chunks:
            "{text}"

            DETAILED STRUCTURED SUMMARY (using markdown headings and bullet points, preserving all details and URLs):
            """
        )
        # --- END Define Detailed Summary Prompts ---
        
        # Local LLM instance and chain (initialized lazily or on demand)
        self.local_llm: Optional[BaseChatModel] = None
        self.summary_chain_local_mapreduce: Optional[Any] = None # Store map_reduce chain

        # --- Load standard summarization chains --- 
        # Primary chain (stuff)
        self.summary_chain_primary = load_summarize_chain(self.primary_llm, chain_type="stuff") 
        
        # Pre-loaded fallback chain (stuff, if provided)
        self.summary_chain_fallback = None
        if self.summarization_llm:
            # Use CONDENSE_PROMPT for fallback summarizer if desired, or standard prompt
            self.summary_chain_fallback = load_summarize_chain(self.summarization_llm, chain_type="stuff")
            logger.info(f"Initialized fallback summarization chain (stuff). Model: {getattr(self.summarization_llm, 'model', 'N/A')}")
        else:
             logger.info(f"No pre-initialized fallback model provided.")

        logger.info(
            f"ContentManager initialized with summarizer: {self.summarizer_model}, "
            f"max_tokens: {self.summary_max_tokens}, chunk_size: {self.use_chunking}, "
            f"chunk_overlap: {self.use_chunking}, "
            f"use_local_summarizer_model: {self.use_local_summarizer_model}, "
            f"progressive_loading: {self.use_progressive_loading}"
        )

        # Check if local model file exists if we are supposed to use it
        if self.use_local_summarizer_model:
            model_path = self.local_models_dir / self.summarizer_model
            if not model_path.exists():
                error_msg = f"Local summarizer model file not found: {model_path}"
                logger.error(error_msg)
                raise RuntimeError(error_msg + ". Local summarizer model is required (USE_LOCAL_SUMMARIZER_MODEL=True). Exiting.")
            else:
                logger.info(f"Verified local summarizer model exists: {model_path}")

    def _generate_content_id(self, url: str) -> str:
        """Generate a short hash identifier for a URL."""
        return hashlib.md5(url.encode()).hexdigest()[:8]

    def _get_model_context_window(self, llm: BaseChatModel) -> int:
        """Get the context window size for the given model.
        
        Args:
            llm: The language model
            
        Returns:
            The context window size in tokens (estimated)
        """
        # Try to get context window from model attributes if available
        if hasattr(llm, "get_num_tokens_from_messages"):
            try:
                # Create a proper empty message instead of a string
                from langchain_core.messages import HumanMessage
                return getattr(llm, "get_num_tokens_from_messages", lambda x: 4096)([HumanMessage(content="")])[0]
            except (ValueError, TypeError, IndexError, AttributeError) as e:
                # Fall back to direct method if possible
                if hasattr(llm, "get_num_tokens"):
                    try:
                        return llm.get_num_tokens("") + 4000  # Add buffer for message formatting
                    except Exception:
                        pass
                # Log the issue but continue with fallback logic
                logger.warning(f"Error getting context window from model: {e}. Using fallback estimation.")
        
        # For Gemini Flash models (handle any version)
        if hasattr(llm, "model") and "flash" in getattr(llm, "model", "").lower():
            # Gemini Flash has a large context window, but let's be slightly conservative
            return 1000000 # Example: 1M tokens
            
        # For Claude models
        if hasattr(llm, "model_name"):
            model_name = getattr(llm, "model_name", "").lower()
            if "claude-3" in model_name and "sonnet" in model_name:
                return 200000  # ~200k tokens
            if "claude-3" in model_name and "opus" in model_name:
                return 200000  # ~200k tokens
            if "claude-3" in model_name and "haiku" in model_name:
                return 150000  # ~150k tokens
                
        # For local models - best effort estimates
        if self.use_local_summarizer_model:
            if "llama" in self.summarizer_model.lower():
                return 4096  # Conservative estimate for Llama models
            if "mistral" in self.summarizer_model.lower():
                return 8192  # Mistral models
            
        # Default fallback: assume a conservative context window
        return 4096  # Conservative default

    def _estimate_document_size(self, docs: List[Document]) -> int:
        """Estimate the token size of a list of documents.
        
        Args:
            docs: List of LangChain Document objects
            
        Returns:
            Estimated token count
        """
        # Sum up estimated tokens from each document
        total_tokens = sum(_estimate_token_count(doc.page_content) for doc in docs)
        return total_tokens
        
    def _select_and_load_local_model(self, content: str) -> Tuple[Optional[BaseChatModel], Optional[str]]:
        """Select and load a local model for summarization.
        
        Args:
            content: The content to summarize, used for size estimation.
            
        Returns:
            Tuple of (model, error_message)
        """
        try:
            if self.local_llm is None:
                # Initialize local LLM on first use
                logger.info(f"Loading local model: {self.summarizer_model}")
                self.local_llm = get_llm_client(self.summarizer_model, local=True, model_kwargs={})
                
            return self.local_llm, None
        except Exception as e:
            error_msg = f"Failed to load local model: {e}"
            logger.error(error_msg)
            return None, error_msg

    def store_content(self, url: str, content_data: Dict[str, Any], source_type: str = "web") -> str:
        """Store content from a URL as a ContentItem with proper source tracking.

        Args:
            url: The source URL
            content_data: Dictionary containing content data (title, full_content)
            source_type: Type of source (e.g., "web", "reddit", "pubmed")

        Returns:
            content_id: A unique identifier for the content
        """
        # Extract content from content_data
        full_content = content_data.get("full_content", "")
        title = content_data.get("title", "Unknown Title")
        
        # Create additional metadata from any other fields in content_data
        metadata = {k: v for k, v in content_data.items() if k not in ["full_content", "title"]}
        
        # Create a ContentItem for better source tracking
        content_item = ContentItem(
            content=full_content,
            source_url=url,
            source_type=source_type,
            title=title,
            metadata=metadata
        )
        
        # Generate a content ID and store mappings
        content_id = content_item.content_id
        self.content_hash_map[content_id] = url
        
        # Store the ContentItem
        self.content_items[url] = content_item
        
        # Create documents using the ContentItem's method and store in traditional storage
        if not full_content:
            logger.warning(f"No content provided for URL: {url}. Storing empty document list.")
            self.documents[url] = []
            content_item.documents = []
        else:
            # Create and store documents
            docs = content_item.create_documents(self.splitter, self.use_chunking)
            self.documents[url] = docs
            
            logger.info(f"Stored content from {url} as {len(docs)} documents with source type '{source_type}' (ID: {content_id})")

        return content_id
        
    def generate_sources_section(self) -> str:
        """Generate a properly formatted sources section for the final output.
        
        Returns:
            Formatted sources section string with all used sources
        """
        sources = []
        
        # Include all content items
        for url in self.content_items:
            item = self.content_items[url]
            sources.append(f"{item.source_url}")
                
        if sources:
            return "<sources>\n" + "\n".join(sources) + "\n</sources>"
        else:
            return "<sources>\nNo sources were used.\n</sources>"

    async def get_summary(self, url_or_id: str, callbacks: Optional[List[BaseCallbackHandler]] = None, content: Optional[str] = None) -> str:
        """Get a summary for a URL/ID or directly provided content.
        
        Args:
            url_or_id: URL or content ID
            callbacks: Optional callbacks for the LLM call
            content: Optional raw content string to summarize directly (bypasses storage)

        Returns:
            Summary string or error message
        """
        url = self.content_hash_map.get(url_or_id, url_or_id)
        docs = []
        metadata = {"source": url, "content_id": self._generate_content_id(url)}
        
        # --- Added: Check for direct content being an error --- 
        if content is not None and isinstance(content, str) and content.strip().startswith("Error:"):
            logger.warning(f"Skipping summarization for {url} because provided content is an error message: {content}")
            # Optionally store the error as the summary?
            # self.summaries[url] = content 
            return content # Return the error message directly
        # --- End Added ---
        
        # --- Get Documents --- 
        if content is not None:
            # If content is provided directly, create Document object(s)
            if self.use_chunking:
                docs = self.splitter.create_documents([content], metadatas=[metadata])
                logger.info(f"Created {len(docs)} document chunks from directly provided content")
            else:
                docs = [Document(page_content=content, metadata=metadata)]
                logger.info(f"Created single document from directly provided content (chunking disabled)")
                
            # Store this content temporarily if it's not already stored
            if url not in self.content_items:
                temp_content_item = ContentItem(
                    content=content,
                    source_url=url,
                    source_type="direct",
                    title="Direct Content",
                    metadata=metadata
                )
                self.content_items[url] = temp_content_item
        else:
            if url in self.summaries:
                return self.summaries[url]
                
            if url not in self.documents or not self.documents[url]:
                logger.warning(f"No content/documents found for URL/ID: {url_or_id}")
                return f"[No content available for {url_or_id}]"
                
            docs = self.documents[url]
        # --- End Get Documents --- 

        if not docs:
             return f"[No processable documents found for {url_or_id}]" # Should not happen if checks above are correct

        # --- Analyze document size vs. context window --- 
        # Select model and determine if we need map_reduce based on content size vs. context window
        model_to_use = None
        use_map_reduce = False
        model_desc = "Unknown"
        chain_to_use = None
        final_summary = f"[Summary generation failed]" # Default error
        
        # Calculate document size
        doc_token_count = self._estimate_document_size(docs)
        logger.info(f"Preparing to summarize {len(docs)} documents, estimated total tokens: {doc_token_count} for {url}")
        
        # Determine which base model to use
        if self.use_local_summarizer_model:
            # Try to use the configured local model
            model_to_use, error_msg = self._select_and_load_local_model(" ".join([doc.page_content for doc in docs]))
            if model_to_use:
                model_desc = f"Local Model ({self.summarizer_model})"
            else:
                # Fallback logic if local loading fails (if strict check was disabled)
                if self.summarization_llm:
                    model_to_use = self.summarization_llm
                    model_desc = f"Cloud Fallback ({getattr(model_to_use, 'model', 'unknown')})"
                else:
                    model_to_use = self.primary_llm
                    model_desc = "Primary LLM"
        elif not self.use_local_summarizer_model and self.summarization_llm:
            # Use pre-loaded Cloud (Gemini) summarizer if configured
            model_to_use = self.summarization_llm
            model_desc = f"Cloud Model ({self.summarizer_model})"
        else:
             # Should not happen if agent initialization is correct
             logger.warning("Could not determine summarization model in ContentManager. Falling back to primary LLM.")
             model_to_use = self.primary_llm
             model_desc = "Primary LLM (Fallback)"
             # Potentially raise an error here instead of falling back?
             # raise RuntimeError("Summarization model could not be determined.")
        
        # Get context window for the selected model
        model_context_window = self._get_model_context_window(model_to_use)
        logger.info(f"Model {model_desc} has estimated context window of {model_context_window} tokens")
        
        # Decide on summarization strategy based on document size vs. context window
        # Use 80% of context window as threshold to account for prompt tokens and overhead
        threshold = int(model_context_window * 0.8)
        if doc_token_count > threshold:
            use_map_reduce = True
            logger.info(f"Document size ({doc_token_count} tokens) exceeds {threshold} tokens threshold for {model_desc}, using map_reduce strategy")
        else:
            use_map_reduce = False
            logger.info(f"Document size ({doc_token_count} tokens) fits within {threshold} tokens threshold for {model_desc}, using stuff strategy")
        
        # --- Create chain based on strategy --- 
        try:
            if use_map_reduce:
                # Use map_reduce with custom prompts
                chain_to_use = load_summarize_chain(
                    model_to_use,
                    chain_type="map_reduce",
                    map_prompt=CONDENSE_PROMPT,
                    combine_prompt=COMBINE_PROMPT
                )
                model_desc += " [map_reduce with custom prompts]"
                logger.info(f"Created map_reduce chain with custom prompts for {model_desc}")
            else:
                # Use stuff strategy
                chain_to_use = load_summarize_chain(model_to_use, chain_type="stuff")
                model_desc += " [stuff]"
                logger.info(f"Using stuff chain for {model_desc}")
        except Exception as e:
            logger.error(f"Error creating summarization chain for {model_desc}: {e}", exc_info=True)
            final_summary = f"[Summary failed: Error creating chain for {model_desc}: {e}]"
            # Fall back to primary chain if available
            if not use_map_reduce and model_to_use != self.primary_llm:
                try:
                    chain_to_use = self.summary_chain_primary
                    model_desc = "Primary LLM [stuff fallback]"
                    logger.info(f"Falling back to primary LLM stuff chain after chain creation error")
                except Exception:
                    logger.error(f"Also failed to use primary fallback chain", exc_info=True)
        
        # --- Run Summarization --- 
        if chain_to_use:
            logger.info(
                f"Attempting summarization for '{url}': "
                f"Model='{model_desc}', Strategy={'map_reduce' if use_map_reduce else 'stuff'}, "
                f"Doc Tokens={doc_token_count}, Context Threshold={threshold}"
            )
            try:
                logger.info(f"==> INVOKING chain {model_desc} for {url}") # Log before invoke
                # Run the selected chain
                result = await chain_to_use.ainvoke(
                    {"input_documents": docs},
                    {"callbacks": callbacks}
                )
                logger.info(f"<== COMPLETED chain invocation for {url}") # Log after invoke
                # --- FIX: Handle both dict and str chain outputs ---
                if isinstance(result, dict):
                    final_summary = result.get("output_text", "[Summary generation failed: No output text]")
                elif isinstance(result, str):
                    final_summary = result # Use the string output directly
                else:
                    # Handle unexpected result types
                    logger.warning(f"Unexpected result type from summarization chain: {type(result)}. Using fallback error message.")
                    final_summary = "[Summary generation failed: Unexpected output type]"
                # --- END FIX ---
                logger.info(f"Generated summary using {model_desc} for {url} - {len(final_summary)} chars")
                
            except Exception as e:
                logger.error(f"Error generating summary with {model_desc}: {e}", exc_info=True)
                
                # If we have a fallback chain and aren't already using it
                if chain_to_use is not self.summary_chain_fallback and self.summary_chain_fallback:
                    logger.info(f"Trying fallback chain after error with {model_desc}")
                    try:
                        result = await self.summary_chain_fallback.ainvoke(
                            {"input_documents": docs},
                            {"callbacks": callbacks}
                        )
                        
                        # Process result from fallback chain
                        if isinstance(result, dict):
                            final_summary = result.get("output_text", "[Summary generation failed: No output text]")
                        elif isinstance(result, str):
                            final_summary = result
                        else:
                            logger.warning(f"Unexpected result type from fallback chain: {type(result)}")
                            final_summary = f"[Summary generation failed: {e}]"
                    except Exception as fallback_e:
                        logger.error(f"Fallback summarization also failed: {fallback_e}", exc_info=True)
                        final_summary = f"[Summary generation failed: {e}]"
                
        # --- Cache the summary ---
        self.summaries[url] = final_summary
        
        return final_summary

    async def get_content_preview(self, url: str, callbacks: Optional[List[BaseCallbackHandler]] = None) -> Dict[str, Any]:
        """Get a preview of the content including a summary if available.
        
        If USE_PROGRESSIVE_LOADING is enabled, returns just a summary for faster response.
        Otherwise, returns the full content.

        Args:
            url: The URL or content ID to get a preview for
            callbacks: Optional callbacks for the LLM call

        Returns:
            Dict with preview data
        """
        url_or_id = url # Use the input directly, assuming it's the URL or an ID resolved elsewhere
        url = self.content_hash_map.get(url_or_id, url_or_id)
        
        if url not in self.documents:
            return {"error": f"No content found for URL/ID: {url_or_id}"}
            
        docs = self.documents[url]
        if not docs:
            return {"error": f"Empty document list for URL/ID: {url_or_id}"}
            
        metadata = docs[0].metadata
        title = metadata.get("title", "Unknown Title")
        source = metadata.get("source", url)
        
        if not self.use_progressive_loading:
            full_content = "\n\n".join([doc.page_content for doc in docs])
            return {
                "title": title,
                "source": source,
                "preview_text": full_content,
                "is_full_content": True
            }
            
        summary = await self.get_summary(url, callbacks=callbacks)
        
        return {
            "title": title,
            "source": source,
            "preview_text": summary,
            "is_full_content": False
        }

    def get_full_content(self, url_or_id: str) -> Dict[str, Any]:
        """Get the full content for a URL or content ID.

        Args:
            url_or_id: URL or content ID

        Returns:
            Dict with full content data
        """
        url = self.content_hash_map.get(url_or_id, url_or_id)
        
        if url not in self.documents:
            return {"error": f"No content found for: {url_or_id}"}
            
        docs = self.documents[url]
        if not docs:
            return {"error": f"Empty document list for: {url_or_id}"}
            
        metadata = docs[0].metadata
        title = metadata.get("title", "Unknown Title")
        source = metadata.get("source", url)
        
        full_content = "\n\n".join([doc.page_content for doc in docs])
        
        return {
            "title": title,
            "source": source,
            "full_content": full_content,
            "chunk_count": len(docs),
            "content_id": metadata.get("content_id", self._generate_content_id(url))
        }

    def clear_content(self, url_or_id: str = None):
        """Clear content and summaries for a URL or content ID.
        If no URL is provided, clears all content.

        Args:
            url_or_id: Optional URL or content ID to clear. If None, clears all content.
        """
        if url_or_id is None:
            logger.info("Clearing all stored content and summaries")
            self.documents.clear()
            self.summaries.clear()
            self.content_hash_map.clear()
        else:
            url = self.content_hash_map.get(url_or_id, url_or_id)
            if url in self.documents:
                logger.info(f"Clearing content for {url}")
                del self.documents[url]
            if url in self.summaries:
                del self.summaries[url]
            # Try to remove from content_hash_map if it's a content ID
            if url_or_id in self.content_hash_map:
                del self.content_hash_map[url_or_id]

    def get_stored_content_info(self) -> List[Dict[str, Any]]:
        """Get information about all stored content.

        Returns:
            List of dictionaries with content information
        """
        result = []
        for url, docs in self.documents.items():
            if not docs:
                continue
                
            # Get content ID from first document metadata
            metadata = docs[0].metadata
            content_id = metadata.get("content_id", self._generate_content_id(url))
            title = metadata.get("title", "Unknown Title")
            
            # Calculate total size
            total_size = sum(len(doc.page_content) for doc in docs)
            
            # Check if a summary exists
            has_summary = url in self.summaries
            
            result.append({
                "url": url,
                "content_id": content_id,
                "title": title,
                "chunks": len(docs),
                "total_size": total_size,
                "has_summary": has_summary
            })
            
        return result 

    async def summarize_search_results_as_table(self, search_results: list, top_n: int = 5, callbacks: Optional[list] = None) -> str:
        """
        Summarize a list of search results into a markdown table using the summarizer model.
        Each result should be a dict with keys: title, url/link, snippet.
        Returns a markdown table as a string.
        The LLM is instructed to select the 5 most promising, unbiased, and non-promotional results.
        """
        if not search_results or not isinstance(search_results, list):
            return "No search results to summarize."
        # Format as plain text for the LLM
        input_lines = []
        for i, result in enumerate(search_results, 1):
            title = result.get('title', result.get('Title', ''))
            url = result.get('url', result.get('link', ''))
            snippet = result.get('snippet', result.get('description', ''))
            input_lines.append(f"{i}. Title: {title}\nURL: {url}\nSnippet: {snippet}")
        input_text = "\n\n".join(input_lines)
        # Custom prompt for table formatting
        table_prompt = (
            "You are to select the 5 most promising, unbiased, and non-promotional search results from the list below. "
            "Format your output as a markdown table with columns: Index, Title, URL, Snippet. "
            "Do not add any commentary or explanation. Only include the 5 results you judge as best for a neutral, evidence-based research summary.\n\n"
            "Search Results:\n{input_text}"
        ).format(input_text=input_text)
        # Use the summarizer model directly
        if not self.summarization_llm:
            logger.error("No summarization LLM available for table summarization. Returning plain text.")
            return input_text
        # Call the LLM
        try:
            messages = [
                SystemMessage(content="You are a precise assistant. Always output a markdown table with the requested columns and no extra text. Choose the 5 most promising, unbiased, and non-promotional results."),
                HumanMessage(content=table_prompt)
            ]
            response = await self.summarization_llm.ainvoke(messages, callbacks=callbacks)
            if hasattr(response, 'content'):
                return response.content.strip()
            return str(response).strip()
        except Exception as e:
            logger.error(f"Error summarizing search results as table: {e}", exc_info=True)
            return input_text 