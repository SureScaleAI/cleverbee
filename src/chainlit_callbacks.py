import asyncio
import logging
import re
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

import chainlit as cl
from langchain_core.callbacks.base import AsyncCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_core.messages import BaseMessage

# Assuming TokenCallbackManager is available for injecting token updates
from src.token_callback import TokenCallbackManager, TokenCostProcess, _calculate_cost
from config.settings import (
    GEMINI_MODEL_NAME, 
    CLAUDE_MODEL_NAME, 
    PRIMARY_MODEL_TYPE,
    SUMMARIZER_MODEL,  # Import the actual summarizer model name
    USE_LOCAL_SUMMARIZER_MODEL, # Import the flag for local models
    TRACK_TOKEN_USAGE,
    PRIMARY_MODEL_NAME,
    CLAUDE_COST_PER_1K_INPUT_TOKENS,
    CLAUDE_COST_PER_1K_OUTPUT_TOKENS,
    GEMINI_COST_PER_1K_INPUT_TOKENS,
    GEMINI_COST_PER_1K_OUTPUT_TOKENS,
    NEXT_STEP_MODEL,
    # Removed GEMINI_SUMMARY_MODEL_NAME, ACTIVE_..._MODEL imports
)

logger = logging.getLogger(__name__)

# Helper function to format message content for display
def _format_message_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        return "\n".join([_format_message_content(item) for item in content])
    elif isinstance(content, dict):
        # Simple dict formatting, might need refinement for complex structures
        return "\n".join([f"**{k}:** {_format_message_content(v)}" for k, v in content.items()])
    else:
        return str(content)

# Function to detect and format reasoning/thinking steps as collapsible elements
def _format_reasoning_steps(content: str) -> str:
    """Identifies reasoning patterns in LLM output and formats them as collapsible elements.
    
    Detects common reasoning patterns like "Thinking Step-by-Step", "Reasoning", etc.,
    and wraps them in Chainlit's collapsible element syntax.
    
    Args:
        content: The raw LLM response text
        
    Returns:
        Formatted content with reasoning sections wrapped in collapsible elements
    """
    # Patterns to identify reasoning/thinking sections
    reasoning_patterns = [
        r"\*\*(?:1\.|Step 1:).*?Thinking Step-by-Step.*?\*\*",
        r"\*\*Reasoning:?\*\*",
        r"(?:^|\n)(?:1\.|Step 1:).*?Thinking",
        r"(?:^|\n)Let me think through this",
        r"(?:^|\n)I'll analyze this step by step",
        r"(?:^|\n)Let's break this down",
        r"(?:^|\n)First, I'll consider",
    ]
    
    # Check if content contains any reasoning patterns
    has_reasoning_pattern = any(re.search(pattern, content, re.IGNORECASE) for pattern in reasoning_patterns)
    
    if not has_reasoning_pattern:
        return content  # No reasoning patterns found, return original content
    
    # Find the end of reasoning sections - typically marked by conclusions, next steps, or actions
    conclusion_patterns = [
        r"\*\*(?:Conclusion|Summary|Action|Next Steps|Plan):?\*\*",
        r"(?:^|\n)(?:Conclusion|Summary|Action|Next Steps|Plan):",
        r"(?:^|\n)Based on (?:my|this) (?:analysis|reasoning)",
        r"(?:^|\n)Now I'll",
        r"(?:^|\n)I will now",
    ]
    
    # Try to find the start and end indices of the reasoning section
    reasoning_start = 0
    reasoning_end = len(content)
    
    for pattern in reasoning_patterns:
        match = re.search(pattern, content, re.IGNORECASE)
        if match and match.start() < reasoning_start:
            reasoning_start = match.start()
    
    for pattern in conclusion_patterns:
        match = re.search(pattern, content, re.IGNORECASE)
        if match and match.start() > reasoning_start and match.start() < reasoning_end:
            reasoning_end = match.start()
    
    # If no clear conclusion marker found but we have a reasoning pattern, 
    # look for tool call section or just wrap the entire content
    if reasoning_end == len(content) and has_reasoning_pattern:
        tool_call_match = re.search(r"\`\`\`json", content)
        if tool_call_match and tool_call_match.start() > reasoning_start:
            reasoning_end = tool_call_match.start()
    
    # Extract reasoning section and remaining content
    if reasoning_start < reasoning_end:
        reasoning_section = content[reasoning_start:reasoning_end].strip()
        remaining_content = content[:reasoning_start] + content[reasoning_end:]
        
        # Create a collapsible element for the reasoning section
        formatted_content = (
            remaining_content[:reasoning_start].strip() + 
            f"\n\n<details>\n<summary>View Reasoning</summary>\n\n{reasoning_section}\n\n</details>\n\n" +
            remaining_content[reasoning_start:].strip()
        )
        return formatted_content
    
    return content  # Fallback if we couldn't properly identify sections

class ChainlitCallbackHandler(AsyncCallbackHandler):
    """Callback handler for Chainlit UI updates during LangChain execution."""

    # Define author names
    PRIMARY_AUTHOR = "Researcher"
    SUMMARY_AUTHOR = "Summarizer"
    NEXT_STEP_AUTHOR = "NextStepAgent"

    def __init__(self, token_processor: Optional[TokenCostProcess] = None):
        """Initialize the callback handler.
        
        Args:
            token_processor: An optional TokenCostProcess instance to display token usage.
        """
        super().__init__()
        # Store dict containing step object and author name
        # self.current_steps: Dict[str, cl.Step] = {}
        self.current_steps: Dict[str, Dict[str, Any]] = {}
        self.token_manager = token_processor # Renamed internal attribute for clarity, but kept API name
        # Store the root message for potential updates (e.g., final summary)
        self._root_message: Optional[cl.Message] = None 
        # Cache root logger level check for efficiency
        self._log_traceback = logger.isEnabledFor(logging.DEBUG)

    # --- Lifecycle Methods --- 

    async def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Called when a chat model starts. Creates and sends the step immediately.""" # Docstring updated
        # Check if UI updates should be skipped
        if metadata and metadata.get("skip_ui_updates"):
            logger.debug(f"Skipping UI updates for LLM call (run_id: {run_id}) due to skip_ui_updates flag")
            return
            
        # Try getting model name from metadata or serialized info
        model_name = "LLM" # Default
        if metadata and metadata.get('model_name'):
            model_name = metadata.get('model_name')
        elif kwargs.get("invocation_params"):
            model_name = kwargs["invocation_params"].get("model", kwargs["invocation_params"].get("model_name", "LLM"))
        elif serialized and serialized.get('name'):
             model_name = serialized.get('name')
             
        # Clean up potential prefixes like models/
        if model_name != "LLM":
             model_name = model_name.split('/')[-1]

        logger.info(f"LLM Start Detected: Model='{model_name}' (run_id: {run_id})") # INFO: High-level start
        logger.debug(f"on_chat_model_start details: run_id={run_id}, parent_run_id={parent_run_id}") # DEBUG: Details
        logger.debug(f"  serialized: {serialized}") # DEBUG: Verbose details
        logger.debug(f"  metadata: {metadata}")     # DEBUG: Verbose details
        logger.debug(f"  kwargs: {kwargs}")         # DEBUG: Verbose details

        parent_step = self._get_parent_step(parent_run_id)
        
        # --- Determine Author --- 
        author_name = self.PRIMARY_AUTHOR # Default to primary
        status_text = "Thinking..." # Default status

        # Simplify model names for comparison
        summarizer_model_simple = SUMMARIZER_MODEL.lower().split('/')[-1] if SUMMARIZER_MODEL else None
        next_step_model_simple = NEXT_STEP_MODEL.lower().split('/')[-1] if NEXT_STEP_MODEL else None
        model_name_simple = model_name.lower().split('/')[-1]

        # Check if the current model matches the SUMMARIZER_MODEL
        if summarizer_model_simple and (
            summarizer_model_simple in model_name_simple or model_name_simple.startswith(summarizer_model_simple)
        ):
            author_name = self.SUMMARY_AUTHOR
            local_tag = " (Local)" if USE_LOCAL_SUMMARIZER_MODEL else ""
            status_text = f"Summarizing content...{local_tag}"
        # <<< ADDED CHECK: Check if it matches the NEXT_STEP_MODEL >>>
        elif next_step_model_simple and (
            next_step_model_simple in model_name_simple or model_name_simple.startswith(next_step_model_simple)
        ):
            author_name = self.NEXT_STEP_AUTHOR # Use the dedicated author name
            status_text = "Deciding next step..."
        # Keep the default (PRIMARY_AUTHOR) if it's neither Summarizer nor Next Step
        else:
             author_name = self.PRIMARY_AUTHOR
             status_text = "Planning/Reasoning..."
            
        logger.info(f"Chainlit Callback: Determined author_name='{author_name}' for run_id {run_id}")
        
        # Create a message instead of a step for a more user-friendly display
        thinking_msg = cl.Message(
            content=status_text,
            author=author_name,
            parent_id=parent_step.id if parent_step else None
        )
        await thinking_msg.send()
        
        # Store message object AND author name
        self.current_steps[str(run_id)] = {"step": thinking_msg, "author": author_name}

    async def on_llm_end(
        self, 
        response: LLMResult,
        *,
        run_id: UUID, 
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any
    ) -> None:
        """Called when LLM call ends. Sends output messages parented to the existing step.""" # Docstring updated
        logger.debug(f"on_llm_end called: run_id={run_id}, parent_run_id={parent_run_id}") # Debug only
        
        # Check if UI updates should be skipped
        if kwargs.get("metadata") and kwargs.get("metadata").get("skip_ui_updates"):
            logger.debug(f"Skipping UI updates for LLM call end (run_id: {run_id}) due to skip_ui_updates flag")
            return
        
        # <<< ADDED CHECK FOR TAG >>>
        # Access tags from kwargs if they exist
        tags = kwargs.get("tags", [])
        if "final_summary_llm" in tags:
            logger.info(f"Skipping on_llm_end message send for final summary (run_id: {run_id})")
            # Remove the thinking message associated with this run
            step_info = self.current_steps.pop(str(run_id), None)
            if step_info and step_info.get("step"):
                try:
                    await step_info["step"].remove()
                    logger.debug(f"Removed thinking message for final summary run {run_id}")
                except Exception as e:
                    logger.warning(f"Failed to remove thinking message for final summary run {run_id}: {e}")
            return # Don't process further for the final summary step
        # <<< END CHECK FOR TAG >>>
            
        # Get previously stored step dict
        step_info = self.current_steps.get(str(run_id))
        if not step_info:
            logger.warning(f"No stored step info found for LLM call: {run_id}")
            return # Nothing to update
        
        # Extract author and step (which is now actually a message)
        author = step_info.get("author", self.PRIMARY_AUTHOR) # Default if missing
        thinking_msg = step_info.get("step") # This is now a cl.Message, not a cl.Step
        
        # Get output content (mostly the same logic)
        output_content = "(No text content)"
        tool_calls_str = "" # String for any tool calls
        total_tokens = 'N/A' # Default for token usage
        
        # At least one response expected
        if response.generations and len(response.generations) > 0:
            # Process first generation
            generation = response.generations[0]
            if len(generation) > 0:
                # Get first generation's text content (most common)
                first_gen = generation[0]
                
                if hasattr(first_gen, 'text') and first_gen.text:
                    output_content = first_gen.text
                elif hasattr(first_gen, 'message'):
                    # Handle chat message format
                    if hasattr(first_gen.message, 'content') and first_gen.message.content:
                        output_content = first_gen.message.content
                    
                    # Handle tool calls if present (adjust based on actual structure)
                    if hasattr(first_gen.message, 'tool_calls') and first_gen.message.tool_calls:
                        tool_calls = first_gen.message.tool_calls
                        # Format tool calls into a readable string
                        tool_calls_str = "\n".join([
                            f"**Tool:** `{tc.name}`\n**Input:** ```json\n{tc.args}\n```"
                            for tc in tool_calls if hasattr(tc, 'name') and hasattr(tc, 'args')
                        ])
                
                # Format additional first_gen model output attributes if present
                if hasattr(first_gen, 'generation_info') and first_gen.generation_info:
                    # If timestamp in generation info, log it for tracking
                    if 'timestamp' in first_gen.generation_info:
                        logger.debug(f"Generation timestamp: {first_gen.generation_info['timestamp']}")
                    
                    # Log finish reason if provided 
                    if 'finish_reason' in first_gen.generation_info:
                        finish_reason = first_gen.generation_info['finish_reason']
                        if finish_reason != 'stop':
                            logger.warning(f"Generation finished with reason: {finish_reason}")
        
        # Get token usage from llm_output if available
        if hasattr(response, 'llm_output') and response.llm_output:
            llm_output_dict = response.llm_output
            
            if isinstance(llm_output_dict, dict) and 'token_usage' in llm_output_dict:
                token_usage = llm_output_dict['token_usage']
                
                if isinstance(token_usage, dict):
                    prompt_tokens = token_usage.get('prompt_tokens', 0)
                    completion_tokens = token_usage.get('completion_tokens', 0)
                    # Get total tokens, checking for key or calculating
                    if 'total_tokens' in token_usage:
                        total_tokens = token_usage['total_tokens']
                    else:
                        total_tokens = prompt_tokens + completion_tokens
                    
                    # Log token usage at debug level
                    logger.debug(f"Token usage: prompt={prompt_tokens}, completion={completion_tokens}, total={total_tokens}")
        
        # --- Remove the original thinking message --- 
        try:
            await thinking_msg.remove()
            logger.debug(f"Removed thinking message for {run_id}")
            # Remove from our tracking dict too
            self.current_steps.pop(str(run_id), None) 
        except Exception as e:
             logger.warning(f"Failed to remove thinking message for {run_id}: {e}")
        # --- Message Removed ---
        
        # Format reasoning steps as collapsible elements if present
        if output_content and output_content != "(No text content)":
            formatted_content = _format_reasoning_steps(output_content)
            
            # Determine if this content has reasoning to apply appropriate tags
            has_reasoning = formatted_content != output_content
            
            # Check if content has tool calls to add appropriate tags 
            has_tool_calls = bool(tool_calls_str) or ("ACTION CONFIRMATION:" in output_content)
            
            # Build metadata with useful information
            metadata = {
                "model_name": self.model_name if hasattr(self, 'model_name') else "unknown",
                "has_reasoning": has_reasoning,
                "has_tool_calls": has_tool_calls
            }
            
            # Add token usage to metadata if available
            if total_tokens != 'N/A' and total_tokens != 0:
                metadata["token_usage"] = {
                    "total_tokens": total_tokens
                }
                if hasattr(response, 'llm_output') and response.llm_output:
                    llm_output_dict = response.llm_output
                    if isinstance(llm_output_dict, dict) and 'token_usage' in llm_output_dict:
                        token_usage = llm_output_dict['token_usage']
                        if isinstance(token_usage, dict):
                            metadata["token_usage"]["prompt_tokens"] = token_usage.get('prompt_tokens', 0)
                            metadata["token_usage"]["completion_tokens"] = token_usage.get('completion_tokens', 0)
            
            # Create tags based on message content
            tags = []
            if has_reasoning:
                tags.append("reasoning")
            if has_tool_calls:
                tags.append("tool-action")
            if author == self.SUMMARY_AUTHOR:
                tags.append("summary")
            
            # Send the (potentially) formatted content with tags and metadata
            await cl.Message(
                content=formatted_content,
                author=author,
                tags=tags,
                metadata=metadata
            ).send()
            
        # Send tool call information separately if present
        if tool_calls_str:
            await cl.Message(
                content=f"**Action:**\n{tool_calls_str}", 
                author=author,
                tags=["tool-call"],
                metadata={"tool_names": [tc.name for tc in tool_calls if hasattr(tc, 'name')]}
            ).send()
            
        # --- Removed Token Usage Display logic from on_llm_end ---
        # The final summary display happens only at the end via display_final_token_summary
        if total_tokens == 'N/A' or total_tokens == 0:
             logger.debug(f"No token usage reported or tracked for LLM run {run_id}.")
        # --- End Token Usage Display ---

    async def on_llm_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Called when an LLM call errors. Updates the user with an error message."""
        step_info = self.current_steps.get(str(run_id)) # Get stored dict
        
        # Enhanced error logging with details about the error type and message
        error_type = type(error).__name__
        error_message = str(error)
        logger.error(f"LLM Error (run_id: {run_id}): {error_type} - {error_message}", exc_info=self._log_traceback)
        
        # Log additional diagnostic information that might be in kwargs
        if 'metadata' in kwargs and kwargs['metadata']:
            logger.error(f"Error metadata: {kwargs['metadata']}")
        
        # Check if we can retry the error based on its type/message
        retryable_error = any(err in error_message for err in [
            "Rate limit", "timeout", "connection", "server error",
            "contents.parts must not be empty", "GenerateContentRequest.contents",
            "Invalid argument provided to Gemini"
        ])
        
        retry_suggestion = ""
        if retryable_error:
            retry_suggestion = " This appears to be a temporary error. You may want to try again."
        
        if step_info:
            thinking_msg = step_info.get("step") # This is now a message
            author = step_info.get("author", self.PRIMARY_AUTHOR)
            
            # Remove the thinking message
            try:
                await thinking_msg.remove()
                logger.debug(f"Removed thinking message for errored {run_id}")
            except Exception as e:
                logger.warning(f"Failed to remove thinking message for errored {run_id}: {e}")
            
            # Send an error message with the proper author - FIXED: removed is_error parameter
            await cl.Message(
                content=f"⚠️ Error: {error_message}{retry_suggestion}",
                author=author
            ).send()
            
            # Remove from tracking
            self.current_steps.pop(str(run_id), None)
        else:
            # Error occurred but no step was found (should be rare)
            await cl.ErrorMessage(content=f"LLM Error: {error_message}{retry_suggestion}").send()

    async def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Called when a tool starts. Creates and sends the step immediately."""
        tool_name = metadata.get('tool_name', serialized.get('name', 'Tool')) if metadata else serialized.get('name', 'Tool')
        logger.info(f"Tool Start: {tool_name} (run_id: {run_id})") # INFO: High-level start
        logger.debug(f"on_tool_start details: run_id={run_id}, parent_run_id={parent_run_id}") # DEBUG: Details
        logger.debug(f"  tool_input: {input_str}") # DEBUG: Input details
        logger.debug(f"  serialized: {serialized}") # DEBUG: Verbose details
        logger.debug(f"  metadata: {metadata}")     # DEBUG: Verbose details

        parent_step = self._get_parent_step(parent_run_id)
        
        # Send Step immediately
        step = cl.Step(
            name=f"🛠️ Running Tool: `{tool_name}`",
            parent_id=parent_step.id if parent_step else None,
            type="tool"
        )
        step.input = f"```json\n{input_str}\n```" # Set input on the object
        step.id = str(run_id) # Assign STRING version of run_id
        await step.send()
        # Store step object and author in a dictionary for consistency
        self.current_steps[str(run_id)] = {"step": step, "author": "Tool"} # Store Step object and author

    async def on_tool_end(
        self,
        output: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Called when a tool ends. Sends output message parented to existing step."""
        step_info = self.current_steps.pop(str(run_id), None) # Remove step from tracking on end
        if not step_info or not step_info.get("step"):
            logger.warning(f"on_tool_end called for run_id {run_id} but no step found.")
            return
        
        step = step_info["step"]
        logger.info(f"Tool End: {step.name} completed (run_id: {run_id})") # INFO: High-level end
        logger.debug(f"Tool Output (run_id: {run_id}): {output[:200]}...") # DEBUG: Output preview
        step.output = output
        # await step.update() # No need to update, just send ToolMessage

        # Extract tool name for tagging and metadata
        tool_name = step.name.split('`')[1] if '`' in step.name else "unknown-tool"
        
        # --- Send Tool Output as Message ---
        tool_output_message = cl.Message(
             content=f"**Tool Result (`{tool_name}`):**\n```\n{output}\n```", # Format as code block
             parent_id=step.parent_id, # Parent to the original LLM step request
             author="Tool", # Specific Tool author
             tags=["tool-result", tool_name], # Add tool name as specific tag
             metadata={
                 "tool_name": tool_name,
                 "output_length": len(output)
             }
        )
        await tool_output_message.send()
        # --- End Send Tool Output ---

        # --- Remove Tool Step ---
        try:
             await step.remove()
             logger.debug(f"Removed tool step {step.id}")
        except Exception as e:
            logger.warning(f"Failed to remove tool step {step.id}: {e}")
        # --- End Remove Tool Step ---

    async def on_tool_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Called when tool execution fails."""
        logger.warning(f"Entering on_tool_error for run_id={run_id}, error type={type(error)}")
        parent_step = self._get_parent_step(parent_run_id)
        step_name = f"Tool Error: {kwargs.get('name', 'Unknown Tool')}"
        
        # --- Robust Error Message Extraction ---
        error_message = "Unknown error"
        try:
            # Most common case: Standard Exception
            if isinstance(error, Exception):
                # Extract arguments if available, otherwise just stringify
                error_message = str(error.args[0]) if error.args else str(error)
            # Handle specific types if needed (e.g., if Step object was received)
            elif hasattr(error, 'message'): # Check for a common 'message' attribute
                error_message = str(getattr(error, 'message'))
            elif hasattr(error, '__str__'): # Fallback to string representation
                 error_message = str(error)
        except Exception as extraction_err:
             logger.error(f"Failed to extract detailed error message in on_tool_error: {extraction_err}", exc_info=self._log_traceback)
             error_message = f"Failed to extract error details ({type(error).__name__})"
        # --- End Robust Error Message Extraction ---
        
        logger.error(f"Tool Error (run_id={run_id}): {step_name} failed. Reason: {error_message}", exc_info=self._log_traceback)
        
        await cl.Message(
            content=f"❌ Error during tool execution: `{step_name}` failed.\n**Reason:** {error_message}",
            parent_id=parent_step.id if parent_step else None, 
            author="System",
            # Indent the message slightly to indicate it's under the LLM's action
            # indent=1 
        ).send()

    async def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """Called when a chain starts. Creates and sends step if not filtered.""" # Docstring update
        
        # --- Log received data --- 
        logger.debug(f"on_chain_start: run_id={run_id}, parent_run_id={parent_run_id}")
        logger.debug(f"  serialized: {serialized}")
        logger.debug(f"  metadata: {metadata}")
        # --- End Log --- 
        
        # --- Safely get chain_name --- 
        chain_name = "Chain"
        if metadata and metadata.get('chain_name'):
            chain_name = metadata.get('chain_name')
        elif serialized and serialized.get('id'):
            chain_name = serialized.get('id', ['Chain'])[-1]
        # --- End safe get ---
        logger.debug(f"  Determined chain_name: {chain_name}") # Log determined name

        # --- Filtering --- 
        ignored_chain_types = ["ChatPromptTemplate", "RunnablePassthrough", "RunnableAssign", "RunnableLambda", "RunnableSequence"]
        is_ignored_by_id = False
        if serialized and serialized.get('id'): # Check serialized and 'id' exists
             is_ignored_by_id = any(ignored in str(id_part) for id_part in serialized.get('id', []) for ignored in ignored_chain_types)

        logger.debug(f"  Filtering Check: chain_name='{chain_name}', IDs={serialized.get('id', []) if serialized else 'N/A'}, is_ignored_by_id={is_ignored_by_id}") # Log filtering decision points

        # Ignore if name is the default "Chain", explicitly in the list, or matched by ID
        if chain_name == "Chain" or chain_name in ignored_chain_types or is_ignored_by_id:
            logger.debug(f"  Result: Ignoring step creation for chain: {chain_name} (run_id: {run_id})") # Keep log
            return # Return silently, do not store or send step
        else:
            logger.debug(f"  Result: Creating step for chain: {chain_name} (run_id: {run_id})")
        # --- End Filtering --- 
        
        parent_step = self._get_parent_step(parent_run_id)
        
        # Send Step immediately
        step = cl.Step(
            name=f"⛓️ Entering Chain: `{chain_name}`",
            parent_id=parent_step.id if parent_step else None,
            type="chain"
        )
        formatted_input = "\n".join([f"**{k}:** `{v}`" for k, v in inputs.items() if k != 'history' and k != 'chat_history'])
        step.input = f"**Inputs:**\n{formatted_input}" # Set input
        step.id = str(run_id) # Assign STRING version of run_id
        await step.send()
        # Store step object and author in a dictionary
        self.current_steps[str(run_id)] = {"step": step, "author": "Chain"} # Standardize storage

    async def on_chain_end(
        self,
        outputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Called when a chain ends. Sends output message parented to existing step."""
        step_info = self.current_steps.get(str(run_id))
        chain_name = "Unknown Chain"
        step_object = None # Initialize
        
        # Properly check step_info structure and extract step
        if step_info and isinstance(step_info, dict) and "step" in step_info and isinstance(step_info["step"], cl.Step):
            step_object = step_info["step"]
            chain_name = step_object.name # Get name from the step object
        
        # Check if this chain was ignored during start
        ignored_on_start = step_object is None # If no step object was found, it was ignored
        if ignored_on_start and kwargs.get('name'): # Log name if available even if ignored
                 chain_name = kwargs['name']

        if not ignored_on_start and step_object:
            logger.info(f"Chain End: {chain_name} (run_id: {run_id})") # INFO: High-level end
            logger.debug(f"Chain Output (run_id: {run_id}): {outputs}") # DEBUG: Output details
            
            # Remove chain step immediately on end
            try:
                await step_object.remove()
                logger.debug(f"Removed chain step {step_object.id}")
                self.current_steps.pop(str(run_id), None)
            except Exception as e:
                logger.warning(f"Failed to remove chain step {step_object.id}: {e}")
        elif step_info and not step_object:
             # Handle case where step_info exists but is not the expected dict
             logger.warning(f"Chain step_info found for {run_id} but has unexpected structure or missing step: {step_info}")
             self.current_steps.pop(str(run_id), None)  # Clean up anyway
        else:
            # Log end even if start was ignored, but at DEBUG level
            logger.debug(f"Chain End (previously ignored): {chain_name} (run_id: {run_id})")

    async def on_chain_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Called when a chain errors. Marks existing step as error."""
        step_info = self.current_steps.pop(str(run_id), None) # Remove step from tracking on error
        chain_name = "Unknown Chain"
        if step_info and step_info.get("step"):
             step = step_info["step"]
             chain_name = step.name.split('`')[1]
             step.output = f"Chain Error: {error}"
             step.error = str(error)
             await step.update() # Show error in the step
             logger.error(f"Chain Error: {chain_name} (run_id: {run_id}) failed: {error}", exc_info=self._log_traceback) # Conditional traceback
             # Don't remove the step, let it show the error
        else:
             # Try to get name from kwargs if available for logging
             if kwargs.get('name'): chain_name = kwargs['name']
             logger.error(f"Chain Error: {chain_name} (run_id: {run_id}) failed: {error}", exc_info=self._log_traceback) # Conditional traceback

    # Helper method to find the parent step
    def _get_parent_step(self, parent_run_id: Optional[UUID]) -> Optional[Any]: # Return any object with an id
        if parent_run_id and str(parent_run_id) in self.current_steps:
            parent_info = self.current_steps[str(parent_run_id)]
            # Return the step/message object from the stored dict as long as it has an id
            if isinstance(parent_info, dict) and "step" in parent_info:
                parent_obj = parent_info["step"]
                if hasattr(parent_obj, "id"):
                    return parent_obj
                else:
                    logger.warning(f"Parent {parent_run_id} object found but has no id attribute")
                    return None
            else:
                logger.warning(f"Parent {parent_run_id} data found but is not in expected format. Type: {type(parent_info)}")
                return None
        return None

    # --- Chainlit Specific Methods (Optional but useful) ---
    def set_root_message(self, message: cl.Message):
        """Sets the root message for potential future updates."""
        self._root_message = message

    async def display_final_token_summary(self):
        """Displays the final token usage summary in the chat."""
        # Check if token_manager exists and try to access its processor
        if not self.token_manager:
            logger.warning("Token manager not available for final summary.")
            await cl.Message(content="Token usage statistics not available for this session.", author="System Info").send()
            return
            
        try:
            # <<< ADDED LOGGING >>>
            logger.info(f"Attempting to display final token summary. Token Manager object: {self.token_manager}")
            
            # <<< MODIFIED: self.token_manager IS the processor >>>
            processor = self.token_manager 
            # logger.info(f"Accessed token_cost_processor: {processor}") # Log the processor object - Removed as redundant
            
            if not processor:
                # <<< ADDED LOGGING >>>
                logger.error("token_cost_processor (self.token_manager) is None. Cannot generate summary.")
                raise AttributeError("token_cost_processor (self.token_manager) is None")
                
            model_usage = processor.model_usage
            # <<< ADDED LOGGING >>>
            logger.info(f"Retrieved model_usage data: {model_usage}") # Log the actual usage data
            
            if not model_usage:
                logger.warning("No model usage data available in token_cost_processor")
                await cl.Message(content="No token usage was recorded during this session.", author="System Info").send()
                return
                
            total_cost = processor.total_cost
            
            # Build simplified model usage table with cleaner model names
            summary_lines = ["### Token Usage & Cost Summary"]
            
            # Create the table header
            summary_lines.append("\n| Model | Input Tokens | Output Tokens | Total | Cost |")
            summary_lines.append("| --- | ---: | ---: | ---: | ---: |")
            
            total_input_tokens = 0
            total_output_tokens = 0
            
            # Mapping for display names (using base names for prefix matching)
            display_names = {
                'claude-3-7-sonnet': 'Claude 3.7 Sonnet',
                'gemini-2.5-pro': 'Gemini 2.5 Pro', 
                'gemini-2.5-flash': 'Gemini 2.5 Flash',
                'gemini-2.0-flash': 'Gemini 2.0 Flash',
                # Removed specific preview keys as prefix matching handles them
            }
            
            # Sort models for consistent display order
            for model_key in sorted(model_usage.keys()):
                usage_data = model_usage[model_key]
                
                # Clean up model name for display using prefix matching
                model_key_lower = model_key.lower()
                display_name = model_key # Default to original key if no match found
                
                # Sort display_names keys by length descending to prioritize more specific matches
                sorted_keys = sorted(display_names.keys(), key=len, reverse=True)
                for key in sorted_keys:
                    if model_key_lower.startswith(key):
                        display_name = display_names[key]
                        break # Use the first (most specific) match found
                
                # Ensure tokens are integers for formatting
                input_tokens = int(usage_data.get('input_tokens', 0))
                output_tokens = int(usage_data.get('output_tokens', 0))
                total_tokens = input_tokens + output_tokens
                
                # Calculate model cost
                model_cost = usage_data.get('total_cost', 0.0)
                
                # Accumulate totals
                total_input_tokens += input_tokens
                total_output_tokens += output_tokens
                
                if total_tokens > 0: # Only display models that were used
                    line = f"| **{display_name}** | {input_tokens:,} | {output_tokens:,} | {total_tokens:,} | ${model_cost:.4f} |"
                    summary_lines.append(line)

            # Add total row
            total_tokens = total_input_tokens + total_output_tokens
            summary_lines.append(f"| **TOTAL** | **{total_input_tokens:,}** | **{total_output_tokens:,}** | **{total_tokens:,}** | **${total_cost:.4f}** |")
            
            # Add cache hit information if available
            if hasattr(self.token_manager, 'cache_hits') and self.token_manager.cache_hits:
                cache_hits = self.token_manager.cache_hits
                
                summary_lines.append("\n### Cache Savings")
                summary_lines.append("\n| Model | Cache Hits | Input Tokens Saved | Output Tokens Saved | Total Tokens Saved | Cost Saved |")
                summary_lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
                
                total_saved_tokens = 0
                total_saved_cost = 0.0
                total_hits = 0
                
                for model_name, cache_data in cache_hits.items():
                    # Get clean display name 
                    display_name = display_names.get(model_name.lower(), model_name)
                    
                    input_tokens = cache_data['input_tokens']
                    output_tokens = cache_data['output_tokens']
                    total_tokens = cache_data['total_tokens']
                    hits = cache_data['hits']
                    
                    total_hits += hits
                    total_saved_tokens += total_tokens
                    
                    # Calculate cost savings for this model
                    saved_cost = _calculate_cost(model_name, input_tokens, output_tokens)
                    total_saved_cost += saved_cost
                    
                    summary_lines.append(f"| **{display_name}** | {hits} | {input_tokens:,} | {output_tokens:,} | {total_tokens:,} | ${saved_cost:.4f} |")
                
                # Add total savings row
                summary_lines.append(f"| **TOTAL SAVINGS** | **{total_hits}** | - | - | **{total_saved_tokens:,}** | **${total_saved_cost:.4f}** |")
                
                # Add net cost after savings
                net_cost = max(0, total_cost - (self.token_manager.total_saved_cost if hasattr(self.token_manager, 'total_saved_cost') else 0.0))
                summary_lines.append(f"\n**Net Cost (after cache savings): ${net_cost:.4f}**")
            
            summary_message_content = "\n".join(summary_lines)
            await cl.Message(content=summary_message_content, author="System Info").send()
        except AttributeError as ae:
            # <<< UPDATED LOGGING >>>
            logger.error(f"AttributeError accessing token manager/processor attributes: {ae}", exc_info=True)
            await cl.Message(content=f"Error retrieving token usage details: {ae}", author="System Info").send()
        except Exception as e:
            # <<< UPDATED LOGGING >>>
            logger.error(f"Unexpected error in display_final_token_summary: {e}", exc_info=True)
            await cl.Message(content=f"Error displaying token usage summary: {e}", author="System Info").send()

    async def ask_for_captcha_completion(self, captcha_page_url: Optional[str] = None):
        """Sends a message to Chainlit asking the user to solve a CAPTCHA and waits for confirmation.
        
        Args:
            captcha_page_url: The URL where the CAPTCHA might be found (optional).
        """
        logger.info("Requesting manual CAPTCHA completion via Chainlit UI.")
        message_content = ("🤖 CAPTCHA detected! Please solve the challenge in the separate browser window. "
                           "Click 'Continue' below once you are done.")
        if captcha_page_url:
            message_content += f"\n(CAPTCHA likely on: {captcha_page_url})"
            
        actions = [
            cl.Action(name="continue", value="continue", label="✅ Continue", description="Click after solving CAPTCHA")
        ]
        
        # Use AskActionMessage to pause and wait for user click
        # Set a timeout (e.g., 5 minutes) to avoid hanging indefinitely
        try:
            await cl.AskActionMessage(
                content=message_content,
                actions=actions,
                author="System", # Or maybe "Browser Tool"?
                timeout=300 # 5 minutes timeout
            ).send()
            logger.info("User clicked 'Continue' in Chainlit after CAPTCHA prompt.")
        except TimeoutError:
            logger.error("Timed out waiting for user to confirm CAPTCHA completion in Chainlit.")
            # Decide how to handle timeout - maybe raise an error to stop the process?
            raise TimeoutError("User did not confirm CAPTCHA completion within the time limit.")
        except Exception as e:
             logger.error(f"Error asking for CAPTCHA completion in Chainlit: {e}", exc_info=True)
             # Fallback or re-raise depending on desired behavior
             raise 