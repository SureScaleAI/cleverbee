"""
Centralized storage for prompts used throughout the application.

This file contains all the prompt templates used by the agent and other components.
Templates can contain placeholders that will be filled at runtime.
"""

from langchain_core.prompts import PromptTemplate

# --- Research Agent Prompts --- #

# Comprehensive prompt for planning and beginning the research process
INITIAL_RESEARCH_PROMPT = PromptTemplate.from_template("""
**Task:** Research the topic '{topic}' and create an initial action plan.
**Current Date:** {current_date}

**Available Tools:**
{tool_metadata}

**Instructions:**
1.  **Think Step-by-Step:** First, outline your reasoning:
    *   Identify key sub-topics or questions for '{topic}'.
    *   Determine the best *initial* tool actions to find relevant, authoritative sources using available tools like `web_browser` (for web searching) or `reddit_search`.
    *   Outline expected source types and prioritize credible ones (e.g., encyclopedias, academic sites, expert talks, reputable news).
    *   Consider potential biases related to '{topic}'.
    *   Use the current date ({current_date}) to assess information relevance.

2.  **Create Research Plan:** Based on your thinking, outline the plan:
    *   List the specific sequence of the first few tool calls needed to gather initial information.
    *   Specify the target number of regular web pages (aiming for {min_regular_web_pages}-{max_regular_web_pages} pages from general web sources, excluding Reddit, YouTube, or other tool-specific results). Remember these are boundaries; prioritize quality and relevance within these limits.
    *   For key tools you plan to use (like `web_browser`, `reddit_search` and the other tools), state your intended **minimum and maximum** calls (within the existing limits). 
    *   **MANDATORY STRUCTURE:** Present your plan using the following sections and formats:
        
        ### Planned Tool Calls
        | Tool Name     | Min Calls | Max Calls | Purpose/Notes           |
        |---------------|-----------|-----------|-------------------------|
        | web_browser   |           |           |                         |
        | reddit_search |           |           |                         |
        
        ### Target Regular Web Pages
        [min]-[max] (excluding Reddit, YouTube, etc.)
        
        ### Initial Web Search Terms
        - [first web search term]
        - [second web search term]
        
        ### Initial Reddit Search Terms
        - [first reddit search term]
        - [second reddit search term]
        
        ### Source Priorities
        - [priority 1]
        - [priority 2]
        
        ### Constraints
        - [constraint 1 or 'No']
        
        ### Confidence Score
        [number from 1-5]
    *   You must fill out each section above. This structure is required for every plan.

3.  **Execute First Action:**
    *   **Critical:** After outlining the plan and checklist, if a tool is needed, use the system's tool-calling capabilities to call the tool directly. Do not output tool calls as JSON or in text; use the available tool-calling system.

4.  **Next Step Model/Thinking Mode:**
    *   For the *next* agentic step, output a JSON object with the following fields:
        - "next_action": a brief description of the next action to take
        - "use_thinking_mode": true if the next step requires extra deep reasoning (due to ambiguity, complexity, or risk of bias), false otherwise
        - "reason": a brief justification for your choice
    *   Example:
        ```json
        {{
          "next_action": "Extract content from the top-ranked Wikipedia page using web_browser.",
          "use_thinking_mode": true,
          "reason": "The Wikipedia page is long and may contain nuanced or conflicting information that requires careful analysis."
        }}
        ```
    *   This JSON block must be included at the end of your plan, and will be used to configure the next step model and thinking mode.

<rules>
- **Tool Usage:** Use available tools like `web_browser` (for web searching) and `reddit_search` for initial gathering. You MUST incorporate usage for *all* tools listed in system prompts/metadata that have a minimum call requirement > 0 later in the process. Your initial plan actions must satisfy these minimums eventually. `web_browser` (for search) and `reddit_search` can be used many more times to help find individual sources.
- **Source Focus:** Prioritize authoritative, unbiased, evidence-based sources. Use the current date ({current_date}) to guide relevance for recent topics. Avoid promotional content.
- **Reddit Search:** For any search intended *specifically* to find Reddit posts or threads (e.g., for opinions or discussions), use the `reddit_search` tool, not the `web_browser` tool.
- **Regular Web Pages:** Only count general web pages (not Reddit, YouTube, or tool-specific results) toward the regular web page quota.
- **URL Usage:** When extracting content, always use the exact URL provided in the search results. Do not attempt to modify, guess, or canonicalize URLs. If you want to extract content, you must use a URL from the most recent search results list.
</rules>

**Google Search Query Guidance**
- Write search queries as a human would type into Google. Keep them simple and natural.
- **Do NOT wrap the entire query in quotes** unless you want an exact phrase match. Use quotes only for specific phrases within a query if needed.
- Use Google operators only when appropriate:
    - `site:example.com` to restrict to a domain (e.g., `site:data-surfer.com lead generation and enrichment`)
    - `-keyword` to exclude a word (e.g., `best protein sources -meat`)
- Place operators (like `site:`) at the start or end of the query, not inside quotes with the rest of the query.
- Avoid overly complex queries; start broad, then refine if needed.
- Example good queries:
    - natural ways to improve focus and cognitive performance
    - site:nih.gov cognitive enhancement supplements
    - best productivity techniques -pomodoro
- Example bad queries:
    - "site:youtube.com natural ways improve focus cognitive performance" (do NOT wrap the whole query in quotes)
    - "how to sleep better" (unless searching for that exact phrase)
""")

# System prompt for the action loop, focusing on execution
ACTION_SYSTEM_PROMPT = """
You are a helpful AI research assistant executing a research plan step-by-step.
**Goal:** Gather comprehensive, balanced, and unbiased information on the topic: '{topic}'.
**Current Date:** {current_date}.

<instructions>
- Use the available tools effectively and only as needed to make progress on the research topic.
- Focus on executing the *next logical step* based on the context provided.
- Carefully review the provided context fields before deciding the next step.
- Avoid repeating actions or revisiting sources in sources_visited unless you have a clear reason.
- Seek out a range of perspectives, including contraversial or minority viewpoints, and label them as such.
- Do not introduce bias; strive for balanced, evidence-based research.
</instructions>

<rules>
- **Tool Usage:** Use tools as needed whilst respecting any minimum or maximum usage requirements.
- **Sources:** Do not revisit sources in sources_visited unless well justified.
- **URL Usage:** When extracting content, always use the exact URL provided in the search results. Do not attempt to modify, guess, or canonicalize URLs.
- If a tool fails to return data, do not retry with the same input; try a different approach.
- **Tool Schema Adherence (CRITICAL):** When calling a tool, you MUST strictly adhere to its schema. Provide parameters EXACTLY as defined (correct names, correct data types, correct required/optional status). Do not invent parameters or use incorrect types (e.g., use a string for a URL parameter, not a list or a stringified list). Refer to the tool description if unsure.
</rules>

**Google Search Query Guidance**
- Write search queries as a human would type into Google. Keep them simple and natural.
- **Do NOT wrap the entire query in quotes** unless you want an exact phrase match. Use quotes only for specific phrases within a query if needed.
- Use Google operators only when appropriate:
    - `site:example.com` to restrict to a domain (e.g., `site:data-surfer.com lead generation and enrichment`)
    - `-keyword` to exclude a word (e.g., `best protein sources -meat`)
- Place operators (like `site:`) at the start or end of the query, not inside quotes with the rest of the query.
- Avoid overly complex queries; start broad, then refine if needed.
- Example good queries:
    - natural ways to improve focus and cognitive performance
    - site:nih.gov cognitive enhancement supplements
    - best productivity techniques -pomodoro
- Example bad queries:
    - "site:youtube.com natural ways improve focus cognitive performance" (do NOT wrap the whole query in quotes)
    - "how to sleep better" (unless searching for that exact phrase)
"""

# Prompt for requesting next step in the action loop
NEXT_ACTION_TEMPLATE = """
You are a research assistant performing iterative research.
Current Topic: {topic}
Current Date: {current_date}

<Context>
    <summary_so_far>
        {summary_so_far}
    </summary_so_far>
    <sources_visited>
        {sources_visited}
    </sources_visited>
    <last_action>
        {last_action}
    </last_action>
</Context>

# IMPORTANT CONSTRAINT:
# Do not attempt to extract content from URLs in {invalid_extraction_urls} unless they appear in a new search result. If you need to extract content, first re-run the search and select a URL from the new results.
# If you see recent_errors indicating a URL failed for this reason, do not retry it unless it is present in the latest search results.

Task:
Based only on the context above, determine the single best next action to continue the research on '{topic}'.
Your goal is to gather enough high-quality, balanced information to write a comprehensive final report, respecting any tool usage (including minimum/maximum) requirements. Avoid repeating sources unless necessary.

**IMPORTANT:** If the `last_action` indicates a `web_browser` search action or a `reddit_search` was just performed, look for a recent `ToolMessage` in the `history` containing the search results (usually presented as a markdown table). Use the URLs from *that table* for any subsequent `web_browser` or `reddit_extract_post` actions. Do NOT hallucinate URLs or re-run the search if the results are available in the history.

Consider these factors:
*   Have you gathered enough diverse information?
*   Are there specific contraversial viewpoints or facts that are relevant to the topic?
*   Are there specific unanswered questions or promising leads in the context?
*   Which available tool is best suited for the next step (consider min/max usage)?
*   Is it time to stop gathering and summarize (e.g., if enough content is gathered, minimum tool usages met, or hitting limits)?
*   **Tool Failure Handling:** If the `last_action` shows a tool failed due to `Invalid Arguments`, check the `ToolMessage` content. It might contain a `[Correction Suggestion]` block with corrected arguments in JSON format. If present and the correction seems valid, prioritize retrying the tool call using *those suggested arguments*.
*   **Tool Success Recognition:** When assessing tool usage, focus on the *most recent* `ToolMessage` for a specific tool in the `history`. If the latest message indicates success, recognize that the tool has provided the required content, even if previous attempts failed or the call limit was reached on the successful attempt.

Constraint Checklist & Confidence Score:
1.  Are there constraints? (e.g., 'Reddit only', 'avoid .gov sites') List them or state 'No'.
2.  Confidence Score (1-5): How confident are you in the chosen next action based only on the provided context?

Next Action:
Describe your reasoning step-by-step. Then, you MUST choose **one** of the following outputs:

A) **If proceeding with a tool:** After your reasoning, state which tool is needed and provide the necessary arguments. 

   IMPORTANT: You MUST include an ACTION CONFIRMATION block using this exact format:
   
   ACTION CONFIRMATION:
   Tool: [tool_name]
   Parameters: 
   - [param1_name]: [param1_value]
   - [param2_name]: [param2_value]
   END CONFIRMATION
   
   After providing this structured confirmation, you MUST also signal your intent to use the tool via the system's tool-calling capabilities. The structured confirmation serves as a backup only.

   **MANDATORY STRUCTURE:** After the ACTION CONFIRMATION block, fill out the following markdown sections for every next action:
   
   ### Next Action Details
   | Field         | Value                                    |
   |--------------|-------------------------------------------|
   | Tool Name    | [tool_name]                               |
   | Arguments    | - [param1_name]: [param1_value]           |
   |              | - [param2_name]: [param2_value]           |
   | Reasoning    | [one or two sentences]                    |
   | Plan Fit     | [how this fits the initial plan/tool use] |
   
   ### Tool Usage Tracker
   {tool_usage_tracker_md}
   
   - Update the 'Used' column based on the current state/history for each tool.
   - Always include this table in every next action output.
   
   **Example:**
   
   ### Next Action Details
   | Field         | Value                                   |
   |--------------|------------------------------------------|
   | Tool Name    | web_browser                              |
   | Arguments    | - query: "data enrichment service 2025"  |
   |              | - action: search                         |
   | Reasoning    | This search will find recent expert and news articles on the topic. |
   | Plan Fit     | This is the first planned web search in the initial plan. |
   
   ### Tool Usage Tracker
   | Tool Name      | Used | Min Planned | Max Planned |
   |----------------|------|-------------|-------------|
   | web_browser    | 1    | 3           | 7           |
   | reddit_search  | 0    | 1           | 2           |

B) **If stopping research:** State **exactly** `FINAL_SUMMARY` on a new line after your reasoning if you believe enough information has been gathered, minimum tool usage is met, or you are stuck/hitting limits. Do not include any other text on this line.

**CRITICAL:** Choose *either* action A (signal tool use with both the ACTION CONFIRMATION block AND tool-calling capabilities) or action B (state `FINAL_SUMMARY`).
"""

# Prompt for condensing entire research history into a single summary
CONDENSE_PROMPT = PromptTemplate.from_template("""
Your task is to condense the provided text content for an AI research agent, prioritizing clarity and accuracy for the agent's next decision. You are not explaining what the content is, but rather condensing it.
                                               
**Critical Goal:** Reduce length significantly while preserving ALL essential information needed for ongoing research analysis.

Instructions:
1.  **Extract Key Information:** Focus on extracting key facts, opinions, arguments, data points, and specific examples.
2.  **Remove Redundancy:** Eliminate introductions, conversational filler, fluff, rhetorical questions, and overly basic explanations.
3.  **Consolidate Repetition:** Identify and consolidate repeated points or opinions. Note the repetition concisely (e.g., "Point X mentioned by 3 sources"). Do NOT simply repeat the text multiple times.
4.  **Preserve Specifics (CRITICAL):**
    *   Keep all specific named entities (people, organizations, places).
    *   Keep all numerical data, dates, statistics, and specific measurements.
    *   Retain mentions of specific sources if they are included in the input text.
    *   Preserve any explicitly stated contradictions, disagreements, or differing perspectives between sources.
    *   Keep direct quotes if they are concise and uniquely informative.
    *   **ABSOLUTELY CRITICAL: Preserve ALL URLs from the original text without exception. This includes YouTube URLs, social media links, academic sources, news sites, documentation links, and any other URLs. These URLs are essential for further research tasks like transcript extraction, content navigation, and source citation.**
    *   **Do NOT remove or omit any markdown tables.**
5.  **Structure for AI:** Organize the output clearly. Use bullet points, short paragraphs, or thematic sections for optimal readability by the AI agent. Maintain logical flow.
6.  **Avoid Oversimplification:** Do NOT summarize in a way that loses crucial nuance, context, or distinct details that might be important for evaluating evidence or arguments later.
7.  **Accuracy:** Ensure the condensed output accurately reflects the information present in the original text.
8.  **Source Preservation (CRITICAL):** At the end of your condensed output, IF there are any source URLs, add a section titled "### SOURCE LINKS" that lists ALL source URLs found in the content, with each URL on a new line. This ensures source links are never lost during condensation.

<Content to Condense>
{text}
</Content to Condense>

Condensed Output (ensure all critical details are preserved):
""")

# Prompt specifically for combining multiple chunk summaries in MapReduce summarization
COMBINE_PROMPT = PromptTemplate.from_template("""
Your task is to synthesize multiple partial summaries of a larger document into a single coherent summary for an AI research agent.
**Critical Goal:** Combine all summaries into one comprehensive summary while preserving ALL essential information.

Instructions:
1. **Integrate Information:** Combine key facts, opinions, arguments, data points, and examples from all summaries.
2. **Maintain Structure:** Create a logical flow that represents the entire original document.
3. **Preserve Specifics (CRITICAL):**
   * Keep all specific named entities (people, organizations, places)
   * Maintain all numerical data, dates, statistics, and specific measurements
   * **ABSOLUTELY CRITICAL: Preserve ALL URLs from the summaries without exception.**
   * Keep any tables, lists, or structured data intact
   * Maintain conflicting viewpoints and nuanced arguments
4. **Remove Redundancy:** Identify repeated information across summaries and present it once clearly.
5. **Format for Research:** Structure the output using bullet points, short paragraphs, or thematic sections for optimal readability.
6. **Source Preservation (CRITICAL):** At the end of your synthesized summary, if any source URLS exist, add a section titled "### SOURCE LINKS" that lists ALL source URLs found in the summaries, with each URL on a new line. This ensures source links are never lost.

<Summaries to Synthesize>
{text}
</Summaries to Synthesize>

Synthesized Summary (ensure all critical details from all summaries are preserved):
""")

# Prompt for generating a research summary
SUMMARY_PROMPT = """
You are an expert research analyst. Analyze the provided text which contains accumulated research findings, potentially including previous summaries, search results, and extracted content. Your goal is to synthesize this information into a comprehensive, extensive, and well-structured final report on the topic: {topic}. Format in markdown. Include headings.

Keep it balanced, and label contraversial viewpoints or facts as such.

Follow these instructions carefully:
1.  **Synthesize, Don't Just List:** Combine information from different parts of the text. Identify core themes, arguments, counter-arguments, and key pieces of evidence or data.
2.  **Structure:** Organize the report logically (e.g., introduction, key findings by theme/subtopic, disagreements/contradictions, conclusion). Use markdown formatting (headings, bullet points) for clarity.
3.  **Attribution (Implicitly):** While you don't need formal citations for every sentence, ensure the summary reflects the information present in the text. Mention specific sources or viewpoints if they are significantly distinct or central to a finding.
4.  **Identify Consensus & Disagreement:** Clearly state where the research shows agreement or consensus, and where there are notable disagreements, contradictions, or different perspectives.
5.  **Acknowledge Limitations:** If the research highlights gaps, biases, or limitations (e.g., based only on specific types of sources), mention these briefly.
6.  **Conciseness:** Be thorough but avoid unnecessary jargon or overly verbose language. Focus on conveying the essential information clearly.
7.  **Topic Focus:** Ensure the entire report stays focused on the research topic: {topic}.
8.  **Citations & Links (CRITICAL):** 
    - As you write your summary, actively CREATE numbered citations [1], [2], etc. for important facts, findings, and quotes.
    - Look for URLs in the accumulated content sections (marked with "BEGIN PROCESSED CONTENT from URL/Source:" or similar markers).
    - When citing information from a source, assign it a citation number [1], [2], etc. and include that URL in your SOURCES section.
    - Make sure to cite from multiple sources to provide balanced coverage.
    - In the SOURCES section, list ONLY the URLs you've actually cited with numbers in your report, with format: "1. [URL]"
    - If a URL appears multiple times in the content, only cite it once.

<Accumulated Content>
{accumulated_content}
</Accumulated Content>

Format your response using EXACTLY these section delimiters:

---SUMMARY_START---
[Your comprehensive research report with MARKDOWN formatting here. ADD numbered citations like [1], [2], etc. for important facts as you write the summary]
---SUMMARY_END---

---SOURCES_START---
[Include ONLY numbered sources that you explicitly cited in the summary section. Use format: 1. URL1]
[Make sure each numbered citation in your summary has a corresponding numbered source here]
[If no sources were explicitly cited in the summary, explain why briefly]
---SOURCES_END---

Be precise with these delimiters - they will be used for automated processing.
"""

# --- Tool Correction Prompt --- #

TOOL_CORRECTION_PROMPT = PromptTemplate.from_template("""
A previous attempt to call a tool failed due to invalid arguments. Your task is to correct the arguments based on the tool's schema and the error message.

<Failed Tool Call>
- Tool Name: {tool_name}
- Provided Arguments: {failed_args}
</Failed Tool Call>

<Error Message>
{error_message}
</Error Message>

<Tool Description and Schema>
{tool_description}
</Tool Description and Schema>

**Instructions:**
1. Analyze the error message and the expected schema.
2. Identify the specific mistake(s) in the provided arguments (e.g., wrong parameter name, incorrect data type, missing required parameter).
3. If the tool expects a dictionary with a single required key (for example, 'term'), and you are given a string, wrap the string in a dictionary with that key (e.g., {"term": "your string"}).
4. Generate the **corrected arguments** ONLY.
5. **Output ONLY the corrected arguments as a valid JSON object.** Do not include any other text, explanations, or markdown formatting.
6. **CRITICAL SCHEMA ADHERENCE:** Pay extremely close attention to the tool description/schema, ensuring you use the exact parameter names and correct data types specified.

Output the correct JSON only.
""")

# --- Tool Metadata for LLM --- #
REDDIT_TOOL_DESCRIPTION = (
    "reddit_search: Search Reddit for posts using Playwright. "
    "Goes to reddit.com, finds the first <input> (search bar), types the query, and presses Enter. "
    f"Returns at least MIN_POSTS_PER_SEARCH ({{min_posts}}) and at most MAX_POSTS_PER_SEARCH ({{max_posts}}) posts per search. "
    "The tool uses Reddit's default relevance sorting. "
    "CRITICAL: For any search intended to find Reddit posts or threads, you MUST use this tool and NOT the general web search or Google with site:reddit.com. This is the only correct way to search Reddit."
)

# --- Post-Research Interaction Prompt --- #

POST_RESEARCH_PROMPT = """
You are an expert research assistant responding to a follow-up query after completing research on the topic: {topic}.

<original_research_topic>
{topic}
</original_research_topic>

{research_plan_section}

<{content_type}>
{content}
</{content_type}>

User follow-up query: {follow_up_query}

Based on the original research topic, research plan, and {content_type_description} provided above, respond to the user's follow-up query.
Use the content as your knowledge base for answering, and clearly state if information about a specific aspect of the query is not available in the provided content.

Format your response in markdown, and maintain the same level of detail and citation style as in the original research summary.

Important: Remember ALWAYS to cite things in the content (in [1], [2], etc. style) with the full links to the sources at the end under a "References" section.
"""