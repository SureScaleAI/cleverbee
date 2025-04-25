# Potential Project Enhancements & Ideas

*   **Internal Vector Store Integration:**
    *   Integrate a vector store (e.g., ChromaDB, FAISS) to store extracted content chunks.
    *   Use semantic retrieval *during* report generation to pull the most relevant chunks from *all* sources for a more focused synthesis, based on the original topic or sub-topics.
    *   Potentially use the vector store for internal relevance checks before scraping new URLs to avoid redundant processing of similar content.
    *   *(Defer exposing direct user Q&A based on the vector store for now)*.

*   **LCEL Parallelism:**
    *   Explore using `RunnableParallel` from LangChain Expression Language (LCEL).
    *   Identify independent tasks within the research loop that could be parallelized, such as:
        *   Navigating and extracting content from multiple URLs concurrently if identified by the planning step.
        *   Summarizing different extracted documents in parallel.
    *   This could potentially speed up the overall research process. 