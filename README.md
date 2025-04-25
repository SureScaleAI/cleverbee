# Python AI Research Agent

A Python-based research agent using Large Language Models (LLMs) like Claude and Gemini, Playwright for web browsing, and Chainlit for an interactive UI. It performs research by browsing the web, extracting content (HTML), cleaning it, and summarizing findings based on user research topics.

## Features

-   Interactive web UI via Chainlit.
-   **MCP Tool Support:** Integrates external tools via the Model Context Protocol (MCP), defined in `mcp.json`. See [MCP Tool Integration](#mcp-tool-integration).
-   **Multi-LLM Research:** Uses distinct, configurable LLMs for different tasks. Defaults are:
    -   **Primary LLM:** `Gemini 2.5 Pro` for planning and final report generation.
    -   **Next Step LLM:** `Gemini 2.5 Flash` for analyzing research progress and deciding next actions.
    -   **Summarizer LLM:** `Gemini 2.0 Flash` for intermediate web content summarization (`SUMMARIZER_MODEL` in `config.yaml`).
    *(These can be configured in `config.yaml`, supporting Gemini, Claude, and local GGUF models)*.
-   **Automated Web Browsing:** Utilizes Playwright for searching the web and extracting HTML content.
-   **Content Processing:** Cleans HTML to Markdown before summarization.
-   **Integrated Token Tracking:** Monitors token usage and estimates costs for LLM calls, displayed per step and summarized in the UI. Includes cache monitoring for savings.
-   **Highly Configurable:** Settings managed via `config.yaml` (models, tool enablement, memory limits, API keys, logging).
-   **Modular LLM Clients:** Supports different providers (Gemini, Claude, Local GGUF via llama-cpp-python).
-   **Hardware-Aware Setup:** `setup.sh` assists in configuring local models (primarily NVIDIA GPU focused) and API keys.
-   **LLM Caching:** Employs `NormalizingCache` (SQLite-based) for improved performance and cost reduction.
-   **Standard Tools:** Includes web browsing and extraction (`web_browser`), plus Reddit search and post extraction (`reddit_search`, `reddit_extract_post`) capabilities.

## System Compatibility

-   **macOS:** Fully tested and supported, including both Intel and Apple Silicon (via Rosetta 2).
-   **Linux:** Fully supported, with NVIDIA GPU detection and optimization for local models.
-   **Windows:** Limited support via Windows Subsystem for Linux (WSL). A Windows-specific setup script (`setup_windows.ps1`) is available but is in beta and may require further refinement.

## Prerequisites

-   Python 3.x
-   `jq` (JSON processor)
-   `yq` (YAML processor) - `setup.sh` will attempt to install this if missing.
-   `git`
-   Node.js and `npx` (for rebrowser patches during setup). `setup.sh` can install Node.js via `nvm` if needed.
-   (Optional, for Local Models) CMake and a compatible C/C++ compiler.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-directory>
    ```
2.  **Run the setup script:**
    
    For macOS/Linux:
    ```bash
    bash setup.sh
    ```
    
    For Windows (beta/untested):
    ```powershell
    # Run as Administrator
    .\setup_windows.ps1
    ```
    
    This script guides you through:
    -   Checking prerequisites.
    -   Creating a Python virtual environment (`venv`).
    -   Installing base Python dependencies from `requirements.txt`.
    -   **Hardware Check & Model Preference:** Asks if you want to enable local model setup (currently optimized for NVIDIA GPUs >= 24GB VRAM).
    -   **Conditional `llama-cpp-python` Install:** Installs `llama-cpp-python` with appropriate hardware acceleration (CUDA for NVIDIA, Metal for Apple Silicon) if local models are enabled.
    -   **Hugging Face Login:** Prompts login if using local models.
    -   **Model Configuration:** Helps select primary reasoning and summarizer models, updating `config.yaml`.
    -   **Model Download:** Downloads selected local GGUF models (e.g., Llama 3.3 70B) if chosen.
    -   **GPU Layer Optimization:** Recommends and sets `N_GPU_LAYERS` in `config.yaml` for local models.
    -   **Additional Config:** Helps set memory limits and thinking mode in `config.yaml`.
    -   **Playwright Setup:** Installs required browsers and applies `rebrowser-patches`.
    -   **API Key Setup:** Checks for needed API keys (`ANTHROPIC_API_KEY`, `GEMINI_API_KEY`) in `.env` based on selected models and prompts if missing. Creates `.env` from `.env.example` if none exists.
    -   Making `run.sh` executable.

3.  **Verify API Keys:**
    -   Ensure the `.env` file contains the necessary API keys for your chosen cloud models (e.g., `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`). The setup script helps with this.

## Configuration

All major configuration is now handled in `config.yaml`.

-   **`PRIMARY_MODEL_TYPE`**: `"gemini"`, `"claude"`, or `"local"`. Determines the main reasoning LLM.
-   **`GEMINI_MODEL_NAME` / `CLAUDE_MODEL_NAME`**: Specific model IDs for cloud providers.
-   **`LOCAL_MODEL_NAME` / `LOCAL_MODEL_QUANT_LEVEL`**: GGUF filename and quantization (e.g., `Q4_K_M`) for local primary model.
-   **`USE_LOCAL_SUMMARIZER_MODEL`**: `true` or `false`. Set by `setup.sh`.
-   **`SUMMARIZER_MODEL`**: Model ID for summarization tasks (e.g., `"gemini-2.0-flash"` or a local GGUF filename).
-   **`LOCAL_MODELS_DIR`**: Path to local GGUF files.
-   **`N_GPU_LAYERS`**: GPU offloading layers for local models.
-   **`SUMMARY_MAX_TOKENS` / `FINAL_SUMMARY_MAX_TOKENS`**: Token limits for summarization.
-   **`CHUNK_SIZE` / `CHUNK_OVERLAP`**: Text splitting parameters (often set by `setup.sh`).
-   **`ENABLE_THINKING`**: Boolean to enable extended reasoning steps.
-   **`CONVERSATION_MEMORY_MAX_TOKENS`**: Context window limit for agent memory.
-   **Web Search / Browser Settings**: Limits (`MIN/MAX_REGULAR_WEB_PAGES`), timeouts.
-   **`tools` section**: Enables/disables specific agent tools (see below).
-   **Token Tracking & Pricing**: Enable tracking and set costs per model.
-   **`LOG_LEVEL`**: Logging verbosity (`DEBUG`, `INFO`, etc.).

### Tool Configuration (`config.yaml`)

The `tools` section in `config.yaml` controls which tools are available to the agent:

```yaml
# config.yaml
# ... other settings ...

tools:
  web_browser:
    enabled: true # Enables the core web browsing/extraction tool (Playwright based).
  reddit_search:
    enabled: true # Enables the tool specifically designed for searching Reddit.
  reddit_extract_post:
    enabled: true # Enables the tool for extracting content from a specific Reddit post URL.
  # Add other standard or MCP-backed tools here as needed
  # Example for an MCP tool (assuming 'arxiv' is defined in mcp.json):
  # search_arxiv_papers:
  #   enabled: true
  #   mcp_tool_name: get_arxiv_papers # The *actual* tool name exposed by the MCP server
  #   # Optional overrides for description, etc.
  #   # description: "Search arXiv for scientific papers using keywords."

# ... rest of config ...
```

-   **`enabled`**: (boolean) Toggles the tool's availability.
-   For MCP tools, you might add an `mcp_tool_name` if the key in the `tools` section differs from the actual callable tool name provided by the MCP server (defined via its schema). You can also add `description` overrides here.

### MCP Server Configuration (`mcp.json`)

Defines how to run external MCP servers that provide specialized tools. This file is located in the project root.

```json
// mcp.json
{
  "mcpServers": {
    "youtube-transcript": {
      "command": "npx",
      "args": [
        "-y",
        "@sinco-lab/mcp-youtube-transcript"
      ],
      "minToolCalls": 0,  // Optional: Minimum times a tool from this server *must* be called per agent run (0=optional)
      "maxToolCalls": 3,  // Optional: Maximum times tools from this server *can* be called per agent run
      "description": "YouTube video transcripts.", // Optional: Description of the server's purpose
      "timeout": 20.0 // Optional: Client-side timeout (seconds)
    },
    // ... other servers ...
  }
}
```

-   **`command`**: (array of strings) Command to start the server process.
-   **`args`**: (array of strings, optional) Arguments for the command.
-   **`env`**: (object, optional) Environment variables for the server process.
-   **`timeout`**: (float, optional) Client-side request timeout for this server.
-   **`minToolCalls`**: (integer, optional) Specifies the minimum number of times any tool provided by this MCP server *must* be called during a single agent execution cycle. Defaults to 0 if omitted, meaning calls are optional.
-   **`maxToolCalls`**: (integer, optional) Specifies the maximum number of times any tool provided by this MCP server *can* be called during a single agent execution cycle. No limit if omitted.
-   **`description`**: (string, optional) A brief description of the MCP server's capabilities.

## MCP Tool Integration

-   The framework supports Model Context Protocol (MCP) tools via a standardized protocol. MCP allows extending agent capabilities with specialized external services.
-   MCP client wrapper (`src/mcp/client.py`) handles tool discovery, schema validation, and execution.
-   Available MCP servers and how to run them are defined in `mcp.json` (in the project root).
-   Tools provided by MCP servers can be enabled/disabled in the `tools` section of `config.yaml`.
-   The agent discovers tools from enabled MCP servers at startup.
-   **Default MCP Servers:** The `mcp.json` includes configurations for example PubMed and YouTube transcript servers by default.
    -   PubMed: [https://github.com/grll/pubmedmcp](https://github.com/grll/pubmedmcp)
    -   YouTube: [https://github.com/sinco-lab/mcp-youtube-transcript](https://github.com/sinco-lab/mcp-youtube-transcript)
    -   **Important:** If you do not intend to use these, remove their entries from `mcp.json`.

## LLM Integration

-   Supports multiple LLM providers (Claude, Gemini, Local GGUF) through a unified factory (`src/llm_clients/factory.py`).
-   Configurable model selection for different roles:
    -   **Primary LLM:** (`PRIMARY_MODEL_TYPE`, `*_MODEL_NAME`, etc. in `config.yaml`) for planning and final reporting.
    -   **Next Step LLM:** (Configurable, potentially sharing settings with Primary or having its own keys like `NEXT_STEP_MODEL_*`) for deciding subsequent actions during research.
    -   **Summarizer LLM:** (`SUMMARIZER_MODEL` in `config.yaml`) for content summarization.
-   Token usage tracking and cost estimation for configured models.
-   Optional enhanced reasoning (`ENABLE_THINKING: true`) for potentially better complex task handling (may depend on the LLM).

## Caching and Performance

-   Uses `NormalizingCache` (SQLite-based, stored in `.langchain.db`) to cache LLM responses, improving speed and reducing costs.
-   Cache monitoring tracks savings from hits.
-   Use `--no-cache` argument with `run.sh` to clear the cache before starting.
-   **Note:** LangChain caching mechanisms can sometimes be buggy or produce unexpected results. Clearing the cache (`.langchain.db`) might be necessary if you encounter issues.

## License

This project is licensed under the **GNU Affero General Public License, Version 3.0**. See the [LICENSE](LICENSE) file for the full text.

By contributing to this project, you agree that your contributions will be licensed under its AGPLv3 license, and you grant a copyright license to your contributions according to the terms of the [Contributor License Agreement (CLA)](CLA.md).
