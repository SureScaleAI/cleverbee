# CleverBee: Advanced AI Research Assistant

<p align="center">
  <a href="https://cleverb.ee">
    <img src="public/logo_dark.svg" alt="CleverBee Logo" width="300">
  </a>
</p>

<p align="center">
  <a href="https://cleverb.ee">Website</a> •
  <a href="https://cleverb.ee">Documentation</a> •
  <a href="https://github.com/SureScaleAI/cleverbee">GitHub</a>
</p>

---

CleverBee is a powerful Python-based research agent using Large Language Models (LLMs) like Claude and Gemini, Playwright for web browsing, and Chainlit for an interactive UI. It performs research by browsing the web, extracting content (HTML), cleaning it, and summarizing findings based on user research topics.

## ✨ Features

-   🌐 **Interactive web UI** via Chainlit
-   🔧 **MCP Tool Support:** Integrates external tools via the Model Context Protocol (MCP)
-   🧠 **Multi-LLM Research:** Uses distinct, configurable LLMs for different tasks:
    -   **Primary LLM:** `Gemini 2.5 Pro` for planning and final report generation
    -   **Next Step LLM:** `Gemini 2.5 Flash` for analyzing research progress and deciding next actions
    -   **Summarizer LLM:** `Gemini 2.0 Flash` for intermediate web content summarization
-   🌍 **Automated Web Browsing:** Utilizes Playwright for searching the web and extracting HTML content
-   📊 **Content Processing:** Cleans HTML to Markdown before summarization
-   📈 **Integrated Token Tracking:** Monitors token usage and estimates costs for LLM calls
-   ⚙️ **Highly Configurable:** Settings managed via `config.yaml`
-   🚀 **Modular LLM Clients:** Supports different providers (Gemini, Claude, Local GGUF via llama-cpp-python)
-   💾 **LLM Caching:** Employs `NormalizingCache` (SQLite-based) for improved performance and cost reduction

## 🖥️ System Compatibility

-   **macOS:** Fully tested and supported, including both Intel and Apple Silicon (via Rosetta 2)
-   **Linux:** Fully supported, with NVIDIA GPU detection and optimization for local models
-   **Windows:** Limited support via Windows Subsystem for Linux (WSL)

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/SureScaleAI/cleverbee.git
cd cleverbee

# Run the setup script
bash setup.sh

# Start the application
bash run.sh
```

## 📚 Documentation

For full documentation, visit our website: [https://cleverb.ee](https://cleverb.ee)

## 📝 Configuration

All major configuration is handled in `config.yaml`. See the [documentation](https://cleverb.ee) for detailed configuration options.

## 📄 License

This project is licensed under the **GNU Affero General Public License, Version 3.0**. See the [LICENSE](LICENSE) file for the full text.

By contributing to this project, you agree that your contributions will be licensed under its AGPLv3 license, and you grant a copyright license to your contributions according to the terms of the [Contributor License Agreement (CLA)](CLA.md).
