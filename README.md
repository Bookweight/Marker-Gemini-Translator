# Paper Reading Assistant

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)
![Gemini](https://img.shields.io/badge/AI-Gemini%202.0-purple?style=flat-square&logo=google-gemini)
![Obsidian](https://img.shields.io/badge/Obsidian-Integrated-7c3aed?style=flat-square&logo=obsidian)

<div style="background: #e5e7eb; border-radius: 9999px; padding: 2px; width: fit-content; min-width: 180px; position: relative; display: flex; align-items: center;">
  <!-- Glider Background (Absolute) -->
  <div style="position: absolute; left: 2px; top: 2px; bottom: 2px; width: calc(50% - 2px); background: #10b981; border-radius: 9999px; z-index: 1;"></div>
  
  <!-- Active Text (English) -->
  <div style="flex: 1; position: relative; z-index: 2; text-align: center; padding: 6px 16px;">
    <span style="color: white; font-weight: 600; font-family: system-ui, -apple-system, sans-serif; font-size: 14px; display: block;">English</span>
  </div>
  
  <!-- Inactive Link (Chinese) -->
  <a href="README.zh-TW.md" style="flex: 1; position: relative; z-index: 2; text-align: center; text-decoration: none; padding: 6px 16px; display: block;">
    <span style="color: #4b5563; font-weight: 500; font-family: system-ui, -apple-system, sans-serif; font-size: 14px;">繁體中文</span>
  </a>
</div>

An automated tool to discover, download, and translate academic papers for your Obsidian knowledge base.

## Features

*   **Smart Discovery**: Searches Semantic Scholar for papers based on configured keywords and year ranges.
*   **Personalized Ranking**: Ranks papers using vector similarity, **Recency Bonus** (prioritizing fresh research), and **Venue Weighting** (boosting top-tier sources).
*   **Automated Downloading**: Automatically downloads open-access PDFs or attempts retrieval via ArXiv.
*   **Intelligent Organization**: Categorizes papers into folders (e.g., Computer Vision, NLP, Time Series) based on their fields of study.
*   **AI-Powered Translation**: Uses **Google Gemini** to generate high-quality academic translations.
*   **Clean Output**: Automatically removes PDF artifacts (e.g., page numbers, internal links) for a smooth reading experience.
*   **Obsidian Integration**: Generates daily recommendation notes and injects Dataview-compatible metadata.
*   **Hybrid Workflow**: Supports both automated daily batch processing and manual single-file translation.

## Project Structure

```text
.
├── main.py                 # Entry point for daily automation
├── config.yaml             # Configuration file (search terms, paths, translation settings)
├── .env                    # Secrets (API Keys)
├── requirements.txt        # Python dependencies
├── src/
│   ├── translator.py       # Core Translation Logic (Integrates Gemini SDK & Marker)
│   ├── downloader.py       # PDF downloader and translation trigger
│   ├── client.py           # Semantic Scholar API client
│   ├── ranker.py           # Ranking, User Profiling, and Feedback Harvesting
│   └── writer.py           # Obsidian note writer
└── scripts/                # (Legacy) PowerShell helper scripts
```

## Setup

<details>
<summary><strong>Click to expand installation interactions</strong></summary>

1.  **Prerequisites**
    *   Python 3.8+
    *   [Semantic Scholar API Key](https://www.semanticscholar.org/product/api)
    *   [Google Gemini API Key](https://ai.google.dev/)

2.  **Configuration**
    *   Create a `.env` file in the root directory:
        ```env
        S2_API_KEY=your_semantic_scholar_api_key
        GEMINI_API_KEY=your_google_gemini_api_key
        ```
    *   Ensure `config.yaml` is set up with your preferences:
        *   `obsidian.vault_path`: Path to your Obsidian vault.
        *   `search.keywords`: List of topics to search for.
        *   `translation.model`: Gemini Model to use (e.g., `gemini-2.0-flash-exp`).

</details>

## Usage

Run the main script to fetch recommendations, download PDFs, and generate translation notes:

```bash
python main.py
```

The script will:
1.  Fetch recent papers from Semantic Scholar.
2.  Filter and rank them based on your profile.
3.  Create a daily summary note in your Obsidian daily folder.
4.  Download the PDFs to categorized folders (e.g., `Papers/Time Series`).
5.  Invoke the Python translation engine to generate `.zh.md` files with Dataview metadata.

### Manual Translation

If you want to translate a single PDF file (e.g., a paper you downloaded manually), you can invoke the translator directly:

```bash
python src/translator.py "c:/path/to/your/paper.pdf"
```

*   The script will load API Keys from `.env` and settings from `config.yaml`.
*   The output file (`.zh.md`) will be generated in the same directory as the PDF.
