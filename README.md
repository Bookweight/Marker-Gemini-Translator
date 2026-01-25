# Paper Reading Assistant

An automated tool to discover, download, and translate academic papers for your Obsidian knowledge base.

## Features

*   **Smart Discovery**: Searches Semantic Scholar for papers based on configured keywords and year ranges.
*   **Personalized Ranking**: Ranks papers based on your reading history and preferences using vector similarity.
*   **Automated Downloading**: Automatically downloads open-access PDFs or attempts retrieval via ArXiv.
*   **Intelligent Organization**: Categorizes papers into folders (e.g., Computer Vision, NLP, Time Series) based on their fields of study.
*   **AI-Powered Translation**: Uses **Google Gemini** to generate high-quality, academic translations of the abstract and key sections.
*   **Obsidian Integration**: Generates daily recommendation notes and injects Dataview-compatible metadata (YAML frontmatter) into translated files.
*   **Hybrid Workflow**: Supports both automated daily batch processing and manual single-file translation.

## Project Structure

```text
.
├── main.py                 # Entry point for daily automation
├── config.yaml             # Configuration file (search terms, paths)
├── .env                    # Secrets (API Keys)
├── requirements.txt        # Python dependencies
├── src/
│   ├── client.py           # Semantic Scholar API client
│   ├── downloader.py       # PDF downloader and translation trigger
│   ├── harvester.py        # User profile and history manager
│   ├── prompt_engine.py    # Generates prompts for translation
│   ├── ranker.py           # Paper ranking logic
│   └── writer.py           # Obsidian note writer
└── scripts/
    └── Translate-And-Metadata.ps1  # PowerShell script for PDF processing & metadata injection
```

## Setup

1.  **Prerequisites**
    *   Python 3.8+
    *   PowerShell (required for the translation script execution)
    *   [Semantic Scholar API Key](https://www.semanticscholar.org/product/api)

2.  **Configuration**
    *   Create a `.env` file in the root directory:
        ```env
        S2_API_KEY=your_semantic_scholar_api_key
        ```
    *   Ensure `config.yaml` is set up with your preferences:
        *   `obsidian.vault_path`: Path to your Obsidian vault.
        *   `search.keywords`: List of topics to search for.
        *   `search.year_range`: Year range for papers.

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
5.  Trigger the translation script to generate `.zh.md` files with Dataview metadata.
