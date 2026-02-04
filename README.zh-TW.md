# Paper Reading Assistant

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)
![Gemini](https://img.shields.io/badge/AI-Gemini%202.0-purple?style=flat-square&logo=google-gemini)
![Obsidian](https://img.shields.io/badge/Obsidian-Integrated-7c3aed?style=flat-square&logo=obsidian)

<div style="background: #e5e7eb; border-radius: 9999px; padding: 2px; width: fit-content; min-width: 180px; position: relative; display: flex; align-items: center;">
  <!-- Glider Background (Absolute - Right Aligned) -->
  <div style="position: absolute; right: 2px; top: 2px; bottom: 2px; width: calc(50% - 2px); background: #10b981; border-radius: 9999px; z-index: 1;"></div>
  
  <!-- Inactive Link (English) -->
  <a href="README.md" style="flex: 1; position: relative; z-index: 2; text-align: center; text-decoration: none; padding: 6px 16px; display: block;">
    <span style="color: #4b5563; font-weight: 500; font-family: system-ui, -apple-system, sans-serif; font-size: 14px;">English</span>
  </a>

  <!-- Active Text (Chinese) -->
  <div style="flex: 1; position: relative; z-index: 2; text-align: center; padding: 6px 16px;">
    <span style="color: white; font-weight: 600; font-family: system-ui, -apple-system, sans-serif; font-size: 14px; display: block;">繁體中文</span>
  </div>
</div>

一個自動化的學術論文閱讀助手，協助您發現、下載並翻譯論文，並整合至您的 Obsidian 知識庫中。

## 功能特色

*   **智慧發現**：根據設定的關鍵字與年份範圍，自動從 Semantic Scholar 搜尋相關論文。
*   **個人化排序**：利用向量相似度分析，根據您的閱讀歷史與偏好對論文進行排序。
*   **自動下載**：自動下載 Open Access 的 PDF 檔案，或嘗試透過 ArXiv 獲取。
*   **智慧分類**：依據研究領域將論文自動分類至不同資料夾（例如 Computer Vision, NLP, Time Series）。
*   **AI 輔助翻譯**：以 **Google Gemini**（透過 Python SDK）生成高品質的學術翻譯，涵蓋摘要與重點章節。
*   **Obsidian 整合**：生成每日推薦筆記，並在翻譯檔案中注入相容於 Dataview 的 Metadata (YAML frontmatter)。
*   **混合工作流**：支援每日自動批次處理，也保留手動翻譯單一檔案的彈性。

## 專案結構

```text
.
├── main.py                 # 每日自動化執行的進入點
├── config.yaml             # 設定檔 (搜尋關鍵字、路徑、翻譯模型)
├── .env                    # 環境變數 (API Keys)
├── requirements.txt        # Python 依賴套件
├── src/
│   ├── translator.py       # 翻譯核心 (整合 Gemini SDK 與 Marker PDF 解析)
│   ├── downloader.py       # PDF 下載與翻譯觸發器
│   ├── client.py           # Semantic Scholar API 客戶端
│   ├── harvester.py        # 使用者設定檔與歷史紀錄管理
│   ├── ranker.py           # 論文排序邏輯
│   └── writer.py           # Obsidian 筆記寫入器
└── scripts/                # (舊版) PowerShell 輔助腳本
```

## 安裝與設定

<details>
<summary><strong>點擊展開詳細安裝教學</strong></summary>

1.  **前置需求**
    *   Python 3.8+
    *   [Semantic Scholar API Key](https://www.semanticscholar.org/product/api)
    *   [Google Gemini API Key](https://ai.google.dev/)

2.  **配置**
    *   在專案根目錄建立 `.env` 檔案，並填入以下內容：
        ```env
        S2_API_KEY=your_semantic_scholar_api_key
        GEMINI_API_KEY=your_google_gemini_api_key
        ```
    *   確認 `config.yaml` 符合您的需求：
        *   `obsidian.vault_path`: 您的 Obsidian Vault 路徑。
        *   `search.keywords`: 欲關注的研究關鍵字列表。
        *   `translation.model`: 設定使用的 Gemini 模型 (如 `gemini-2.0-flash-exp`)。

</details>

## 使用方法

執行主程式以獲取推薦、下載 PDF 並生成翻譯筆記：

```bash
python main.py
```

程式將會執行以下步驟：
1.  從 Semantic Scholar 獲取最新論文。
2.  根據您的設定檔進行過濾與排序。
3.  在 Obsidian 的每日資料夾中建立一份摘要筆記。
4.  將 PDF 下載至對應的分類資料夾（例如 `Papers/Time Series`）。
5.  呼叫 Python 翻譯核心，生成帶有 Dataview Metadata 的 `.zh.md` 翻譯檔案。

### 手動翻譯

如果您只想翻譯單一 PDF 檔案（例如您自己下載的論文），可以直接呼叫翻譯核心：

```bash
python src/translator.py "c:/path/to/your/paper.pdf"
```

*   程式會自動讀取 `.env` 中的 API Key 與 `config.yaml` 中的設定。
*   翻譯結果 (`.zh.md`) 將會生成在與 PDF 相同的資料夾中。
