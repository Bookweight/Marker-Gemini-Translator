param(
    [Parameter(Mandatory=$true)]
    [string]$InputFile
)

# --- 環境與編碼設定 ---
$OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::InputEncoding = [System.Text.Encoding]::UTF8
$GeminiCommand = "gemini"
$Utf8NoBom = New-Object System.Text.UTF8Encoding($false)

# 1. 檢查輸入檔案
if (-not (Test-Path $InputFile -PathType Leaf)) {
    Write-Host "Error: File '$InputFile' does not exist." -ForegroundColor Red
    exit 1
}

$FullPath = (Get-Item $InputFile).FullName
$OutputFile = $FullPath -replace '\.[^.]+$', '.zh.md'

Write-Host "---"
Write-Host "✅ Processing file: $FullPath" -ForegroundColor Green
Write-Host "➡️ Target: $OutputFile" -ForegroundColor Yellow

# --- 2. 暫存檔案處理 ---
# 為了避免檔名過長或特殊字元問題，暫存檔使用簡單的隨機名稱
$TempFileName = "temp_processing_$(Get-Random).pdf"
$TempFilePath = Join-Path $PSScriptRoot $TempFileName

try {
    Write-Host "📂 Copying to local workspace: $TempFilePath" -ForegroundColor DarkGray
    Copy-Item -Path $FullPath -Destination $TempFilePath -Force
}
catch {
    Write-Host "❌ Failed to create temp file. Error: $_" -ForegroundColor Red
    exit 1
}
Push-Location $PSScriptRoot
try {
    $TranslationPrompt = "You are an expert Academic Translator and Document Formatter specializing in Computer Science papers. Your target format is Obsidian-compatible Markdown.
**TOOL USE POLICY (CRITICAL):**
1. **ALLOWED (INPUT):** You **MUST** use your internal PDF parsing/reading capabilities to consume the attached document. This is required to perform the task.
2. **FORBIDDEN (OUTPUT):** You are **STRICTLY PROHIBITED** from using ""Action Tools"" to save results. Do NOT attempt to use:
   - 'write_file'
   - 'write_todos'
   - 'python_interpreter'
   - 'create_file'
**TASK:**
Translate the attached PDF content into Traditional Chinese (繁體中文) while strictly preserving the original English text for comparison.

**CRITICAL RULES:**

0.**HEADINGS & SECTION TITLES (STRICT FORMATTING):**

    **Main Sections (Level 1 & 2):**
    * **Target:** Standard academic headers (Abstract, Introduction, Related Work, Method, Experiments, Conclusion, References) and Top-Level Numbered Sections (1., 2., 3.).
    * **Format:** Keep the English title, add a hyphen, and append the Chinese translation on the **SAME LINE**.
    * **Template:** `## Original English - Chinese Translation`
    * *Example:* `## Abstract - 摘要`
    * *Example:* `## 1. Introduction - 緒論`

    **Sub-Sections & Descriptive Titles (Level 3+):**
    * **Target:** Nested sections (1.1, 2.3.1) or specific descriptive sub-titles (e.g., ""3.1 Motivation..."").
    * **Format:** Output **ONLY** the Traditional Chinese translation. Keep the numbering if present.
    * **Template:** `### Number Chinese_Translation`
    * *Example:* Input ""3 Motivation: Modeling Long-Range..."" -> Output `### 3 動機：建模長距離...`

1.  **NOISE REMOVAL (Pre-processing):**
    * IGNORE page headers, footers, page numbers (e.g., ""1"", ""arXiv:..."", and running titles. Do not translate or output them.
    * Merge sentences that are broken across lines or pages into a single coherent paragraph before translating.

2.  **TEXT CONTENT (Bilingual Layout):**
    * Processing logic: Read one logical paragraph -> Output **Original English** -> Insert **Empty Line** -> Output **Traditional Chinese Translation** -> Insert **Empty Line**.
    * **Translation Style:** Academic, formal, and precise. Use standard Taiwan academic terminology (e.g., ""Transformer"" -> ""Transformer"", ""optimization"" -> ""mization"" -> ""最佳化"").
    * **Exception:** Do **NOT** translate the **References** section. Output it strictly in original English.

3.  **STRICT TABLE RULES (HTML FORMAT):**
    * **Format:** You MUST reconstruct all tables using **HTML Syntax** (`<table>`, `<thead>`, `<tbody>`, `<tr>`, `<th>`, `<td>`). **DO NOT** use Markdown pipe tables (`|`).
    * **Structure:** Use `colspan=""X""` and `rowspan=""Y""` attributes to accurately represent merged headers or merged row labels. This is critical for preserving the logical hierarchy of complex academic tables.
    * **Reconstruction:** The input text for tables might be fragmented. You must infer the logical columns and rows based on the semantic content and alignment.
    * **English Only:** The content inside the table (headers, numbers, legends) must remain **100% in Original English**. **DO NOT translate** any text inside the `<table>` tags.
    * **Styling:** Keep it semantic. Use `<th>` for headers and `<td>` for data. Do not add complex inline CSS styles.

4.  **MATH & FORMULAS:**
    * Keep all LaTeX formulas intact (e.g., `$x_i$`, `$$\sum$$`). Do not translate variables.
    * Ensure inline math uses single `$` and block math uses `$$`.

**EXECUTION:**
Start translating from the Title and Abstract down to the Conclusion. Then append the untranslated References."
    $CommandString = "$TranslationPrompt" + " @" + "$TempFileName"

    Write-Host "🚀 Gemini is translating... Please wait." -ForegroundColor Cyan

    # 執行並捕捉所有輸出
    $result = & $GeminiCommand "$CommandString" 2>&1
    $resultString = $result | Out-String

    # --- 4. 智慧結果判斷 ---
    $IsSuccess = $resultString -match "(?m)^#\s" -or $resultString -match "\[Original English Text\]"

    if ($IsSuccess) {
        # --- 資料清洗 ---
        if ($resultString -match "(?ms)(.*?)(^(#|___|\*\*\*).*$)") {
            $CleanContent = $matches[2]
        } else {
            $CleanContent = $resultString
        }

        $CleanContent = $CleanContent -replace '(?m)^\s*\[?Original English Text\]?:?\s*', ''
        $CleanContent = $CleanContent -replace '(?m)^\s*\[?Traditional Chinese Translation\]?:?\s*', ''
        
        [System.IO.File]::WriteAllText($OutputFile, $CleanContent, $Utf8NoBom)
        Write-Host "🎉 Translation successful!" -ForegroundColor Cyan

        # --- Metadata 注入 ---
        $TargetFile = Get-Item $OutputFile
        $FullContent = [System.IO.File]::ReadAllText($TargetFile.FullName, $Utf8NoBom)
        
        if (-not $FullContent.StartsWith("---")) {
            $Field = $TargetFile.Directory.Parent.Name
            $CleanTitle = $TargetFile.BaseName -replace "\.zh$", ""
            $CurrentDate = Get-Date -Format "yyyy-MM-dd"
            
            $Yaml = "---`ntitle: `"$CleanTitle`"`nfield: `"$Field`"`nstatus: `"Imported`"`ncreated_date: $CurrentDate`npdf_link: `"[[$( $CleanTitle ).pdf]]`"`ntags: [paper, $Field]`n---`n`n"
            
            [System.IO.File]::WriteAllText($TargetFile.FullName, $Yaml + $FullContent, $Utf8NoBom)
            Write-Host "✅ Metadata injected successfully." -ForegroundColor Green
        }
    } else {
        Write-Host "❌ Translation failed." -ForegroundColor Red
        Write-Host "Gemini Output:`n$resultString"
        exit 1
    }
}
finally {
    # --- 5. 清理暫存檔 ---
    if (Test-Path $TempFilePath) { 
        Remove-Item $TempFilePath -Force 
        Write-Host "🧹 Temp file cleaned." -ForegroundColor DarkGray
    }
    Pop-Location
}

Write-Host "--- Operation Complete ---" -ForegroundColor Cyan
