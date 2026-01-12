param(
    [Parameter(Mandatory=$true)]
    [string]$InputFile
)

# --- Encoding Setup ---
$OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::InputEncoding = [System.Text.Encoding]::UTF8
$GeminiCommand = "gemini"

# --- File Path Validation and Setup ---
if (-not (Test-Path $InputFile -PathType Leaf)) {
    Write-Host "Error: File '$InputFile' does not exist." -ForegroundColor Red
    exit 1
}
$FullPath = (Get-Item $InputFile).FullName
$PSScriptRoot = (Get-Item -Path $PSScriptRoot).FullName
$OutputFile = $FullPath -replace '\.[^.]+$', '.zh.md'
$GlossaryOutputFile = $PSScriptRoot + '\research_topics.md'

Write-Host "---"
Write-Host "✅ Input File: $FullPath" -ForegroundColor Green
Write-Host "➡️ Translation Output: $OutputFile" -ForegroundColor Yellow
Write-Host "➡️ Glossary Output: $GlossaryOutputFile" -ForegroundColor Yellow
Write-Host "---"

# --- Prompt Definitions ---

# 1. Metadata Prompt
$MetadataPrompt = "From the attached PDF document, please identify and extract the following two pieces of information:
1. The name of the conference or journal where the paper was published.
2. The year of publication.

Format the output as a single line of Markdown:
**Published in:** [Conference/Journal Name], [Year]

If you cannot find the information, please output the line:
**Published in:** Information not found"

# 2. Translation Prompt
$TranslationPrompt = "Please translate the entire attached PDF document from beginning to end into Traditional Chinese (繁體中文).
Your task is to create a bilingual, academic-focused Markdown document. Strictly adhere to the following requirements:
1.  **Output Format (Paragraph Style):** For each paragraph, heading, or list item, present the original English text first. Immediately following the English text, on a new line, present its corresponding academic-style Traditional Chinese translation. Separate the English and Chinese blocks with a blank line for readability.
2.  **Translation Quality (Academic Focus):** The translation must be of high academic quality, suitable for a research paper. Use precise, formal, and domain-specific terminology.
3.  **Content Preservation:** Maintain the original document's structure. Represent headings using Markdown syntax (e.g., `## 1. Introduction`). DO NOT translate mathematical formulas, code snippets, proper nouns, and English acronyms (e.g., `PIR`, `VLDB`).
4.  **Formatting Rules:** Do not use `<br>` tags. Use standard Markdown paragraphs.
5.  **Strictly Traditional Chinese:** The entire Chinese output MUST be in Traditional Chinese (繁體中文 - `zh-TW`). Do not mix Simplified Chinese characters.

Process the entire document in a single, continuous operation."

# 3. Glossary Prompt (Dynamic Domain)
$GlossaryPrompt = "Your task is to act as a research assistant and create a glossary of key terms from the attached academic paper. Please follow these steps:
1.  **First, analyze the paper's abstract, introduction, and keywords to determine its specific domain(s)** (e.g., `data mining`, `computer vision`, `database systems`).
2.  **Then, based on the domain(s) you just identified, read through the entire document and extract the most important technical terms, specialized nouns, and key concepts.**

For each term you extract, provide the following in a Markdown list format, using Traditional Chinese:

- **[術語 (繁體中文)] ([English Term])**: [A concise, one-to-two sentence explanation in Traditional Chinese].

Format the output under the following heading:
---
## 核心術語與技術名詞 (Core Terms and Technologies)

Ensure the output is entirely in Traditional Chinese (繁體中文)."


# --- Execution Block ---

# Step 1: Metadata Extraction
Write-Host "--- Step 1: Extracting Publication Info..." -ForegroundColor Green
$metadata_result = & $GeminiCommand -p "$MetadataPrompt @$FullPath" 2>&1
if ($metadata_result -is [string] -and ($metadata_result -match "Error" -or $metadata_result -match "Usage:")) {
    Write-Host "⚠️ Metadata extraction failed. Proceeding without it." -ForegroundColor Yellow
    }
Write-Host "--- Waiting for 10 seconds to avoid API rate limits..." -ForegroundColor DarkGray
Start-Sleep -Seconds 10

# Step 2: Full Translation
Write-Host "--- Step 2: Translating Full Text..." -ForegroundColor Green
$translation_result = & $GeminiCommand -p "$TranslationPrompt @$FullPath" 2>&1
if ($translation_result -is [string] -and ($translation_result -match "Error" -or $translation_result -match "Usage:")) {
    Write-Host "❌ Critical Error: Translation failed. Aborting script." -ForegroundColor Red
    Write-Host "$translation_result" -ForegroundColor Red
    exit 1
}

Write-Host "--- Waiting for 10 seconds to avoid API rate limits..." -ForegroundColor DarkGray
Start-Sleep -Seconds 10

# --- Step 3: Generating Glossary ---
Write-Host "--- Step 3: Generating Glossary..." -ForegroundColor Green
$glossary_result = & $GeminiCommand -p "$GlossaryPrompt @$FullPath" 2>&1
if ($glossary_result -is [string] -and ($glossary_result -match "Error" -or $glossary_result -match "Usage:")) {
    Write-Host "⚠️ Glossary generation failed. The translation file is still complete." -ForegroundColor Yellow
    $glossary_result = "## 核心術語與技術名詞 (Core Terms and Technologies)`nGlossary generation failed due to an API error."
}

# --- File Writing Block ---
Write-Host "--- Finalizing Files..." -ForegroundColor Green

# Write Translation File (Metadata + Content)
$final_translation_content = $metadata_result + "`n`n" + $translation_result
[System.IO.File]::WriteAllLines($OutputFile, $final_translation_content, [System.Text.Encoding]::UTF8)
Write-Host "🎉 Translation successful! File saved to: $OutputFile" -ForegroundColor Cyan

# Write Glossary File
[System.IO.File]::WriteAllLines($GlossaryOutputFile, $glossary_result, [System.Text.Encoding]::UTF8)
Write-Host "✨ Glossary successful! File saved to: $GlossaryOutputFile" -ForegroundColor Cyan

Write-Host "---"
Write-Host "✅ All processing complete!" -ForegroundColor Green
Write-Host "---"
