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

if (-not (Test-Path $InputFile -PathType Leaf)) {
    Write-Host "Error: File '$InputFile' does not exist." -ForegroundColor Red
    exit 1
}

$FullPath = (Get-Item $InputFile).FullName
$OutputFile = $FullPath -replace '\.[^.]+$', '.zh.md'

Write-Host "---"
Write-Host "✅ Processing file: $FullPath" -ForegroundColor Green
Write-Host "➡️ Target: $OutputFile" -ForegroundColor Yellow

# --- 強化後的 Prompt (確保逐段對照格式) ---
$TranslationPrompt = "Please translate the attached PDF document into Traditional Chinese (繁體中文). 

Strictly follow this paragraph structure:
1. **Paragraph-by-Paragraph Bilingual Format:** For every single paragraph or section, output the [Original English Text] first, followed immediately by its [Traditional Chinese Translation]. Do not group all English together.
2. **Maintain Markdown Structure:** Keep all original headings (# ##), list formats, and section numbering.
3. **No Translation for Technical Terms:** Keep formulas, acronyms (e.g., GNN, CNN), and specialized terminology in English.
4. **Academic Tone:** Ensure the translation is professional and rigorous.
5. **Continuous Output:** Translate the entire document from start to finish.

Output format must be Markdown."

$CommandString = "$TranslationPrompt" + " @" + "$FullPath"

# 執行翻譯
Write-Host "🚀 Gemini is translating... Please wait." -ForegroundColor Cyan
$result = & $GeminiCommand -p "$CommandString" 2>&1 

if ($result -is [string] -and ($result -match "Error" -or $result -match "Usage:")) {
    Write-Host "❌ Translation failed." -ForegroundColor Red
    Write-Host "$result"
    exit 1
} else {
    # 修正：確保結果以字串形式連接，避免 PowerShell 分行處理
    $RawText = [string]::Join("`r`n", $result)
    
    # --- 【新增：標籤移除邏輯】 ---
    # 使用正則表達式精準移除標籤，同時處理可能有/無括號或冒號的情況
    # 此動作會移除標籤文字，但保留原始的換行結構
    $CleanText = $RawText -replace '(?m)^\s*\[?Original English Text\]?:?\s*', ''
    $CleanText = $CleanText -replace '(?m)^\s*\[?Traditional Chinese Translation\]?:?\s*', ''
    
    # 寫入清洗後的翻譯內容
    [System.IO.File]::WriteAllText($OutputFile, $CleanText, $Utf8NoBom)
    Write-Host "🎉 Translation successful!" -ForegroundColor Cyan
}

# --- Metadata 注入 (修正變數引用與邏輯) ---
try {
    $TargetFile = Get-Item $OutputFile
    $FullContent = [System.IO.File]::ReadAllText($TargetFile.FullName, $Utf8NoBom)
    
    if ($FullContent.StartsWith("---")) {
        Write-Host "⏩ Skip: Already has Metadata." -ForegroundColor Yellow
    } else {
        # 解析路徑資訊
        $PaperFolderName = $TargetFile.Directory.Name
        $Field = $TargetFile.Directory.Parent.Name
        $CleanTitle = $TargetFile.BaseName -replace "\.zh$", ""
        $CurrentDate = Get-Date -Format "yyyy-MM-dd"

        # 建立符合舊版規範的 YAML
        $Yaml = "---`ntitle: `"$CleanTitle`"`nfield: `"$Field`"`nstatus: `"Imported`"`ncreated_date: $CurrentDate`npdf_link: `"[[$( $CleanTitle ).pdf]]`"`ntags: [paper, $Field]`n---`n`n"

        # 寫回檔案
        [System.IO.File]::WriteAllText($TargetFile.FullName, $Yaml + $FullContent, $Utf8NoBom)
        Write-Host "✅ Metadata injected successfully (Field: $Field)." -ForegroundColor Green
    }
}
catch {
    Write-Host "❌ Metadata Error: $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host "--- Operation Complete ---" -ForegroundColor Cyan