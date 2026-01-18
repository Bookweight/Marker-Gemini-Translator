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
    $TranslationPrompt = "You are a professional academic translator. Your task is to translate the attached PDF document into Traditional Chinese (繁體中文).
    **IMPORTANT SYSTEM INSTRUCTION: DO NOT USE ANY TOOLS.** Do not attempt to run shell commands, file readers, or search tools. 
    Directly extract and translate the text content from the provided PDF file.
    Strictly follow these rules:
        1. **Text Content (Bilingual):** For standard paragraphs, headers, and list items, output the [Original English Text] first, followed immediately by its [Traditional Chinese Translation].
        2. **Tables (Translated Only):** For any tables found, output the **Translated Traditional Chinese Table** directly in Markdown format. Do NOT list the original English table inside the cells. Translate table headers and content, but keep technical terms (like 'ResNet-50', 'Accuracy') in English.
        3. **Structure:** Maintain all original headings (#, ##), bullets, and numbering.
        4. **Accuracy:** Keep formulas and acronyms intact. Ensure the translation is academic and professional.
        5. **Completeness:** Translate the entire document from start to finish.
        Output format must be Markdown."

    # [關鍵修正] 幫路徑加上引號，解決資料夾或檔名中有空白的問題
    # 注意：Gemini CLI 使用 @ 來指定檔案，我們把引號包在路徑外層
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
            $PaperFolderName = $TargetFile.Directory.Name
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