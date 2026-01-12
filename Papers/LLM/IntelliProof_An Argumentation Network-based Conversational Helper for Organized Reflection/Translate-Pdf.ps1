param(
    [Parameter(Mandatory=$true)]
    [string]$InputFile
)
# **關鍵修復 V2：強制設定所有編碼為 UTF-8 (解決亂碼的最後手段)**
$OutputEncoding = [System.Text.Encoding]::UTF8  # 標準輸出流
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8 # 主控台顯示
[Console]::InputEncoding = [System.Text.Encoding]::UTF8  # 主控台輸入
$GeminiCommand = "gemini"

# Check if the input file exists
if (-not (Test-Path $InputFile -PathType Leaf)) {
    Write-Host "Error: File '$InputFile' does not exist." -ForegroundColor Red
    exit 1
}

# Get the full path of the file
$FullPath = (Get-Item $InputFile).FullName

# Create the output file name (Replace extension with .zh.md)
$OutputFile = $FullPath -replace '\.[^.]+$', '.zh.md'

Write-Host "---"
Write-Host "✅ Processing file: $FullPath" -ForegroundColor Green
Write-Host "➡️ Translation result will be saved to: $OutputFile" -ForegroundColor Yellow
Write-Host "---"

# Set the translation prompt for Gemini (Optimized for academic papers)
# We append the file reference using the '@' symbol at the end of the prompt.
$TranslationPrompt = "Please translate the entire attached PDF document from beginning to end into Traditional Chinese (繁體中文) in a single, continuous operation. Do not pause for confirmation or ask to proceed with subsequent pages.

Strictly adhere to the following academic translation and formatting requirements:
1. **Preserve Structure:** The translation must maintain the original document's **section structure and heading levels** (e.g., # Abstract, ## Section 1).
2. **Retain Key Terms:** **DO NOT translate** mathematical formulas, code, specialized terms, proper nouns, or English acronyms in figures/tables. Keep them in their original English.
3. **Academic Style:** Ensure the Chinese translation is rigorous, professional, and consistent with academic paper tone.
4. **Bilingual Output:** For each paragraph, present the original English text first, followed by its Traditional Chinese translation.
5. **Image Placeholder:** If the original document contains an image, insert a placeholder text like `[Image]` in the output.
6. **Output Format:** The final translated result must be outputted in **Markdown format**."

# Construct the command using the '@' file reference syntax:
# gemini -p "PROMPT @FILE_PATH"
$CommandString = "$TranslationPrompt" + " @" + "$FullPath"

# Execute the Gemini CLI translation command
# The output is captured in $result, including errors.
$result = & $GeminiCommand -p "$CommandString" 2>&1 

# Check the execution result
# If the output contains specific error messages or is empty, assume failure.
if ($result -is [string] -and ($result -match "Error" -or $result -match "Usage:")) {
    Write-Host "---"
    Write-Host "❌ Translation failed. Check for API quota limits or file size restrictions." -ForegroundColor Red
    Write-Host "Detailed CLI Output:"
    Write-Host "$result"
    Write-Host "---"
} else {
    # Write the output content to the file
    $result | Out-File -FilePath $OutputFile -Encoding UTF8
    
    Write-Host "---"
    Write-Host "🎉 Translation successful!" -ForegroundColor Cyan
    Write-Host "File saved to: $OutputFile" -ForegroundColor Cyan
    Write-Host "---"
}