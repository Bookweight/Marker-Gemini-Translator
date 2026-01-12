param(
    [Parameter(Mandatory=$true)]
    [string]$InputFile
)
# **關鍵修復 V4：使用 Here-String 語法來避免 PowerShell 解析錯誤**
$OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::InputEncoding = [System.Text.Encoding]::UTF8
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

# 使用 Here-String (@"..."@) 來定義複雜的多行提示，以防止語法解析錯誤
$TranslationPrompt = @"
Please translate the entire attached PDF document from beginning to end into Traditional Chinese (繁體中文) in a single, continuous operation. Do not pause for confirmation or ask to proceed with subsequent pages.

Strictly adhere to the following academic translation and formatting requirements:
1. **Preserve Structure:** The translation must maintain the original document's **section structure and heading levels** (e.g., # Abstract, ## Section 1).
2. **Retain Key Terms:** **DO NOT translate** mathematical formulas, code, specialized terms, proper nouns, or English acronyms in figures/tables. Keep them in their original English.
3. **Academic Style:** Ensure the Chinese translation is rigorous, professional, and consistent with academic paper tone.
4. **Bilingual Output:** For each paragraph, present the original English text first, followed by its Traditional Chinese translation.
5. **Image Placeholder:** If the original document contains an image, insert a placeholder text like `[Image]` in the output.
6. **Output Format:** The final translated result must be outputted in **Markdown format**.
"@

# Construct the command using the '@' file reference syntax:
$CommandString = "$TranslationPrompt" + " @" + "$FullPath"

# Create a temporary file to capture the standard error stream
$tempErrorFile = New-TemporaryFile

try {
    # Execute the Gemini CLI command.
    # Standard Output (stdout) is captured in $result.
    # Standard Error (stderr) is redirected to the temporary file.
    $result = & $GeminiCommand -p "$CommandString" 2> $tempErrorFile.FullName
    
    # Read any content from the error file
    $errorContent = Get-Content $tempErrorFile.FullName -Raw
}
finally {
    # Ensure the temporary file is always removed
    Remove-Item $tempErrorFile -ErrorAction SilentlyContinue
}

# Filter out known, benign status messages from the error content
$filteredErrorContent = $errorContent -split '\r?\n' | Where-Object { $_ -notmatch "Loaded cached credentials." }

# A successful translation must have content in the $result (stdout).
# If $result is empty or only whitespace, it's a failure, regardless of error content.
if ([string]::IsNullOrWhiteSpace($result)) {
    Write-Host "---"
    Write-Host "❌ Translation failed. The command returned no content." -ForegroundColor Red
    
    # If there was actual error content after filtering, display it.
    if ($filteredErrorContent.Length -gt 0) {
        Write-Host "Detailed CLI Error Output:"
        Write-Host ($filteredErrorContent | Out-String) -ForegroundColor Yellow
    } else {
        Write-Host "No specific error message was captured. This could be due to API limits, an invalid file, or network issues."
    }
    Write-Host "---"
} else {
    # Success: Write the captured result (stdout) to the output file
    $result | Out-File -FilePath $OutputFile -Encoding UTF8
    
    Write-Host "---"
    Write-Host "🎉 Translation successful!" -ForegroundColor Cyan
    Write-Host "File saved to: $OutputFile" -ForegroundColor Cyan
    Write-Host "---"
}
