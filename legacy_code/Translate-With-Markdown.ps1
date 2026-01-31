param(
    [Parameter(Mandatory = $true)]
    [string]$InputFile
)

$OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$Utf8NoBom = New-Object System.Text.UTF8Encoding($false)

if (-not (Test-Path $InputFile -PathType Leaf)) {
    Write-Host "Error: File '$InputFile' does not exist." -ForegroundColor Red
    exit 1
}

$OriginalFileObj = Get-Item $InputFile
$OriginalDir = $OriginalFileObj.DirectoryName
$BaseName = $OriginalFileObj.BaseName

# 1. Setup Temp Paths (Short Paths to avoid MAX_PATH issues)
$TempDir = [System.IO.Path]::GetTempPath()
# Using a fixed simple name to ensure it's short
$TempPdfPath = Join-Path $TempDir "process_paper.pdf"
$TempMdPath = Join-Path $TempDir "process_paper.zh.md"
$TempImagesDir = Join-Path $TempDir "images"

# Clean up previous temp files
if (Test-Path $TempPdfPath) { Remove-Item $TempPdfPath -Force }
if (Test-Path $TempMdPath) { Remove-Item $TempMdPath -Force }
if (Test-Path $TempImagesDir) { Remove-Item $TempImagesDir -Recurse -Force }

# 2. Copy Target PDF to Temp
Write-Host "🚚 Copying to temp for processing: $TempPdfPath" -ForegroundColor DarkGray
Copy-Item -Path $OriginalFileObj.FullName -Destination $TempPdfPath -Force

# 3. Setup Python Environment
$VenvPython = Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe"
if (Test-Path $VenvPython) {
    $PythonExe = $VenvPython
}
else {
    $PythonExe = "python"
}

$TranslatorScript = Join-Path $PSScriptRoot "..\src\paper_translator_v6.py"

Write-Host "---"
Write-Host "Processing Temp File..." -ForegroundColor Cyan

# 4. Run Python Script on Temp File
$pyProc = Start-Process -FilePath $PythonExe -ArgumentList "`"$TranslatorScript`"", "`"$TempPdfPath`"" -Wait -NoNewWindow -PassThru

if ($pyProc.ExitCode -eq 0) {
    Write-Host "✅ Processing Complete. Moving results back..." -ForegroundColor Green
    
    # 5. Move Results Back
    
    # [Feature] Create a dedicated folder for the paper (Same name as PDF base name)
    $PaperDir = Join-Path $OriginalDir $BaseName

    # [Fix] Idempotency: Check if we are already in the correct folder to avoid nested folders (e.g. Paper/Paper/Paper.pdf)
    $ParentDirName = Split-Path $OriginalDir -Leaf
    if ($ParentDirName -eq $BaseName) {
        $PaperDir = $OriginalDir
        Write-Host "✅ Already in dedicated paper folder: $PaperDir"
    }
    elseif (-not (Test-Path $PaperDir)) {
        New-Item -ItemType Directory -Path $PaperDir | Out-Null
        Write-Host "📂 Created directory: $PaperDir"
    }

    # Move the original PDF into this folder
    $FinalPdfPath = Join-Path $PaperDir $OriginalFileObj.Name
    # Check if we are not already in the target folder to avoiding move error
    if ($OriginalFileObj.FullName -ne $FinalPdfPath) {
        if (Test-Path $FinalPdfPath) { Remove-Item $FinalPdfPath -Force }
        Move-Item -Path $OriginalFileObj.FullName -Destination $FinalPdfPath -Force
        Write-Host "📦 Moved original PDF to: $FinalPdfPath"
    }

    $FinalMdPath = Join-Path $PaperDir "$BaseName.zh.md"
    $FinalImagesDir = Join-Path $PaperDir "images"

    # Move Markdown
    if (Test-Path $TempMdPath) {
        # Read content for metadata injection FIRST (while it's in temp)
        $FullContent = [System.IO.File]::ReadAllText($TempMdPath, $Utf8NoBom)
        
        # Inject Metadata if needed
        if (-not $FullContent.StartsWith("---")) {
            # [Fix] Smart Field Detection: Look for 'Papers' folder and take the next subfolder
            $PathParts = $OriginalDir.Split([System.IO.Path]::DirectorySeparatorChar)
            $Field = $PathParts[-1] # Default fallback
            for ($i = 0; $i -lt $PathParts.Count; $i++) {
                if ($PathParts[$i] -eq "Papers") {
                    if (($i + 1) -lt $PathParts.Count) {
                        $Field = $PathParts[$i + 1]
                        break
                    }
                }
            }
            $Field = $Field -replace ' ', '_'
            
            # [Fix] Tag formatting: "Time_Series" -> "Time_series" for tags only
            # Field remains as "Time_Series"
            $TagName = $Field
            $parts = $Field.Split('_')
            if ($parts.Count -gt 1) {
                $parts[0] = (Get-Culture).TextInfo.ToTitleCase($parts[0].ToLower())
                for ($k = 1; $k -lt $parts.Count; $k++) {
                    $parts[$k] = $parts[$k].ToLower()
                }
                $TagName = $parts -join '_'
            }
            
            $CleanTitle = $BaseName -replace "\.zh$", ""
            $CurrentDate = Get-Date -Format "yyyy-MM-dd"
             
            $Yaml = @"
---
title: "$CleanTitle"
field: "$Field"
status: "Imported"
created_date: $CurrentDate
pdf_link: "[[$( $OriginalFileObj.Name )]]"
tags: [paper, $TagName]
---

"@
            $FullContent = $Yaml + $FullContent
        }
        
        [System.IO.File]::WriteAllText($FinalMdPath, $FullContent, $Utf8NoBom)
        Write-Host "📄 Markdown saved to: $FinalMdPath"
    }

    # Move Images
    if (Test-Path $TempImagesDir) {
        if (-not (Test-Path $FinalImagesDir)) {
            New-Item -ItemType Directory -Path $FinalImagesDir | Out-Null
        }
        Copy-Item "$TempImagesDir\*" -Destination $FinalImagesDir -Recurse -Force
        Write-Host "🖼️ Images synced to: $FinalImagesDir"
    }

}
else {
    Write-Host "❌ Translation failed (Python Script Error)." -ForegroundColor Red
    exit 1
}