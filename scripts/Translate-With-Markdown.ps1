param(
    [Parameter(Mandatory=$true)]
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
$FullPath = $OriginalFileObj.FullName
$BaseName = $OriginalFileObj.BaseName
$ParentDir = $OriginalFileObj.DirectoryName

# --- [新增] 自動歸檔邏輯 (Auto-Archive Logic) ---
# 目的：建立與論文同名的資料夾，並將 PDF 移入，確保生成的 images 不會與其他論文混雜
$PaperFolder = Join-Path $ParentDir $BaseName
$NewPdfPath = Join-Path $PaperFolder $OriginalFileObj.Name

# 1. 建立專屬資料夾 (如果不存在)
if (-not (Test-Path $PaperFolder)) {
    New-Item -ItemType Directory -Path $PaperFolder | Out-Null
    Write-Host "📂 Created Workspace: $PaperFolder" -ForegroundColor Cyan
}

# 2. 移動 PDF 到專屬資料夾 (如果它還不在裡面的話)
if ($FullPath -ne $NewPdfPath) {
    Move-Item -Path $FullPath -Destination $PaperFolder -Force
    Write-Host "🚚 Moved PDF to Workspace..." -ForegroundColor DarkGray
    # 更新 FullPath 指向新的位置
    $FullPath = $NewPdfPath
}

$OutputFile = $FullPath -replace '\.[^.]+$', '.zh.md'
# [修正] 指向新的 python 腳本
$VenvPython = Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe"

if (Test-Path $VenvPython) {
    $PythonExe = $VenvPython
    Write-Host "🐍 Using Virtual Environment: $PythonExe" -ForegroundColor Green
} else {
    # 如果找不到虛擬環境，嘗試用全域 python (可能會失敗)
    $PythonExe = "python"
    Write-Host "⚠️  Virtual environment not found at $VenvPython. Using global python." -ForegroundColor Yellow
}

# 指向 V10 (Marker) 腳本
$TranslatorScript = Join-Path $PSScriptRoot "..\src\paper_translator_v6.py"

Write-Host "---"
Write-Host "Processing: $FullPath" -ForegroundColor Cyan

# 1. 呼叫 Python 翻譯核心
# 注意：所有複雜邏輯都在 Python 裡了，這裡只要等待它完成
$pyProc = Start-Process -FilePath $PythonExe -ArgumentList "`"$TranslatorScript`"", "`"$FullPath`"", "`"$OutputFile`"" -Wait -NoNewWindow -PassThru

if ($pyProc.ExitCode -eq 0) {
    
    # 2. Metadata 注入
    if (Test-Path $OutputFile) {
        $TargetFile = Get-Item $OutputFile
        $FullContent = [System.IO.File]::ReadAllText($TargetFile.FullName, $Utf8NoBom)
        
        if (-not $FullContent.StartsWith("---")) {
            $Field = $TargetFile.Directory.Parent.Name
            $CleanTitle = $TargetFile.BaseName -replace "\.zh$", ""
            $CurrentDate = Get-Date -Format "yyyy-MM-dd"
            
            $Yaml = $Yaml = @"
---
title: "$CleanTitle"
field: "$Field"
status: "Imported"
created_date: $CurrentDate
pdf_link: "[[$( $CleanTitle ).pdf]]"
tags: [paper, $Field]
---

"@
            [System.IO.File]::WriteAllText($TargetFile.FullName, $Yaml + $FullContent, $Utf8NoBom)
            Write-Host "Metadata injected." -ForegroundColor Green
        }
    }
} else {
    Write-Host "Translation failed (Python Script Error)." -ForegroundColor Red
    exit 1
}