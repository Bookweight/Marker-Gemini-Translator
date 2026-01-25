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

$FullPath = (Get-Item $InputFile).FullName
$OutputFile = $FullPath -replace '\.[^.]+$', '.zh.md'
# [修正] 指向新的 python 腳本
$TranslatorScript = Join-Path $PSScriptRoot "..\src\paper_translator.py"

Write-Host "---"
Write-Host "Processing: $FullPath" -ForegroundColor Cyan

# 1. 呼叫 Python 翻譯核心
# 注意：所有複雜邏輯都在 Python 裡了，這裡只要等待它完成
$pyProc = Start-Process -FilePath "python" -ArgumentList "`"$TranslatorScript`"", "`"$FullPath`"", "`"$OutputFile`"" -Wait -NoNewWindow -PassThru

if ($pyProc.ExitCode -eq 0) {
    
    # 2. Metadata 注入
    if (Test-Path $OutputFile) {
        $TargetFile = Get-Item $OutputFile
        $FullContent = [System.IO.File]::ReadAllText($TargetFile.FullName, $Utf8NoBom)
        
        if (-not $FullContent.StartsWith("---")) {
            $Field = $TargetFile.Directory.Parent.Name
            $CleanTitle = $TargetFile.BaseName -replace "\.zh$", ""
            $CurrentDate = Get-Date -Format "yyyy-MM-dd"
            
            $Yaml = @"
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