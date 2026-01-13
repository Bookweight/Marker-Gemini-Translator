# 指定路徑
$VaultRoot = "C:\Users\User\Desktop\paper reading"
$PapersRoot = Join-Path $VaultRoot "Papers"

# 建立 UTF-8 (無 BOM) 編碼物件
$Utf8NoBom = New-Object System.Text.UTF8Encoding($false)

# 遞迴尋找所有翻譯 MD 檔案
$MdFiles = Get-ChildItem -Path $PapersRoot -Filter "*.zh.md" -Recurse

Write-Host "--- Start Processing: $($MdFiles.Count) files found ---" -ForegroundColor Cyan

foreach ($File in $MdFiles) {
    try {
        # 1. 使用 .NET 讀取內容
        $FullContent = [System.IO.File]::ReadAllText($File.FullName, $Utf8NoBom)
        
        if ($FullContent.StartsWith("---")) {
            Write-Host "Skip: $($File.Name) (Already has YAML)" -ForegroundColor Yellow
            continue
        }

        # 2. 解析欄位 (Field) 與 標題
        $PaperFolderName = $File.Directory.Name
        $Field = $File.Directory.Parent.Name
        $CleanTitle = $File.BaseName -replace "\.zh$", ""

        # 3. 準備 YAML
        $CurrentDate = Get-Date -Format "yyyy-MM-dd"
        $Yaml = "---`ntitle: `"$CleanTitle`"`nfield: `"$Field`"`nstatus: `"Imported`"`ncreated_date: $CurrentDate`npdf_link: `"[[$( $CleanTitle ).pdf]]`"`ntags: [paper, $Field]`n---`n`n"

        # 4. 寫回檔案
        [System.IO.File]::WriteAllText($File.FullName, $Yaml + $FullContent, $Utf8NoBom)
        
        Write-Host "Success: $($File.Name) [Field: $Field]" -ForegroundColor Green
    }
    catch {
        Write-Host "Error processing $($File.Name): $($_.Exception.Message)" -ForegroundColor Red
    }
}

Write-Host "--- All Processed! Please check Obsidian. ---" -ForegroundColor Cyan