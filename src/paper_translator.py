import sys
import os
import re
import time
import subprocess
import pymupdf4llm
from pathlib import Path

# --- 設定 ---
GEMINI_CMD = "gemini"  # Gemini CLI 指令
MAX_RETRIES = 3        # API 失敗重試次數

def is_table_row(line):
    """
    偵測表格行 (更嚴謹的判斷)
    """
    # 1. Markdown 表格分隔線
    if re.match(r'^\s*\|?[\s\-:|]+\|?\s*$', line): return True
    # 2. 包含 '|' 的行
    if "|" in line: return True
    # 3. 連續的多個數字 (針對無格線表格)
    if re.search(r'\d+\.?\d*\s{2,}\d+', line): return True
    return False

def clean_text(text):
    """
    [安全模式] 清洗 pymupdf 雜訊
    """
    # 1. 移除 pymupdf 的假刪除線與底線雜訊
    text = re.sub(r'~~_(.)_~~', r'\1', text) 
    text = re.sub(r'~~(.*?)~~', r'\1', text)
    
    lines = text.split('\n')
    new_lines = []
    buffer = ""
    
    # 嚴格的標題關鍵字
    header_keywords = [
        "Abstract", "Introduction", "Related Work", "Method", "Methodology", 
        "Experiments", "Results", "Conclusion", "Discussion", "References",
        "Appendix", "Limitations"
    ]
    
    in_code_block = False

    for line in lines:
        stripped = line.strip()
        
        # 雜訊過濾
        if not stripped or "pymupdf_layout" in stripped or stripped.startswith("--- PAGE"):
            if buffer: new_lines.append(buffer); buffer = ""
            if not stripped: new_lines.append("") 
            continue

        # Code Block 保護
        if stripped.startswith("```"):
            if buffer: new_lines.append(buffer); buffer = ""
            in_code_block = not in_code_block
            new_lines.append(line)
            continue
            
        if in_code_block or is_table_row(line):
            if buffer: new_lines.append(buffer); buffer = ""
            new_lines.append(line)
            continue

        # 標題偵測
        is_header = False
        if stripped in header_keywords: is_header = True
        elif re.match(r'^\d+\.\s+[A-Z]', stripped) and len(stripped) < 60: is_header = True
        elif re.match(r'^(Figure|Table)\s+\d+[:\.]', stripped): 
            if buffer: new_lines.append(buffer); buffer = ""
            new_lines.append(f"\n{stripped}")
            continue
        
        if is_header:
            if buffer: new_lines.append(buffer); buffer = ""
            prefix = "## " if not stripped.startswith("#") else ""
            new_lines.append(f"\n{prefix}{stripped}\n")
            continue

        # 內文重組
        if buffer:
            if buffer.strip()[-1] in ['.', '?', '!', ':']:
                new_lines.append(buffer)
                buffer = stripped
            elif buffer.endswith("-"):
                buffer = buffer[:-1] + stripped
            else:
                buffer += " " + stripped
        else:
            buffer = stripped

    if buffer: new_lines.append(buffer)
    return "\n".join(new_lines)

def process_paper(pdf_path, output_path):
    # 1. 轉換 PDF -> Markdown
    print(f"Converting PDF: {pdf_path}...", file=sys.stderr)
    try:
        raw_md = pymupdf4llm.to_markdown(pdf_path)
    except Exception as e:
        print(f"PDF Conversion Failed: {e}", file=sys.stderr)
        return False

    # 2. 切分 References (不翻譯)
    print(f"Splitting References...", file=sys.stderr)
    ref_pattern = re.compile(r'(?im)^[\s#*]*(?:References|Bibliography|Literature Cited)[\s*:]*$')
    matches = list(ref_pattern.finditer(raw_md))
    
    body = raw_md
    refs = ""
    if matches:
        split_idx = matches[-1].start()
        if split_idx > len(raw_md) * 0.5:
            body = raw_md[:split_idx]
            refs = raw_md[split_idx:]

    # 3. 清洗正文
    clean_body = clean_text(body)

    # 4. 構建 Prompt
    prompt = f"""
SYSTEM_MODE: DOCUMENT_PROCESSOR
ROLE: You are an academic translation engine.

**TOOL USE POLICY:**
1. **FORBIDDEN:** Do NOT use tools (write_file, etc.).
2. **OUTPUT:** Stream translated text DIRECTLY to stdout.

**TASK:**
Translate the academic paper content below into Traditional Chinese (繁體中文).

**FORMATTING RULES:**
1. **Bilingual:** Original English Paragraph -> Empty Line -> Traditional Chinese Translation.
2. **Structure:** - Main Header: `## Original - 中文`
   - Sub-Header: `### 中文` (Chinese Only)
3. **Tables (HTML):** - Use `<table>` syntax for ALL tables.
   - Keep content **100% ENGLISH**.
4. **Math:** Detect broken math (e.g. "x 2") and repair to LaTeX (`$x^2$`). Keep variables English.

**CONTENT:**
---
{clean_body}
---
"""

    # 5. 呼叫 Gemini (修復 Windows 呼叫問題)
    print(f"Sending to Gemini (Length: {len(prompt)} chars)...", file=sys.stderr)
    translation = ""
    
    # 檢測作業系統
    is_windows = sys.platform.startswith("win")
    
    for attempt in range(MAX_RETRIES):
        try:
            # [核心修正]
            # 1. shell=is_windows: 讓 Windows 能找到 gemini.cmd
            # 2. args=[GEMINI_CMD]: 不加 "-" 參數，直接透過 stdin 傳送
            process = subprocess.Popen(
                [GEMINI_CMD], 
                stdin=subprocess.PIPE, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                text=True, 
                encoding='utf-8',
                shell=is_windows 
            )
            
            # 寫入 Prompt 到 Stdin
            stdout, stderr = process.communicate(input=prompt)
            
            if process.returncode == 0 and len(stdout.strip()) > 100:
                translation = stdout
                break
            else:
                # 過濾掉 "Loaded cached credentials" 這種非致命雜訊
                if "NativeCommandError" not in stderr and "Loaded cached credentials" not in stderr:
                     print(f"⚠️  Attempt {attempt+1} failed. Stderr: {stderr[:200]}...", file=sys.stderr)
                elif len(stdout.strip()) > 100:
                    # 如果 stderr 有東西但 stdout 也有內容，視為成功 (忽略雜訊)
                    translation = stdout
                    break
                
                time.sleep(5)
                
        except Exception as e:
            print(f"Exception on attempt {attempt+1}: {e}", file=sys.stderr)
            time.sleep(5)

    if not translation:
        print("All retry attempts failed.", file=sys.stderr)
        return False

    # 6. 後處理與存檔
    print("Saving...", file=sys.stderr)
    
    # 清洗模型廢話
    translation = re.sub(r'(?i)^.*SYSTEM_MODE.*$', '', translation, flags=re.MULTILINE)
    translation = re.sub(r'(?i)^.*I will now.*$', '', translation, flags=re.MULTILINE)
    
    final_content = translation.strip() + "\n\n" + refs
    final_content = re.sub(r'(?m)(^##)', r'\n\n\1', final_content)

    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(final_content)
        print(f"Done: {output_path}", file=sys.stderr)
        return True
    except Exception as e:
        print(f"Write Failed: {e}", file=sys.stderr)
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python paper_translator.py input.pdf [output.md]")
        sys.exit(1)
        
    input_file = sys.argv[1]
    if len(sys.argv) >= 3:
        output_file = sys.argv[2]
    else:
        output_file = str(Path(input_file).with_suffix('.zh.md'))
        
    success = process_paper(input_file, output_file)
    sys.exit(0 if success else 1)