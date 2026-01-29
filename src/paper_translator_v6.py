import sys
import os
import re
import time
import subprocess
import shutil
from pathlib import Path
import fitz  # PyMuPDF

# --- 設定 ---
GEMINI_CMD = "gemini"
MAX_RETRIES = 3
BATCH_SIZE_LIMIT = 10000 

print("🚀 Loading Marker AI models (V28 Math Zero-Interference)...", file=sys.stderr)

try:
    from marker.converters.pdf import PdfConverter
    from marker.models import create_model_dict
    from marker.output import text_from_rendered
    from marker.config.parser import ConfigParser
except ImportError:
    print("❌ Critical Modules missing.", file=sys.stderr)
    sys.exit(1)

# Marker 設定
config_dict = {
    "output_format": "markdown",
    "disable_image_extraction": False,
    "disable_table_extraction": False,
    "paginate_output": False
}
config_parser = ConfigParser(config_dict)

converter = PdfConverter(
    config=config_parser.generate_config_dict(),
    artifact_dict=create_model_dict(),
    processor_list=config_parser.get_processors(),
    renderer=config_parser.get_renderer()
)

def clean_filename(name):
    return re.sub(r'[^a-zA-Z0-9_.-]', '_', name)

def detect_content_boundaries(page):
    """邊界偵測 (V26 Logic)"""
    page_height = page.rect.height
    top_limit = 0
    bottom_limit = page_height
    paths = page.get_drawings()
    horizontal_lines = []
    
    for p in paths:
        rect = p["rect"]
        if rect.width > page.rect.width * 0.4 and rect.height < 5:
            horizontal_lines.append(rect.y0)
            
    if horizontal_lines:
        horizontal_lines.sort()
        header_candidates = [y for y in horizontal_lines if y < page_height * 0.2]
        if header_candidates: top_limit = header_candidates[-1]
        footer_candidates = [y for y in horizontal_lines if y > page_height * 0.75]
        if footer_candidates: bottom_limit = footer_candidates[0]
            
    return top_limit, bottom_limit

def extract_images_with_boundary_check(pdf_path, output_dir):
    """圖片提取 (V26 Logic)"""
    images_dir = os.path.join(output_dir, "images")
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
        
    print("⚠️ Extracting images...", file=sys.stderr)
    doc = fitz.open(pdf_path)
    extracted_map = {} 
    
    for page_index in range(len(doc)):
        page = doc[page_index]
        image_list = page.get_images(full=True)
        extracted_map[page_index] = []
        top_limit, bottom_limit = detect_content_boundaries(page)
        
        if image_list:
            for img_index, img in enumerate(image_list):
                xref = img[0]
                is_noise = False
                rects = page.get_image_rects(xref)
                if rects:
                    rect = rects[0]
                    mid_y = (rect.y0 + rect.y1) / 2
                    if mid_y < top_limit or mid_y > bottom_limit:
                        is_noise = True
                
                try:
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]
                    image_name = f"p{page_index}_img{img_index}.{image_ext}"
                    image_path = os.path.join(images_dir, image_name)
                    with open(image_path, "wb") as f:
                        f.write(image_bytes)
                    extracted_map[page_index].append({
                        "filename": image_name, "is_noise": is_noise
                    })
                except Exception:
                    pass
    return extracted_map

def clean_text_noise(text):
    """文字清洗 (V26 Logic)"""
    lines = text.split('\n')
    cleaned_lines = []
    patterns = [
        r'^\s*\d+\s+Page\s+\d+\s+of\s+\d+',
        r'^\s*Page\s+\d+\s*$',
        r'^\s*arXiv:\d+\.\d+.*$',
        r'^\s*https?://doi\.org/.*$',
        r'.*©.*Permission\s+to\s+make.*',
        r'^\s*Vol\.\s+\d+,\s+No\.\s+\d+.*$'
    ]
    compiled_patterns = [re.compile(p, re.IGNORECASE) for p in patterns]
    
    for line in lines:
        is_noise = False
        if len(line) < 100:
            for p in compiled_patterns:
                if p.match(line):
                    is_noise = True
                    break
        if not is_noise: cleaned_lines.append(line)
    return '\n'.join(cleaned_lines)

def force_normalize_headers(text):
    """標題修復 (V23 Logic)"""
    lines = text.split('\n')
    new_lines = []
    h2_pattern = re.compile(r'^[*#]*\s*(\d+\.?\s+[A-Z].*?)[*#]*$') 
    h3_pattern = re.compile(r'^[*#]*\s*(\d+\.\d+\.?\s+.*?)[*#]*$')
    special_headers = ["Abstract", "References", "Introduction", "Conclusion"]

    for line in lines:
        clean_line = line.strip()
        if not clean_line:
            new_lines.append(line)
            continue
        content = re.sub(r'^[*#]+\s*', '', clean_line)
        if h3_pattern.match(clean_line):
            match = h3_pattern.match(clean_line)
            content = re.sub(r'^[*#]+\s*', '', match.group(0))
            new_lines.append(f"### {content}")
        elif h2_pattern.match(clean_line) or any(content.startswith(h) for h in special_headers):
            match = h2_pattern.match(clean_line)
            if match: content = re.sub(r'^[*#]+\s*', '', match.group(0))
            new_lines.append(f"## {content}")
        else:
            new_lines.append(line)
    return '\n'.join(new_lines)

def inject_images_sync_filter(text, image_map):
    """圖片注入 (V26 Logic)"""
    pattern = re.compile(r'!\[(.*?)\]\((.*?)_page_(\d+)_Picture_.*?\)')
    parts = []
    last_end = 0
    page_counter = {} 

    for match in pattern.finditer(text):
        parts.append(text[last_end:match.start()])
        alt = match.group(1) or "Figure"
        page_idx = int(match.group(3))
        
        if page_idx not in page_counter: page_counter[page_idx] = 0
        current_idx = page_counter[page_idx]
        images_on_page = image_map.get(page_idx, [])
        
        if current_idx < len(images_on_page):
            img_data = images_on_page[current_idx]
            if not img_data["is_noise"]:
                fname = img_data["filename"]
                parts.append(f"![{alt}](images/{fname})")
            page_counter[page_idx] += 1
        else:
            parts.append(f"> *[Figure: Vector/Text - Not Extracted]*")
        last_end = match.end()
    parts.append(text[last_end:])
    return "".join(parts)

def convert_with_marker_and_fix(pdf_path, output_dir):
    images_dir = os.path.join(output_dir, "images")
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)

    try:
        rendered = converter(pdf_path)
        ret_val = text_from_rendered(rendered)
        full_text = ret_val[0] if isinstance(ret_val, tuple) and len(ret_val) >= 2 else str(ret_val)

        smart_image_map = extract_images_with_boundary_check(pdf_path, output_dir)
        full_text = inject_images_sync_filter(full_text, smart_image_map)
        full_text = clean_text_noise(full_text)
        return full_text
    except Exception as e:
        print(f"❌ Marker Conversion Failed: {e}", file=sys.stderr)
        return None

def split_text_into_logical_blocks(text):
    """
    [V28 Logic] 切分時保護數學區塊不被切斷
    """
    lines = text.split('\n')
    blocks = []
    current_block = []
    in_table = False
    in_code = False
    in_math = False
    
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"): in_code = not in_code
        if stripped == "$$": in_math = not in_math
        if '|' in line and len(line) > 5: in_table = True
        elif stripped == "": in_table = False
        
        current_block.append(line)
        
        if not in_table and not in_code and not in_math and stripped == "":
            content = "\n".join(current_block).strip()
            if content: blocks.append("\n".join(current_block))
            current_block = []
            
    if current_block: blocks.append("\n".join(current_block))
    return blocks

def is_translatable(block):
    """
    [V28 新增] 判斷區塊是否需要翻譯
    回傳 False 代表：這是公式/代碼/表格/圖片，直接跳過 API 請求
    """
    block = block.strip()
    # 1. 數學公式區塊 ($$ ... $$)
    if block.startswith("$$") and block.endswith("$$"):
        return False
    # 2. 代碼區塊 (``` ... ```)
    if block.startswith("```"):
        return False
    # 3. 圖片連結 (![...](...))
    if re.match(r'^!\[.*?\]\(.*?\)$', block):
        return False
    # 4. 表格 (包含 | 分隔符)
    if "|" in block and "-|-" in block: # 簡單的 Markdown 表格偵測
        return False
    
    return True

def call_gemini(prompt):
    is_windows = sys.platform.startswith("win")
    for attempt in range(MAX_RETRIES):
        try:
            process = subprocess.Popen(
                [GEMINI_CMD], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, encoding='utf-8', shell=is_windows 
            )
            stdout, stderr = process.communicate(input=prompt)
            if process.returncode == 0 and len(stdout.strip()) > 0: return stdout
            if "429" in stderr or "quota" in stderr.lower():
                time.sleep(10)
            else:
                time.sleep(2)
        except Exception as e:
            print(f"❌ API Error: {e}", file=sys.stderr)
            time.sleep(5)
    return None

def translate_batch(batch_blocks, start_id):
    prompt_text = ""
    for i, block in enumerate(batch_blocks):
        prompt_text += f"<<<ID_{start_id + i}>>>\n{block}\n\n"

    # [V28 Prompt] 移除已經在 Python 端過濾掉的規則，精簡 Prompt
    prompt = f"""
SYSTEM_MODE: ACADEMIC_TRANSLATOR
**TASK:** Translate text blocks to Traditional Chinese (Taiwan).

**OUTPUT FORMAT:**
<<<ID_x>>>
[Chinese Translation]

**RULES:**
1. **Style:** Academic, formal.
2. **Inline Math:** Keep inline LaTeX (`$...$`) EXACTLY as is.
3. **No English:** Output ONLY Chinese.

**INPUT:**
{prompt_text}

**OUTPUT:**
"""
    result = call_gemini(prompt)
    translations = {}
    if result:
        matches = re.finditer(r'<<<ID_(\d+)>>>\s*(.*?)(?=(<<<ID_|\Z))', result, re.DOTALL)
        for match in matches:
            idx = int(match.group(1))
            content = match.group(2).strip()
            translations[idx] = content
    return translations

def process_paper(pdf_path, output_path):
    output_dir = os.path.dirname(output_path)
    raw_output_path = str(Path(output_path).with_suffix('.raw.md'))
    
    print(f"🔥 Processing PDF with Marker...", file=sys.stderr)
    raw_md = convert_with_marker_and_fix(pdf_path, output_dir)
    if not raw_md: return False

    print("📐 Normalizing Headers...", file=sys.stderr)
    raw_md = force_normalize_headers(raw_md)

    ref_match = re.search(r'^##\s+(References|Bibliography)', raw_md, re.MULTILINE | re.IGNORECASE)
    
    pre_text = ""
    body_text = raw_md
    post_text = ""
    
    if ref_match:
        split_idx = ref_match.start()
        post_text = raw_md[split_idx:]
        body_text = raw_md[:split_idx]
    
    abstract_match_in_body = re.search(r'^##\s+Abstract', body_text, re.MULTILINE | re.IGNORECASE)
    if abstract_match_in_body:
        split_idx = abstract_match_in_body.start()
        pre_text = body_text[:split_idx]
        body_text = body_text[split_idx:]

    original_blocks = split_text_into_logical_blocks(body_text)
    print(f"🧩 Translating body: {len(original_blocks)} blocks.", file=sys.stderr)

    final_blocks = []
    current_batch = []
    current_batch_len = 0
    batch_start_index = 0
    
    # 這裡我們需要一個 Map 來記錄「哪些 ID 被跳過了」
    # 或者簡單一點：我們只把「可翻譯」的區塊加入 current_batch
    # 但是 current_batch 裡的 ID 必須跟 original_blocks 的 index 對應嗎？
    # V28 策略：Batch 裡面的 ID 使用 original_blocks 的真實 Index
    
    print(f"🚀 Starting Batch Translation (Max: {BATCH_SIZE_LIMIT} chars)...", file=sys.stderr)

    # 1. 收集翻譯結果
    all_translations = {} 

    for i, block in enumerate(original_blocks):
        # [V28 核心] 本地端過濾：如果不可翻譯，直接跳過，不加入 Batch
        if not is_translatable(block):
            continue 

        current_batch.append((i, block)) # 存入 (Index, Content)
        current_batch_len += len(block)

        if current_batch_len >= BATCH_SIZE_LIMIT or i == len(original_blocks) - 1:
            # 建構 payload，注意這裡 start_id 不再是單純的計數，而是真實 Index
            prompt_text = ""
            for idx, content in current_batch:
                prompt_text += f"<<<ID_{idx}>>>\n{content}\n\n"
            
            # 呼叫 API (這裡把 translate_batch 內聯展開以便處理自訂 ID)
            # [V28 Prompt] 
            prompt = f"""
SYSTEM_MODE: ACADEMIC_TRANSLATOR
**TASK:** Translate text blocks to Traditional Chinese (Taiwan).

**OUTPUT FORMAT:**
<<<ID_x>>>
[Chinese Translation]

**RULES:**
1. **Style:** Academic, formal.
2. **Inline Math:** Keep inline LaTeX (`$...$`) EXACTLY as is.
3. **No English:** Output ONLY Chinese.

**INPUT:**
{prompt_text}

**OUTPUT:**
"""
            print(f"📤 Sending Batch (Count: {len(current_batch)})...", file=sys.stderr)
            result = call_gemini(prompt)
            
            if result:
                matches = re.finditer(r'<<<ID_(\d+)>>>\s*(.*?)(?=(<<<ID_|\Z))', result, re.DOTALL)
                for match in matches:
                    idx = int(match.group(1))
                    content = match.group(2).strip()
                    all_translations[idx] = content
            
            current_batch = []
            current_batch_len = 0
            time.sleep(2)

    # 2. 重組文章
    for i, block in enumerate(original_blocks):
        trans_text = all_translations.get(i, "")
        
        # 如果 trans_text 為空 (可能是被 is_translatable 過濾掉，或是 API 沒回傳)
        # 就只顯示原文
        if not trans_text or trans_text == "[ORIGINAL]":
            final_blocks.append(block)
        else:
            final_blocks.append(f"{block}\n\n> {trans_text}")

    full_content = ""
    if pre_text: full_content += pre_text + "\n\n"
    
    body_content = "\n\n".join(final_blocks)
    body_content = re.sub(r'<<<ID_\d+>>>', '', body_content)
    full_content += body_content
    
    if post_text: full_content += "\n\n" + post_text

    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(full_content)
        print(f"✅ Success: {output_path}", file=sys.stderr)
    except Exception as e:
        print(f"❌ Write Error: {e}", file=sys.stderr)

if __name__ == "__main__":
    if len(sys.argv) < 2: sys.exit(1)
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) >= 3 else str(Path(input_file).with_suffix('.zh.md'))
    process_paper(input_file, output_file)