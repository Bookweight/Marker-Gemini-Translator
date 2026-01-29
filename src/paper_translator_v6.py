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

print("🚀 Loading Marker AI models (V26 Boundary Detector)...", file=sys.stderr)

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
    """
    [V26] 偵測頁面上的「分隔橫線」，定義有效內容區域。
    回傳: (top_limit, bottom_limit)
    """
    # 預設邊界 (如果沒抓到線，就用保守值)
    page_height = page.rect.height
    top_limit = 0
    bottom_limit = page_height

    # 取得所有繪圖路徑 (Drawings)
    paths = page.get_drawings()
    
    horizontal_lines = []
    
    for p in paths:
        rect = p["rect"]
        # 判斷是否為橫線：寬度夠寬，高度極小
        # 寬度至少要是頁面寬度的 40% 才算分隔線
        if rect.width > page.rect.width * 0.4 and rect.height < 5:
            horizontal_lines.append(rect.y0)
            
    if horizontal_lines:
        horizontal_lines.sort()
        
        # 策略：
        # 1. 最上面的線通常是 Header Separator (但要避免抓到表格內的線)
        #    我們假設 Header 線通常位於頁面頂部 20% 區域內
        header_candidates = [y for y in horizontal_lines if y < page_height * 0.2]
        if header_candidates:
            # 取最下面的一條 header line (以防 header 區塊有兩條線)
            top_limit = header_candidates[-1]
            
        # 2. 最下面的線通常是 Footer Separator (如果是註釋線)
        #    假設 Footer 線位於頁面底部 25% 區域內
        footer_candidates = [y for y in horizontal_lines if y > page_height * 0.75]
        if footer_candidates:
            # 取最上面的一條 footer line (因為註釋是在線下方)
            bottom_limit = footer_candidates[0]
            
    return top_limit, bottom_limit

def extract_images_with_boundary_check(pdf_path, output_dir):
    """
    [V26] 提取圖片，並標記是否為「越界」的雜訊
    注意：我們必須提取所有圖片以維持與 Marker 的索引對齊，但我們可以標記它為 skip
    """
    images_dir = os.path.join(output_dir, "images")
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
        
    print("⚠️ Extracting images with Boundary Check...", file=sys.stderr)
    doc = fitz.open(pdf_path)
    extracted_map = {} 
    
    for page_index in range(len(doc)):
        page = doc[page_index]
        image_list = page.get_images(full=True)
        extracted_map[page_index] = []
        
        # 1. 偵測該頁的邊界
        top_limit, bottom_limit = detect_content_boundaries(page)
        
        if image_list:
            for img_index, img in enumerate(image_list):
                xref = img[0]
                is_noise = False
                
                # 2. 檢查圖片位置
                rects = page.get_image_rects(xref)
                if rects:
                    rect = rects[0]
                    # 如果圖片中心點在邊界外，視為雜訊
                    mid_y = (rect.y0 + rect.y1) / 2
                    if mid_y < top_limit or mid_y > bottom_limit:
                        is_noise = True
                
                # 即使是雜訊，我們也要存下來 (為了佔位)，但在 Metadata 標記它
                try:
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]
                    image_name = f"p{page_index}_img{img_index}.{image_ext}"
                    image_path = os.path.join(images_dir, image_name)
                    
                    with open(image_path, "wb") as f:
                        f.write(image_bytes)
                    
                    extracted_map[page_index].append({
                        "filename": image_name,
                        "is_noise": is_noise,
                        "debug_info": f"y={mid_y:.1f}, bounds=({top_limit:.1f}, {bottom_limit:.1f})"
                    })
                except Exception:
                    pass
                    
    return extracted_map

def clean_text_noise(text):
    """
    [V26] 清洗常見的頁緣文字雜訊
    """
    lines = text.split('\n')
    cleaned_lines = []
    
    # 常見雜訊模式
    patterns = [
        r'^\s*\d+\s+Page\s+\d+\s+of\s+\d+', # "20 Page 30 of 47"
        r'^\s*Page\s+\d+\s*$',              # "Page 30"
        r'^\s*arXiv:\d+\.\d+.*$',           # arXiv ID
        r'^\s*https?://doi\.org/.*$',       # DOI Links (若獨立一行)
        r'.*©.*Permission\s+to\s+make.*',   # 版權宣告
        r'^\s*Vol\.\s+\d+,\s+No\.\s+\d+.*$' # 期刊卷號
    ]
    
    compiled_patterns = [re.compile(p, re.IGNORECASE) for p in patterns]
    
    for line in lines:
        is_noise = False
        if len(line) < 100: # 雜訊通常不長
            for p in compiled_patterns:
                if p.match(line):
                    is_noise = True
                    break
        
        if not is_noise:
            cleaned_lines.append(line)
            
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
    """
    [V26] 同步過濾注入
    遇到 Marker 的圖片標籤時，檢查對應的 PyMuPDF 圖片是否為雜訊。
    - 如果是雜訊：刪除標籤 (不顯示)。
    - 如果是正文圖：正常注入。
    - 如果對應不到 (Vector圖)：顯示提示。
    """
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
            
            if img_data["is_noise"]:
                # 是雜訊 (例如 Logo)，直接隱藏，不要佔位
                # 但計數器要 +1，因為 Marker 也有算這張圖
                pass 
            else:
                # 是好圖，注入
                fname = img_data["filename"]
                parts.append(f"![{alt}](images/{fname})")
            
            page_counter[page_idx] += 1
        else:
            # Marker 認為有圖，但 PyMuPDF 沒抓到 (可能是向量圖)
            # 這種情況通常不是雜訊 (雜訊通常是 Logo，是點陣圖)
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

        # 1. 提取圖片並標記雜訊
        smart_image_map = extract_images_with_boundary_check(pdf_path, output_dir)
        
        # 2. 注入圖片 (自動過濾雜訊)
        full_text = inject_images_sync_filter(full_text, smart_image_map)
        
        # 3. 清洗文字雜訊 (Header/Footer Text)
        full_text = clean_text_noise(full_text)
        
        return full_text
    except Exception as e:
        print(f"❌ Marker Conversion Failed: {e}", file=sys.stderr)
        return None

def split_text_into_logical_blocks(text):
    lines = text.split('\n')
    blocks = []
    current_block = []
    in_table = False
    in_code = False
    for line in lines:
        if line.strip().startswith("```"): in_code = not in_code
        if '|' in line and len(line) > 5: in_table = True
        elif line.strip() == "": in_table = False
        current_block.append(line)
        if not in_table and not in_code and line.strip() == "":
            content = "\n".join(current_block).strip()
            if content: blocks.append("\n".join(current_block))
            current_block = []
    if current_block: blocks.append("\n".join(current_block))
    return blocks

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

    prompt = f"""
SYSTEM_MODE: ACADEMIC_TRANSLATOR
**TASK:** Translate to Traditional Chinese (Taiwan).

**OUTPUT FORMAT:**
<<<ID_x>>>
[Chinese Translation]

**RULES:**
1. **Style:** Academic, formal.
2. **SKIP:** If block is Table, Code, `![]`, or Reference list, output: `[ORIGINAL]`
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

    print(f"🚀 Starting Batch Translation (Max: {BATCH_SIZE_LIMIT} chars)...", file=sys.stderr)

    for i, block in enumerate(original_blocks):
        current_batch.append(block)
        current_batch_len += len(block)

        if current_batch_len >= BATCH_SIZE_LIMIT or i == len(original_blocks) - 1:
            print(f"📤 Sending Batch: {batch_start_index} to {i}...", file=sys.stderr)
            translations = translate_batch(current_batch, batch_start_index)
            
            for j, orig_block in enumerate(current_batch):
                global_idx = batch_start_index + j
                trans_text = translations.get(global_idx, "")
                
                if trans_text == "[ORIGINAL]" or not trans_text:
                    final_blocks.append(orig_block)
                else:
                    final_blocks.append(f"{orig_block}\n\n> {trans_text}")

            current_batch = []
            current_batch_len = 0
            batch_start_index = i + 1
            time.sleep(2)

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