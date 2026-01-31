import sys
import os
import re
import time
import logging
import shutil
import yaml
import datetime
from pathlib import Path
import fitz  # PyMuPDF
from dotenv import load_dotenv

# Try importing marker
try:
    from marker.converters.pdf import PdfConverter
    from marker.models import create_model_dict
    from marker.output import text_from_rendered
    from marker.config.parser import ConfigParser
    MARKER_AVAILABLE = True
except ImportError:
    MARKER_AVAILABLE = False

# Try importing google-genai
try:
    from google import genai
    from google.genai import types
    SDK_AVAILABLE = True
except ImportError:
    SDK_AVAILABLE = False

class PaperTranslator:
    def __init__(self, config):
        self.logger = logging.getLogger("Translator")
        self.config = config
        self.model_name = config.get('translation', {}).get('model', 'gemini-2.0-flash-exp')
        self.batch_size = config.get('translation', {}).get('batch_size', 3000)
        
        # Initialize API Client
        load_dotenv()
        self.api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.client = None
        
        if self.api_key and SDK_AVAILABLE:
            self.client = genai.Client(api_key=self.api_key)
            self.logger.info(f"✅ Gemini SDK Initialized (Model: {self.model_name})")
        else:
            self.logger.warning("⚠️ GEMINI_API_KEY not found or SDK missing. Translation might fail.")

        # Initialize Marker
        if MARKER_AVAILABLE:
            config_dict = {
                "output_format": "markdown",
                "disable_image_extraction": False,
                "disable_table_extraction": False,
                "paginate_output": False
            }
            config_parser = ConfigParser(config_dict)
            self.converter = PdfConverter(
                config=config_parser.generate_config_dict(),
                artifact_dict=create_model_dict(),
                processor_list=config_parser.get_processors(),
                renderer=config_parser.get_renderer()
            )
        else:
            self.logger.error("❌ Marker modules missing.")

    def _clean_filename(self, name):
        return re.sub(r'[^a-zA-Z0-9_.-]', '_', name)

    def organize_paper_folder(self, pdf_path):
        """
        Move PDF into a dedicated folder with the same name.
        Returns the new path of the PDF.
        """
        pdf_path = Path(pdf_path).resolve()
        paper_name = pdf_path.stem
        parent_dir = pdf_path.parent
        
        # Check if already in a dedicated folder (Parent name == Paper name)
        if parent_dir.name == paper_name:
            self.logger.info(f"📂 Already in dedicated folder: {parent_dir}")
            return pdf_path
            
        # Create dedicated folder
        new_dir = parent_dir / paper_name
        new_dir.mkdir(exist_ok=True)
        
        new_pdf_path = new_dir / pdf_path.name
        
        # Move PDF
        try:
            if pdf_path.exists():
                shutil.move(str(pdf_path), str(new_pdf_path))
                self.logger.info(f"📦 Organized paper into: {new_dir}")
                return new_pdf_path
        except Exception as e:
            self.logger.warning(f"Failed to organize folder: {e}")
            return pdf_path # Return original on failure
            
        return new_pdf_path

    def _detect_content_boundaries(self, page):
        """V26 Logic"""
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

    def _extract_images(self, pdf_path, output_dir):
        """V26 Logic"""
        images_dir = output_dir / "images"
        images_dir.mkdir(exist_ok=True)
        
        doc = fitz.open(pdf_path)
        extracted_map = {} 
        
        for page_index in range(len(doc)):
            page = doc[page_index]
            image_list = page.get_images(full=True)
            extracted_map[page_index] = []
            top_limit, bottom_limit = self._detect_content_boundaries(page)
            
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
                        image_path = images_dir / image_name
                        with open(image_path, "wb") as f:
                            f.write(image_bytes)
                        extracted_map[page_index].append({
                            "filename": image_name, "is_noise": is_noise
                        })
                    except Exception:
                        pass
        return extracted_map

    def _clean_text_noise(self, text):
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

    def _force_normalize_headers(self, text):
        """V23 Logic"""
        lines = text.split('\n')
        new_lines = []
        h2_pattern = re.compile(r'^[*#]*\s*(\d+\.?\s+[A-Z].*?)[*#]*$') 
        h3_pattern = re.compile(r'^[*#]*\s*(\d+\.\d+\.?\s+.*?)[*#]*$')
        roman_pattern = re.compile(r'^[*#]*\s*([IVX]+\.?\s+[A-Z].*?)[*#]*$')
        special_headers = ["Abstract", "References", "Introduction", "Conclusion"]

        for line in lines:
            clean_line = line.strip()
            if not clean_line:
                new_lines.append(line)
                continue
            
            if h3_pattern.match(clean_line):
                match = h3_pattern.match(clean_line)
                content = re.sub(r'^[*#]+\s*', '', match.group(0))
                new_lines.append(f"### {content}")
            elif h2_pattern.match(clean_line):
                 match = h2_pattern.match(clean_line)
                 content = re.sub(r'^[*#]+\s*', '', match.group(0))
                 new_lines.append(f"## {content}")
            elif roman_pattern.match(clean_line):
                 match = roman_pattern.match(clean_line)
                 content = re.sub(r'^[*#]+\s*', '', match.group(0))
                 new_lines.append(f"## {content}")
            else:
                 is_special = False
                 for h in special_headers:
                     if re.match(rf'^[*#]*\s*{h}', clean_line, re.IGNORECASE):
                        clean_content = re.sub(r'[*#]', '', clean_line).strip()
                        new_lines.append(f"## {clean_content}")
                        is_special = True
                        break
                 if not is_special:
                     new_lines.append(line)
            
            last_line = new_lines[-1]
            if last_line.startswith("## ") or last_line.startswith("### "):
                header_match = re.match(r'^(#+\s+)(.*?)([\*—–-]{3,}|:|—)(.*)', last_line)
                if header_match:
                     heading_text = header_match.group(2).strip()
                     body_text = header_match.group(4).strip()
                     if len(heading_text) < 50:
                         new_lines.pop()
                         new_lines.append(f"{header_match.group(1)}{heading_text}")
                         new_lines.append("") 
                         new_lines.append(body_text)

        return '\n'.join(new_lines)

    def _inject_images(self, text, image_map):
        """V26 Logic with Fixed Regex"""
        pattern = re.compile(r'!\[(.*?)\]\(.*?_page_(\d+).*?\)')
        parts = []
        last_end = 0
        page_counter = {} 
        
        for match in pattern.finditer(text):
            parts.append(text[last_end:match.start()])
            alt = match.group(1) or "Figure"
            page_idx = int(match.group(2))
            
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

    def _split_into_blocks(self, text):
        lines = text.split('\n')
        blocks = []
        current_block = []
        in_code = False
        in_math = False
        in_table = False
        
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

    def _is_translatable(self, block):
        # [Fix-V29] Remove HTML tags before checking (e.g. <span id="..."> $$...$$)
        clean_block = re.sub(r'<[^>]+>', '', block).strip()
        
        if clean_block.startswith("$$"): return False
        if block.strip().startswith("```"): return False
        if re.match(r'^!\[.*?\]\(.*?\)$', block.strip()): return False
        if "|" in block and "-|-" in block: return False
        if re.match(r'^\(\d+(\.\d+)?\)$', block.strip()): return False
        if "vector/text - not extracted" in block.lower(): return False
        return True

    def _call_gemini_sdk(self, prompt):
        """Call Gemini via Google GenAI SDK"""
        if not self.client: return None
        
        for attempt in range(3):
            try:
                # Use Google GenAI SDK
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt
                )
                if response.text:
                    return response.text
            except Exception as e:
                self.logger.warning(f"API Error (Attempt {attempt+1}): {e}")
                time.sleep(2 * (attempt + 1))
        return None

    def _insert_frontmatter(self, text, pdf_path):
        # Restore PS1 Metadata Logic
        if text.startswith("---"): return text
        
        try:
            abs_path = pdf_path.resolve()
            parts = abs_path.parts
            field = pdf_path.parent.name # Fallback
            
            if "Papers" in parts:
                idx = parts.index("Papers")
                # field is the folder strictly inside "Papers" (e.g. Papers/Time Series/...)
                if idx + 1 < len(parts):
                    field = parts[idx + 1]
            
            # Formatting
            field = field.replace(' ', '_')
            
            # Tag: Sentence Case (Time_Series -> Time_series)
            tag = field
            tag_parts = field.split('_')
            if len(tag_parts) > 1:
                p0 = tag_parts[0].title()
                others = [p.lower() for p in tag_parts[1:]]
                tag = "_".join([p0] + others)
                
            title = pdf_path.stem.replace('.zh', '')
            date_str = datetime.date.today().isoformat()
            pdf_name = pdf_path.name
            
            frontmatter = f"""---
title: "{title}"
field: "{field}"
status: "Imported"
created_date: {date_str}
pdf_link: "[[{pdf_name}]]"
tags: [paper, {tag}]
---

"""
            return frontmatter + text
        except Exception as e:
            self.logger.warning(f"Metadata error: {e}")
            return text

    def translate_paper(self, pdf_path, output_path=None):
        pdf_path = Path(pdf_path)
        if not output_path:
            output_path = pdf_path.with_suffix('.zh.md')
        else:
            output_path = Path(output_path)
            
        # 1. Temp file handling for Long Paths / Stability
        temp_dir = Path(os.getenv('TEMP'))
        temp_pdf = temp_dir / "process_paper.pdf"
        try:
            shutil.copy2(pdf_path, temp_pdf)
        except Exception as e:
            self.logger.error(f"Failed to copy to temp: {e}")
            return

        # 2. Extract with Marker
        output_dir = output_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            rendered = self.converter(str(temp_pdf))
            text_val = text_from_rendered(rendered)
            full_text = text_val[0] if isinstance(text_val, tuple) and len(text_val) >= 2 else str(text_val)
            
            # Post-Process Text
            smart_image_map = self._extract_images(str(temp_pdf), output_dir)
            full_text = self._inject_images(full_text, smart_image_map)
            full_text = self._clean_text_noise(full_text)
            full_text = self._force_normalize_headers(full_text)
            
        except Exception as e:
            self.logger.error(f"Marker conversion failed: {e}")
            return
        
        # 3. Translation Pipeline
        ref_match = re.search(r'^##\s+(References|Bibliography)', full_text, re.MULTILINE | re.IGNORECASE)
        pre_text = full_text
        post_text = ""
        if ref_match:
            split_idx = ref_match.start()
            pre_text = full_text[:split_idx]
            post_text = full_text[split_idx:]
            
        blocks = self._split_into_blocks(pre_text)
        final_blocks = []
        
        # Batching
        current_batch = []
        current_batch_ids = []
        current_len = 0
        all_translations = {}
        
        for i, block in enumerate(blocks):
            if not self._is_translatable(block):
                continue
                
            block_len = len(block)
            if current_len + block_len > self.batch_size:
                # Send Batch
                self._process_batch(current_batch, current_batch_ids, all_translations)
                current_batch = []
                current_batch_ids = []
                current_len = 0
            
            current_batch.append(block)
            current_batch_ids.append(i)
            current_len += block_len
            
        if current_batch:
            self._process_batch(current_batch, current_batch_ids, all_translations)
            
        # Reconstruct
        for i, block in enumerate(blocks):
            trans_text = all_translations.get(i, "")
            
            if not trans_text or trans_text == "[ORIGINAL]":
                final_blocks.append(block)
            else:
                if block.strip().startswith("#"):
                     # Clean headers
                     clean_trans = re.sub(r'^#+\s*', '', trans_text).replace('\n', ' ').strip()
                     clean_trans = re.sub(r'<[^>]+>', '', clean_trans).strip()
                     clean_trans = re.sub(r'^([壹貳參肆伍陸柒捌玖拾甲乙丙丁戊]+[、.．])\s*', '', clean_trans).strip()
                     final_blocks.append(f"{block.strip()} - {clean_trans}")
                else:
                    # [Fix] Deduplication
                    if block.strip() == trans_text.strip():
                        final_blocks.append(block)
                    else:
                        final_blocks.append(f"{block}\n\n> {trans_text}")

        # Final Assembly
        body = "\n\n".join(final_blocks)
        result_text = body
        
        if post_text:
            # Fix references
            post_text = re.sub(r'\n\s*-\s+(?!\[\d+\])(.*)', r' \1', post_text)
            post_text = re.sub(r'(<span[^>]*>)?\s*\[\d+\]', r'\n\n\g<0>', post_text)
            post_text = re.sub(r'^\s*-\s*\[(\d+)\]', r'[\1]', post_text, flags=re.MULTILINE)
            post_text = re.sub(r'\n+\s*(\[\d+\])', r'\n\n\1', post_text)
            result_text += "\n\n" + post_text
            
        # [Fix] Insert Metadata
        result_text = self._insert_frontmatter(result_text, pdf_path)
            
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(result_text)
            
        self.logger.info(f"✅ Translated saved to: {output_path}")

    def _process_batch(self, batch, ids, results):
        if not batch: return
        prompt_text = ""
        for idx, text in zip(ids, batch):
            prompt_text += f"<<<ID_{idx}>>>\n{text}\n\n"
            
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
"""
        response_text = self._call_gemini_sdk(prompt)
        if response_text:
            matches = re.finditer(r'<<<ID_(\d+)>>>\s*(.*?)(?=(<<<ID_|\Z))', response_text, re.DOTALL)
            for match in matches:
                mid = int(match.group(1))
                content = match.group(2).strip()
                results[mid] = content

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/translator.py <pdf_path> [output_path]")
        sys.exit(1)
        
    logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
    
    # Load config from root
    config_path = Path("config.yaml")
    config = {}
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    else:
        print("⚠️ config.yaml not found, using defaults.")
        
    translator = PaperTranslator(config)
    input_file = Path(sys.argv[1])
    
    # [Fix] Organize into folder first
    if input_file.exists():
        input_file = translator.organize_paper_folder(input_file)
    
    output_file = Path(sys.argv[2]) if len(sys.argv) >= 3 else None
    
    translator.translate_paper(input_file, output_file)
