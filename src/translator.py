import sys
import os
import re
import time
import logging
import shutil
import yaml
import datetime
import requests
from pathlib import Path
import fitz  # PyMuPDF
from dotenv import load_dotenv

# Define custom exception for Quota Exceeded
class QuotaExceededError(Exception):
    pass

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
    from google.api_core.exceptions import ResourceExhausted
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
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.client = None
        
        if self.api_key and SDK_AVAILABLE:
            self.client = genai.Client(api_key=self.api_key)
            self.logger.info(f"[SUCCESS] Gemini SDK Initialized (Model: {self.model_name})")
        else:
            self.logger.warning("[WARNING] GEMINI_API_KEY not found or SDK missing. Translation might fail.")

        if self.groq_api_key:
            self.logger.info("[SUCCESS] Groq Fallback Initialized")
        else:
            self.logger.warning("[WARNING] GROQ_API_KEY not found. Fallback disabled.")

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
            self.logger.error("[ERROR] Marker modules missing.")

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
            self.logger.info(f"[INFO] Already in dedicated folder: {parent_dir}")
            return pdf_path
            
        # Create dedicated folder
        new_dir = parent_dir / paper_name
        new_dir.mkdir(exist_ok=True)
        
        new_pdf_path = new_dir / pdf_path.name
        
        # Move PDF
        try:
            if pdf_path.exists():
                shutil.move(str(pdf_path), str(new_pdf_path))
                self.logger.info(f"[INFO] Organized paper into: {new_dir}")
                return new_pdf_path
        except Exception as e:
            self.logger.warning(f"Failed to organize folder: {e}")
            return pdf_path # Return original on failure
            
        return new_pdf_path

    def _save_and_link_images(self, text, rendered_images, output_dir):
        """
        Saves images from Marker's native output and updates links.
        Ensures 1-to-1 mapping. Removes broken links.
        """
        images_dir = output_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        
        # Save all available images
        saved_images = set()
        for filename, image_data in rendered_images.items():
            image_path = images_dir / filename
            try:
                # Marker usually returns PIL images
                if hasattr(image_data, 'save'):
                    image_data.save(image_path)
                else:
                    with open(image_path, "wb") as f:
                        f.write(image_data)
                saved_images.add(filename)
            except Exception as e:
                self.logger.warning(f"[WARNING] Failed to save image {filename}: {e}")

        # Fix links in text
        def replace_link(match):
            alt_text = match.group(1)
            image_ref = match.group(2)
            filename = Path(image_ref).name
            
            if filename in saved_images:
                return f"![{alt_text}](images/{filename})"
            else:
                self.logger.warning(f"[WARNING] Image referenced but missing: {filename}")
                return "" # Remove broken link

        # Regex to find markdown images: ![alt](path)
        new_text = re.sub(r'!\[(.*?)\]\((.*?)\)', replace_link, text)
        return new_text

    def _clean_text_noise(self, text):
        # [Fix] Remove Marker artifacts globally
        text = re.sub(r'<span id="page-\d+-\d+"></span>', '', text)
        text = re.sub(r'\[(\d+)\]\(#page-\d+-\d+\)', r'[\1]', text) # Keep citation number, remove link
        
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
            
            # [Fix] Explicitly separate headers
            is_header = stripped.startswith("#") and not in_code and not in_math
            
            if is_header:
                if current_block:
                    content = "\n".join(current_block).strip()
                    if content: blocks.append("\n".join(current_block))
                    current_block = []
                blocks.append(line)
                continue

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

    def _call_groq_api(self, prompt):
        """Fallback to Groq API"""
        if not self.groq_api_key:
            return None

        self.logger.info("[INFO] Switching to Groq API (Fallback)...")
        headers = {
            "Authorization": f"Bearer {self.groq_api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "llama-3.3-70b-versatile",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3
        }
        
        for attempt in range(3):
            try:
                response = requests.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers=headers,
                    json=data,
                    timeout=60
                )
                response.raise_for_status()
                result = response.json()
                return result['choices'][0]['message']['content']
            except Exception as e:
                self.logger.warning(f"[WARNING] Groq API Attempt {attempt+1} failed: {e}")
                time.sleep(2)
        self.logger.error(f"[ERROR] Groq API Failed after 3 attempts.")
        return None

    def _call_gemini_sdk(self, prompt):
        """Call Gemini via Google GenAI SDK with Groq Fallback"""
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
                    
            except ResourceExhausted:
                self.logger.warning(f"[WARNING] Gemini Quota Exceeded (Attempt {attempt+1})")
                
                # Check if we should fallback to Groq immediately
                groq_result = self._call_groq_api(prompt)
                if groq_result:
                    return groq_result
                
                # If Groq also fails or is not configured, re-raise as internal QuotaExceededError
                # only if we want to stop completely. 
                # But here we are in a retry loop.
                # If Groq failed, we might want to wait and retry Gemini? 
                # Or just give up? 
                # Let's give up on this batch to fail fast if we really have no quota.
                self.logger.error("[ERROR] Both Gemini and Groq failed (Quota/Error).")
                raise QuotaExceededError("Gemini Quota Exceeded and Groq Fallback failed/unavailable.")

            except Exception as e:
                # Check for 429 in string representation if not caught by ResourceExhausted
                if "429" in str(e) or "ResourceExhausted" in str(e):
                    self.logger.warning(f"[WARNING] Gemini Quota Exceeded (429 detected in generic error)")
                    groq_result = self._call_groq_api(prompt)
                    if groq_result:
                        return groq_result
                    raise QuotaExceededError("Gemini Quota Exceeded and Groq Fallback failed/unavailable.")
                
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
            full_text = self._save_and_link_images(full_text, rendered.images, output_dir)
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
                        # [Fix] Properly quote multi-line translations
                        quoted_trans = "\n> ".join(trans_text.splitlines())
                        final_blocks.append(f"{block}\n\n> {quoted_trans}")

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
            
        self.logger.info(f"[SUCCESS] Translated saved to: {output_path}")

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
4. **Completeness:** Translate every sentence fully. Do not summarize or omit contributions.
5. **Structure:** Maintain all original markdown structure (lists, bolding, etc.).

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
