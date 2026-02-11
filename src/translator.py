import datetime
import logging
import os
import re
import shutil
import sys
import time
from pathlib import Path

import requests
import yaml
from dotenv import load_dotenv


# Define custom exception for Quota Exceeded
class QuotaExceededError(Exception):
    pass


# Try importing marker
try:
    from marker.config.parser import ConfigParser
    from marker.converters.pdf import PdfConverter
    from marker.models import create_model_dict
    from marker.output import text_from_rendered

    MARKER_AVAILABLE = True
except ImportError:
    MARKER_AVAILABLE = False

# Try importing google-genai
try:
    from google import genai  # type: ignore
    from google.api_core.exceptions import ResourceExhausted

    SDK_AVAILABLE = True
except ImportError:
    SDK_AVAILABLE = False


class PaperTranslator:
    def __init__(self, config):
        self.logger = logging.getLogger("Translator")
        self.config = config
        self.model_name = config.get("translation", {}).get(
            "model", "gemini-2.0-flash-exp"
        )
        self.batch_size = config.get("translation", {}).get("batch_size", 3000)

        # Initialize API Client
        load_dotenv()
        self.api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.client = None

        if self.api_key and SDK_AVAILABLE:
            self.client = genai.Client(api_key=self.api_key)
            self.logger.info(
                f"[SUCCESS] Gemini SDK Initialized (Model: {self.model_name})"
            )
        else:
            self.logger.warning(
                "[WARNING] GEMINI_API_KEY not found or SDK missing. Translation might fail."
            )

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
                "paginate_output": False,
            }
            config_parser = ConfigParser(config_dict)
            self.converter = PdfConverter(
                config=config_parser.generate_config_dict(),
                artifact_dict=create_model_dict(),
                processor_list=config_parser.get_processors(),
                renderer=config_parser.get_renderer(),
            )
        else:
            self.logger.error("[ERROR] Marker modules missing.")

    def _clean_filename(self, name):
        return re.sub(r"[^a-zA-Z0-9_.-]", "_", name)

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
            return pdf_path  # Return original on failure

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
                if hasattr(image_data, "save"):
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
                self.logger.warning(
                    f"[WARNING] Image referenced but missing: {filename}"
                )
                return ""  # Remove broken link

        # Regex to find markdown images: ![alt](path)
        new_text = re.sub(r"!\[(.*?)\]\((.*?)\)", replace_link, text)
        return new_text

    def _clean_text_noise(self, text, keep_page_tags=False):
        # [Fix] Remove Marker artifacts globally
        if not keep_page_tags:
            text = re.sub(r'<span id="page-\d+-\d+"></span>', "", text)
        text = re.sub(
            r"\[(\d+)\]\(#page-\d+-\d+\)", r"[\1]", text
        )  # Keep citation number, remove link

        lines = text.split("\n")

        cleaned_lines = []
        patterns = [
            r"^\s*\d+\s+Page\s+\d+\s+of\s+\d+",
            r"^\s*Page\s+\d+\s*$",
            r"^\s*arXiv:\d+\.\d+.*$",
            r"^\s*https?://doi\.org/.*$",
            r".*©.*Permission\s+to\s+make.*",
            r"^\s*Vol\.\s+\d+,\s+No\.\s+\d+.*$",
        ]
        compiled_patterns = [re.compile(p, re.IGNORECASE) for p in patterns]

        for line in lines:
            is_noise = False
            if len(line) < 100:
                for p in compiled_patterns:
                    if p.match(line):
                        is_noise = True
                        break
            if not is_noise:
                cleaned_lines.append(line)
        return "\n".join(cleaned_lines)

    def _force_normalize_headers(self, text):
        """V23 Logic"""
        lines = text.split("\n")
        new_lines = []
        h2_pattern = re.compile(r"^[*#]*\s*(\d+\.?\s+[A-Z].*?)[*#]*$")
        h3_pattern = re.compile(r"^[*#]*\s*(\d+\.\d+\.?\s+.*?)[*#]*$")
        roman_pattern = re.compile(r"^[*#]*\s*([IVX]+\.?\s+[A-Z].*?)[*#]*$")
        special_headers = ["Abstract", "References", "Introduction", "Conclusion"]

        for line in lines:
            clean_line = line.strip()
            if not clean_line:
                new_lines.append(line)
                continue

            if h3_pattern.match(clean_line):
                match = h3_pattern.match(clean_line)
                content = re.sub(r"^[*#]+\s*", "", match.group(0))
                new_lines.append(f"### {content}")
            elif h2_pattern.match(clean_line):
                match = h2_pattern.match(clean_line)
                content = re.sub(r"^[*#]+\s*", "", match.group(0))
                new_lines.append(f"## {content}")
            elif roman_pattern.match(clean_line):
                match = roman_pattern.match(clean_line)
                content = re.sub(r"^[*#]+\s*", "", match.group(0))
                new_lines.append(f"## {content}")
            else:
                is_special = False
                for h in special_headers:
                    if re.match(rf"^[*#]*\s*{h}", clean_line, re.IGNORECASE):
                        clean_content = re.sub(r"[*#]", "", clean_line).strip()
                        new_lines.append(f"## {clean_content}")
                        is_special = True
                        break
                if not is_special:
                    new_lines.append(line)

            last_line = new_lines[-1]
            if last_line.startswith("## ") or last_line.startswith("### "):
                header_match = re.match(
                    r"^(#+\s+)(.*?)([\*—–-]{3,}|:|—)(.*)", last_line
                )
                if header_match:
                    heading_text = header_match.group(2).strip()
                    body_text = header_match.group(4).strip()
                    if len(heading_text) < 50:
                        new_lines.pop()
                        new_lines.append(f"{header_match.group(1)}{heading_text}")
                        new_lines.append("")
                        new_lines.append(body_text)

        return "\n".join(new_lines)

    def _split_into_blocks(self, text):
        lines = text.split("\n")
        blocks = []
        current_block = []
        in_code = False
        in_math = False
        in_table = False

        for line in lines:
            stripped = line.strip()
            if stripped.startswith("```"):
                in_code = not in_code
            if stripped == "$$":
                in_math = not in_math
            if "|" in line and len(line) > 5:
                in_table = True
            elif stripped == "":
                in_table = False

            # [Fix] Explicitly separate headers
            is_header = stripped.startswith("#") and not in_code and not in_math

            if is_header:
                if current_block:
                    content = "\n".join(current_block).strip()
                    if content:
                        blocks.append("\n".join(current_block))
                    current_block = []
                blocks.append(line)
                continue

            current_block.append(line)

            if not in_table and not in_code and not in_math and stripped == "":
                content = "\n".join(current_block).strip()
                if content:
                    blocks.append("\n".join(current_block))
                current_block = []

        if current_block:
            blocks.append("\n".join(current_block))
        return blocks

    def _is_translatable(self, block):
        # [Fix-V29] Remove HTML tags before checking (e.g. <span id="..."> $$...$$)
        clean_block = re.sub(r"<[^>]+>", "", block).strip()

        if clean_block.startswith("$$"):
            return False
        if block.strip().startswith("```"):
            return False
        if re.match(r"^!\[.*?\]\(.*?\)$", block.strip()):
            return False
        if "|" in block and "-|-" in block:
            return False
        if re.match(r"^\(\d+(\.\d+)?\)$", block.strip()):
            return False
        if "vector/text - not extracted" in block.lower():
            return False
        return True

    def _call_groq_api(self, prompt):
        """Fallback to Groq API"""
        if not self.groq_api_key:
            return None

        self.logger.info("[INFO] Switching to Groq API (Fallback)...")
        headers = {
            "Authorization": f"Bearer {self.groq_api_key}",
            "Content-Type": "application/json",
        }
        data = {
            "model": "llama-3.3-70b-versatile",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
        }

        for attempt in range(3):
            try:
                response = requests.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers=headers,
                    json=data,
                    timeout=60,
                )
                response.raise_for_status()
                result = response.json()
                return result["choices"][0]["message"]["content"]
            except Exception as e:
                self.logger.warning(
                    f"[WARNING] Groq API Attempt {attempt+1} failed: {e}"
                )
                time.sleep(2)
        self.logger.error("[ERROR] Groq API Failed after 3 attempts.")
        return None

    def _call_gemini_sdk(self, prompt):
        """Call Gemini via Google GenAI SDK with Groq Fallback and Retry Backoff"""
        if not self.client:
            return None

        # [Fix] Exponential Backoff for Quota Limits
        base_wait = 20  # Seconds

        for attempt in range(3):
            try:
                # Use Google GenAI SDK
                response = self.client.models.generate_content(
                    model=self.model_name, contents=prompt
                )
                if response.text:
                    return response.text

            except ResourceExhausted:
                wait_time = base_wait * (attempt + 1)
                self.logger.warning(
                    f"[WARNING] Gemini Quota Exceeded (Attempt {attempt+1}). Waiting {wait_time}s..."
                )
                time.sleep(wait_time)

                # If last attempt, try Groq Fallback
                if attempt == 2:
                    self.logger.warning(
                        "[FALLBACK] Gemini failed after retries. Trying Groq..."
                    )
                    groq_result = self._call_groq_api(prompt)
                    if groq_result:
                        return groq_result

                    # If Groq fails too
                    raise QuotaExceededError(
                        "Gemini Quota Exceeded and Groq Fallback failed/unavailable."
                    )

            except Exception as e:
                # Check for 429 in string representation if not caught by ResourceExhausted
                if "429" in str(e) or "ResourceExhausted" in str(e):
                    wait_time = base_wait * (attempt + 1)
                    self.logger.warning(
                        f"[WARNING] Gemini Quota Exceeded (429 detected in generic error). Waiting {wait_time}s..."
                    )
                    time.sleep(wait_time)

                    if attempt == 2:
                        self.logger.warning(
                            "[FALLBACK] Gemini failed after retries. Trying Groq..."
                        )
                        groq_result = self._call_groq_api(prompt)
                        if groq_result:
                            return groq_result
                        raise QuotaExceededError(
                            "Gemini Quota Exceeded and Groq Fallback failed/unavailable."
                        )
                else:
                    self.logger.warning(f"API Error (Attempt {attempt+1}): {e}")
                    time.sleep(2 * (attempt + 1))

        return None

    def _insert_frontmatter(self, text, pdf_path):
        # Restore PS1 Metadata Logic
        if text.startswith("---"):
            return text

        try:
            abs_path = pdf_path.resolve()
            parts = abs_path.parts
            field = pdf_path.parent.name  # Fallback

            if "Papers" in parts:
                idx = parts.index("Papers")
                # field is the folder strictly inside "Papers" (e.g. Papers/Time Series/...)
                if idx + 1 < len(parts):
                    field = parts[idx + 1]

            # Formatting
            field = field.replace(" ", "_")

            # Tag: Sentence Case (Time_Series -> Time_series)
            tag = field
            tag_parts = field.split("_")
            if len(tag_parts) > 1:
                p0 = tag_parts[0].title()
                others = [p.lower() for p in tag_parts[1:]]
                tag = "_".join([p0] + others)

            title = pdf_path.stem.replace(".zh", "")
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

    def _heuristic_check_structure(self, text):
        """
        Scans text for layout anomalies (e.g. Roman numeral disorder).
        Returns a set of page numbers (integers) that are suspicious.
        """
        suspicious_pages = set()

        # 1. Split by Pages using Marker tags <span id="page-X-Y">
        # We assume X is the page number.
        page_pattern = re.compile(r'<span id="page-(\d+)-\d+"></span>')
        parts = page_pattern.split(text)
        # parts[0] is text before first page tag
        # parts[1] is page num, parts[2] is text of that page...

        # Roman Numeral Pattern (I, II, III ... X)
        # Regex to find top-level headers: ## I. INTRODUCTION
        roman_header = re.compile(r"^##\s+([IXV]+)\.\s+", re.MULTILINE)

        last_roman_val = 0

        # Map roman to int
        roman_map = {
            "I": 1,
            "II": 2,
            "III": 3,
            "IV": 4,
            "V": 5,
            "VI": 6,
            "VII": 7,
            "VIII": 8,
            "IX": 9,
            "X": 10,
        }

        current_page_num = 1

        # Iterate through pages
        # parts list: [pre_text, p_num_1, text_1, p_num_2, text_2, ...]
        if len(parts) > 1:
            # Skip parts[0]
            for i in range(1, len(parts), 2):
                try:
                    p_num = int(parts[i])
                except Exception:
                    p_num = current_page_num + 1

                content = parts[i + 1]

                # Check Roman Headers in this page
                matches = roman_header.finditer(content)
                for m in matches:
                    roman_str = m.group(1).upper()
                    val = roman_map.get(roman_str, 99)

                    if 0 < val < 20:  # Sanity check
                        # If we see a smaller number after a bigger number (e.g. III then II)
                        # And it's not a restart (like references? no references usually don't have numbers)
                        if val < last_roman_val:
                            self.logger.warning(
                                f"[CHECK] Found disorder: {roman_str} ({val}) after {last_roman_val} on Page {p_num}"
                            )
                            suspicious_pages.add(p_num)
                            suspicious_pages.add(p_num - 1)  # Add previous page context
                        last_roman_val = val

                current_page_num = p_num

        return suspicious_pages

    def _repair_structure_with_llm(self, text, suspicious_pages):
        """
        Sends pages to LLM to fix reading order.
        """
        if not self.client:
            self.logger.warning("Gemini SDK not available, skipping repair.")
            return text

        page_pattern = re.compile(r'(<span id="page-(\d+)-\d+"></span>)')

        # We need to reconstruct the text page by page to target specific ones
        # But regex split removes the delimiter. We want to keep it.
        # Let's verify if we can just split and look at page numbers.

        parts = page_pattern.split(text)
        # parts: [pre, tag1, p1, content1, tag2, p2, content2 ... ]

        reconstructed_parts = [parts[0]]

        i = 1
        while i < len(parts):
            # tag_full = parts[i]
            p_num_str = parts[i + 1]
            content = parts[i + 2]

            p_num = int(p_num_str)

            if p_num in suspicious_pages:
                self.logger.info(f"[REPAIR] Repairing Page {p_num}...")

                # Construct Prompt
                prompt = f"""
SYSTEM_MODE: LAYOUT_REPAIR
**TASK**: Fix the reading order of the following text.
**CONTEXT**: The text comes from a PDF with a double-column layout. The parser sometimes mixes up the reading order (e.g., merging columns incorrectly, placing headers at the wrong bottom/top of the page).
**INSTRUCTIONS**:
1. Reorder the text to make it logically fast-flowing and coherent.
2. **DO NOT TRANSLATE**. Keep ONLY English.
3. **DO NOT SUMMARIZE**. Keep all original content.
4. Return only the repaired Markdown text.

**INPUT TEXT**:
{content}
"""
                try:
                    repaired_content = self._call_gemini_sdk(prompt)
                    if repaired_content:
                        content = repaired_content + "\n"  # Add newline for safety
                    else:
                        self.logger.warning(
                            f"Repair failed for Page {p_num} (No response), keeping original."
                        )
                except Exception as e:
                    self.logger.error(f"Repair failed for Page {p_num}: {e}")

            # Reconstruct
            reconstructed_parts.append(
                f'<span id="page-{p_num}-0"></span>'
            )  # Simplified tag restoration
            reconstructed_parts.append(content)

            i += 3

        return "".join(reconstructed_parts)

    def translate_paper(self, pdf_path, output_path=None):
        pdf_path = Path(pdf_path)
        if not output_path:
            output_path = pdf_path.with_suffix(".zh.md")
        else:
            output_path = Path(output_path)

        # 1. Temp file handling for Long Paths / Stability
        temp_dir = Path(os.getenv("TEMP"))
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
            full_text = (
                text_val[0]
                if isinstance(text_val, tuple) and len(text_val) >= 2
                else str(text_val)
            )

            # Post-Process Text
            full_text = self._save_and_link_images(
                full_text, rendered.images, output_dir
            )

            # [Step 2.5] Structure Repair (Pre-processing Hybrid)
            # 1. Clean noise but KEEP page tags for analysis
            analyzable_text = self._clean_text_noise(full_text, keep_page_tags=True)

            # 2. Heuristic Check
            suspicious_pages = self._heuristic_check_structure(analyzable_text)

            if suspicious_pages:
                self.logger.warning(
                    f"[STRUCTURE] Detected suspicious layout on pages: {suspicious_pages}"
                )
                # 3. Targeted LLM Repair
                analyzable_text = self._repair_structure_with_llm(
                    analyzable_text, suspicious_pages
                )
                self.logger.info("[STRUCTURE] Repair completed.")

            # 4. Final Clean (Remove page tags)
            full_text = self._clean_text_noise(analyzable_text, keep_page_tags=False)

            # full_text = self._clean_text_noise(full_text) # Replaced by flow above
            full_text = self._force_normalize_headers(full_text)

        except Exception as e:
            self.logger.error(f"Marker conversion failed: {e}")
            return

        # 3. Translation Pipeline
        ref_match = re.search(
            r"^##\s+(References|Bibliography)", full_text, re.MULTILINE | re.IGNORECASE
        )
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
        all_batches = []

        for i, block in enumerate(blocks):
            if not self._is_translatable(block):
                continue

            block_len = len(block)
            if current_len + block_len > self.batch_size:
                all_batches.append((current_batch, current_batch_ids))
                current_batch = []
                current_batch_ids = []
                current_len = 0

            current_batch.append(block)
            current_batch_ids.append(i)
            current_len += block_len

        if current_batch:
            all_batches.append((current_batch, current_batch_ids))

        # Parallel Execution
        all_translations = {}
        translation_failed = False

        if all_batches:
            # [Debug] Log first 3 blocks to verify alignment
            first_batch, first_ids = all_batches[0]
            self.logger.info(f"[DEBUG] Batch 0 IDs: {first_ids}")
            for k in range(min(3, len(first_batch))):
                self.logger.info(
                    f"[DEBUG] ID_{first_ids[k]} Preview: {first_batch[k][:50]}..."
                )

            self.logger.info(
                f"Starting parallel translation for {len(all_batches)} batches (Max Workers: 2)..."
            )
            import concurrent.futures

            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                    future_to_idf = {
                        executor.submit(self._process_batch, batch, ids): (batch, ids)
                        for batch, ids in all_batches
                    }

                    for future in concurrent.futures.as_completed(future_to_idf):
                        try:
                            batch_results = future.result()
                            if batch_results:
                                all_translations.update(batch_results)
                        except QuotaExceededError as e:
                            self.logger.critical(
                                f"[CRITICAL] Translation aborted due to Quota Limits: {e}"
                            )
                            translation_failed = True
                            # Cancel pending futures
                            for f in future_to_idf:
                                f.cancel()
                            break
                        except Exception as e:
                            self.logger.error(f"Batch translation failed: {e}")

            except Exception as e:
                self.logger.error(f"Parallel execution error: {e}")
                translation_failed = True

        if translation_failed:
            self.logger.error(
                "[ABORTED] Translation process stopped due to critical errors. File NOT saved."
            )
            return

        # Reconstruct
        for i, block in enumerate(blocks):
            trans_text = all_translations.get(i, "")

            if not trans_text or trans_text == "[ORIGINAL]":
                final_blocks.append(block)
            else:
                if block.strip().startswith("#"):
                    # Clean headers
                    clean_trans = (
                        re.sub(r"^#+\s*", "", trans_text).replace("\n", " ").strip()
                    )
                    clean_trans = re.sub(r"<[^>]+>", "", clean_trans).strip()
                    clean_trans = re.sub(
                        r"^([壹貳參肆伍陸柒捌玖拾甲乙丙丁戊]+[、.．])\s*",
                        "",
                        clean_trans,
                    ).strip()
                    final_blocks.append(f"{block.strip()}\n\n{clean_trans}")
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
            # [Fix] Removed aggressive merging of list items
            # post_text = re.sub(r'\n\s*-\s+(?!\[\d+\])(.*)', r' \1', post_text)
            post_text = re.sub(r"(<span[^>]*>)?\s*\[\d+\]", r"\n\n\g<0>", post_text)
            post_text = re.sub(
                r"^\s*-\s*\[(\d+)\]", r"[\1]", post_text, flags=re.MULTILINE
            )
            post_text = re.sub(r"\n+\s*(\[\d+\])", r"\n\n\1", post_text)
            result_text += "\n\n" + post_text

        # [Fix] Insert Metadata
        result_text = self._insert_frontmatter(result_text, pdf_path)

        # [Fix-V30] Post-Processing Cleaning
        result_text = self._clean_metadata(result_text)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(result_text)

        self.logger.info(f"[SUCCESS] Translated saved to: {output_path}")

    def _clean_metadata(self, text):
        """
        Cleans up translation artifacts, PDF links, and HTML entities.
        """
        # 1. Remove PDF Internal Links: [Link](#page-8-0) or (#page-8-0)
        text = re.sub(r"\[(.*?)\]\(#page-\d+-\d+\)", r"\1", text)
        text = re.sub(r"\(#page-\d+-\d+\)", "", text)

        # 2. Remove "Cited on page X" noise
        # Pattern: (Cited on page [2](#page-1-3)) or just (Cited on page 2)
        text = re.sub(r"\s*\(Cited on page.*?\)", "", text)

        # 3. Fix HTML Entities
        replacements = {
            "&lt;": "<",
            "&gt;": ">",
            "&amp;": "&",
            "<sup>&</sup>lt;sup>": "<sup>",  # Specific artifact seen in bug report
            "&quot;": '"',
            "&apos;": "'",
        }
        for old, new in replacements.items():
            text = text.replace(old, new)

        # 4. Remove duplicate Author lines if adjacent (Header vs Content)
        # Simple heuristic: if line N is identical to line N-2 (ignoring md syntax)
        lines = text.split("\n")
        cleaned_lines = []
        for i, line in enumerate(lines):
            # Check if this line repeats a recent header
            if i > 1 and len(line.strip()) > 3:
                prev_header = lines[i - 2].strip().replace("#", "").strip()
                if line.strip() == prev_header:
                    continue  # Skip duplicate
            cleaned_lines.append(line)

        return "\n".join(cleaned_lines)

    def _process_batch(self, batch, ids):
        if not batch:
            return {}

        results = {}
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
6. **IDs:** You must return the exact same IDs as provided. Do not skip any ID. If a block is empty or just a symbol, return it as is.

**INPUT:**

{prompt_text}
"""
        try:
            response_text = self._call_gemini_sdk(prompt)
            if response_text:
                matches = re.finditer(
                    r"<<<ID_(\d+)>>>\s*(.*?)(?=(<<<ID_|\Z))", response_text, re.DOTALL
                )
                for match in matches:
                    mid = int(match.group(1))
                    content = match.group(2).strip()
                    if mid in ids:
                        results[mid] = content
                    else:
                        self.logger.warning(
                            f"[BATCH] Received unknown ID {mid}, ignoring."
                        )

            # [Validation] Check for missing IDs
            missing_ids = [i for i in ids if i not in results]
            if missing_ids:
                self.logger.warning(f"[BATCH] Missing IDs in response: {missing_ids}")

        except QuotaExceededError:
            raise  # Propagate up
        except Exception as e:
            self.logger.error(f"Error in batch processing: {e}")

        return results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/translator.py <pdf_path> [output_path]")
        sys.exit(1)

    logging.basicConfig(
        level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s"
    )

    # Load config from root
    config_path = Path("config.yaml")
    config = {}
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
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
