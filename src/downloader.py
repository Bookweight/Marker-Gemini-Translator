import logging
import requests
import time
import re
from pathlib import Path
from pypdf import PdfReader, PdfWriter
from typing import List, Dict, Any
from src.translator import PaperTranslator, QuotaExceededError

class PaperDownloader:
    def __init__(self, config: Dict[str, Any]):
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.vault_path = Path(config['obsidian']['vault_path'])
        unclassified_rel = config['obsidian'].get('unclassified_folder', 'Papers/unclassified')
        self.default_dir = self.vault_path / unclassified_rel
        self.default_dir.mkdir(parents=True, exist_ok=True)
        self.papers_root = self.default_dir.parent
        self.folder_map = {
            "Time Series": ["Time Series", "Forecasting", "Anomaly Detection"],
            "Computer Vision": ["Computer Vision", "Image", "Object Detection", "Segmentation"],
            "Natural Language Processing": ["NLP", "Language Model", "Text", "LLM"],
            "Graph Neural Networks": ["Graph", "GNN", "Link Prediction"],
            "Recommendation System": ["Recommendation", "Recommender"],
            "Deep Learning": ["Deep Learning", "Neural Network"],
            # 您現有的其他資料夾
            "LLM": ["Large Language Model", "LLM"],
            "Knowledge Graph": ["Knowledge Graph"],
            "Sentiment Analysis": ["Sentiment"],
            "Point Cloud Registration": ["Point Cloud"],
            "Hearing Loss Simulation": ["Hearing", "Audio"],
            "Database": ["Database"]
        }
        # Initialize Translator
        self.translator = PaperTranslator(config)
        self.skip_translation = False
        
    def _determine_save_dir(self, paper: Dict[str, Any]) -> Path:
        """根據論文 Metadata 決定存檔資料夾"""
        
        # 1. 準備檢查的文字 (標題 + 領域標籤)
        title = paper.get('title', '').lower()
        fields = [f.lower() for f in (paper.get('fieldsOfStudy') or [])]
        # 有些論文只有 s2FieldsOfStudy
        if not fields:
            s2_fields = paper.get('s2FieldsOfStudy') or []
            fields = [f['category'].lower() for f in s2_fields]
        
        text_to_check = title + " " + " ".join(fields)

        # 2. 比對關鍵字
        for folder_name, keywords in self.folder_map.items():
            for kw in keywords:
                if kw.lower() in text_to_check:
                    target_dir = self.papers_root / folder_name
                    # 如果資料夾不存在，自動建立 (或視需求決定是否建立)
                    if not target_dir.exists():
                        # 這裡選擇自動建立，確保分類成功
                        target_dir.mkdir(parents=True, exist_ok=True)
                    return target_dir
        
        # 3. 沒對應到 -> 回傳 unclassified
        return self.default_dir
    
    def _compress_pdf(self, input_path: Path) -> Path:
        """
        [新增] 簡單的 PDF 壓縮/重寫邏輯
        如果檔案超過 20MB，嘗試重寫以減少 metadata 或無用數據。
        注意：pypdf 的壓縮能力有限，若要強力壓縮圖片需要用 ghostscript，
        但這能解決部分格式臃腫的問題。
        """
        file_size_mb = input_path.stat().st_size / (1024 * 1024)
        if file_size_mb < 20:
            return input_path
            
        self.logger.warning(f"[WARNING] File too large ({file_size_mb:.2f} MB), attempting to compress: {input_path.name}")
        
        try:
            reader = PdfReader(input_path)
            writer = PdfWriter()
            
            for page in reader.pages:
                writer.add_page(page)
                
            # 加入壓縮參數
            for page in writer.pages:
                page.compress_content_streams()  # 壓縮內容流
                
            temp_output = input_path.with_suffix('.compressed.pdf')
            with open(temp_output, "wb") as f:
                writer.write(f)
                
            # 檢查是否真的變小了
            new_size = temp_output.stat().st_size / (1024 * 1024)
            if new_size < 20:
                self.logger.info(f"[SUCCESS] Compression successful: {new_size:.2f} MB")
                input_path.unlink() # 刪除原檔
                temp_output.rename(input_path) # 取代原檔
                return input_path
            else:
                self.logger.warning(f"[WARNING] Still too large after compression ({new_size:.2f} MB), translation may fail.")
                temp_output.unlink()
                return input_path
                
        except Exception as e:
            self.logger.error(f"壓縮過程發生錯誤: {e}")
            return input_path

    def _download_pdf(self, url: str, title: str, target_dir: Path) -> Path: # [修改] 增加 target_dir 參數
        """下載單篇 PDF"""
        safe_title = re.sub(r'[\\/*?:"<>|]', "", title)[:100].strip()
        filename = f"{safe_title}.pdf"
        file_path = target_dir / filename # [修改] 使用傳入的資料夾

        if file_path.exists():
            if file_path.stat().st_size < 2048:
                self.logger.warning(f"發現無效舊檔 ({filename})，刪除重試...")
                file_path.unlink()
            else:
                self.logger.info(f"[SKIP] PDF exists ({target_dir.name}), skipping download")
                return file_path
            
        self.logger.info(f"[DOWNLOADING] ({target_dir.name}): {filename}")
        
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            }
            response = requests.get(url, headers=headers, stream=True, timeout=60)
            response.raise_for_status()
            
            content_type = response.headers.get('Content-Type', '').lower()
            if 'application/pdf' not in content_type and 'binary/octet-stream' not in content_type:
                self.logger.warning(f"下載內容非 PDF: {url}")
                return None

            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            if file_path.stat().st_size < 2048:
                file_path.unlink()
                return None

            self.logger.info(f"下載完成")
            
            if file_path and file_path.exists():
                self._compress_pdf(file_path)
            return file_path

        except Exception as e:
            self.logger.warning(f"下載失敗: {e}")
            if file_path.exists():
                file_path.unlink()
            return None

    def _run_translation_script(self, pdf_path: Path, paper_metadata: Dict):
        """Invoke Python Translator directly"""
        zh_md_path = pdf_path.with_suffix('.zh.md')
        if zh_md_path.exists():
            self.logger.info(f"[SKIP] Translation exists: {zh_md_path.name}")
            return
            
        # [Fix] Organize Folder First (Restore Structure)
        try:
            pdf_path = self.translator.organize_paper_folder(pdf_path)
            # Update output path based on new location
            zh_md_path = pdf_path.with_suffix('.zh.md')
        except Exception as e:
            self.logger.warning(f"Organization step skipped: {e}")

        self.logger.info(f"[TRANSLATING] Starting Native Python Translation: {title}...")
        try:
            self.translator.translate_paper(pdf_path, zh_md_path)
        except QuotaExceededError:
            self.logger.error("[ERROR] Translation Quota Exceeded! Switching to 'Download Only' mode.")
            raise  # Re-raise to be caught in process_papers
        except Exception as e:
            self.logger.error(f"Translation Crash: {e}", exc_info=True)

    def process_papers(self, papers: List[Dict[str, Any]]):
        """主流程"""
        for paper in papers:
            title = paper.get('title', 'Untitled')
            target_dir = self._determine_save_dir(paper)
            pdf_path = None
            
            # 2. 下載邏輯 (傳入 target_dir)
            pdf_info = paper.get('openAccessPdf')
            if pdf_info and pdf_info.get('url'):
                pdf_path = self._download_pdf(pdf_info['url'], title, target_dir)
            
            if not pdf_path:
                external_ids = paper.get('externalIds') or {}
                arxiv_id = external_ids.get('ArXiv')
                if arxiv_id:
                    arxiv_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
                    pdf_path = self._download_pdf(arxiv_url, title, target_dir)

            # 3. 翻譯邏輯 (修正參數傳遞)
            if pdf_path:
                if self.skip_translation:
                    self.logger.info(f"[SKIP] Skipping translation (Quota Exceeded): {title}")
                else:
                    try:
                        # [FIXED] 這裡補上 paper 參數
                        self._run_translation_script(pdf_path, paper)
                    except QuotaExceededError:
                        self.skip_translation = True
                        self.logger.warning("[WARNING] Global Translation Quota Hit. Disabling translation for remaining papers.")