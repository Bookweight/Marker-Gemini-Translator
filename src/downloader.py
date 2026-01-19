import os
import re
import logging
import requests
import subprocess
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

class PaperDownloader:
    """
    論文下載與翻譯控制器
    功能:
    1. 下載 PDF (優先使用 Semantic Scholar 連結，失敗則嘗試 arXiv)
    2. 呼叫 PowerShell 腳本進行自動翻譯
    """
    def __init__(self, config: Dict[str, Any]):
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        # 設定路徑
        vault_raw = self.config['obsidian']['vault_path']
        unclassified_rel = self.config['obsidian'].get('unclassified_folder', 'Papers/unclassified')
        self.target_dir = Path(vault_raw) / unclassified_rel
        self.target_dir.mkdir(parents=True, exist_ok=True)
        
        # PowerShell 腳本路徑
        project_root = Path(__file__).resolve().parent.parent
        self.script_path = project_root / "scripts" / "Translate-And-Metadata.ps1"

    def process_papers(self, papers: List[Dict[str, Any]]):
        """主流程：遍歷論文並執行下載與翻譯"""
        if not papers:
            return

        self.logger.info(f"啟動自動下載流程: 準備處理 {len(papers)} 篇論文...")
        
        for paper in papers:
            title = paper.get('title', 'Untitled')
            
            # --- 下載策略核心 ---
            url = self._get_download_url(paper)
            
            if not url:
                self.logger.warning(f"跳過下載 (找不到 S2 連結或 arXiv ID): {title}")
                continue
            
            try:
                # 2. 下載 PDF
                pdf_path = self._download_pdf(url, title)
                if pdf_path:
                    # 3. 執行翻譯
                    self._run_translation_script(pdf_path)
                    
                    # 避免對 arXiv 請求過於頻繁 (禮貌性延遲)
                    if "arxiv.org" in url:
                        time.sleep(3) 
                        
            except Exception as e:
                self.logger.error(f"處理失敗 [{title}]: {e}")

    def _get_download_url(self, paper: Dict[str, Any]) -> Optional[str]:
        """
        解析下載連結：
        1. 優先使用 openAccessPdf (官方 Open Access)
        2. 如果沒有，嘗試組裝 arXiv 連結
        """
        # 策略 1: Semantic Scholar 直接提供的連結
        pdf_info = paper.get('openAccessPdf')
        if pdf_info and pdf_info.get('url'):
            return pdf_info['url']
            
        # 策略 2: arXiv 救援機制
        external_ids = paper.get('externalIds') or {}
        arxiv_id = external_ids.get('ArXiv')
        
        if arxiv_id:
            # 構建官方 PDF 連結
            arxiv_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
            self.logger.info(f"觸發 arXiv 救援機制: {arxiv_url}")
            return arxiv_url
            
        return None

    def _download_pdf(self, url: str, title: str) -> Path:
        """下載單篇 PDF"""
        # 清理檔名
        safe_title = re.sub(r'[\\/*?:"<>|]', "", title)
        safe_title = safe_title[:100].strip()
        filename = f"{safe_title}.pdf"
        file_path = self.target_dir / filename
        
        # 如果檔案已存在，跳過
        if file_path.exists():
            self.logger.info(f"PDF 已存在，跳過下載: {filename}")
            return file_path
            
        self.logger.info(f"下載中: {filename}")
        
        try:
            # arXiv 需要類似瀏覽器的 User-Agent，否則會拒絕連線
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            response = requests.get(url, headers=headers, stream=True, timeout=60)
            response.raise_for_status()
            
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    
            self.logger.info(f"✅ 下載完成: {file_path}")
            return file_path
        except Exception as e:
            self.logger.error(f"下載請求失敗 ({url}): {e}")
            return None

    def _run_translation_script(self, pdf_path: Path):
        """呼叫 PowerShell 進行翻譯"""
        if not self.script_path.exists():
            self.logger.error(f"找不到翻譯腳本: {self.script_path}")
            return

        # 檢查翻譯檔是否已存在
        zh_md_path = pdf_path.with_suffix('.zh.md')
        if zh_md_path.exists():
             self.logger.info(f"翻譯檔已存在，跳過翻譯: {zh_md_path.name}")
             return

        self.logger.info(f"呼叫 Gemini 進行翻譯: {pdf_path.name}...")
        
        # 構建指令
        cmd = [
            "powershell", 
            "-NoProfile", 
            "-ExecutionPolicy", "Bypass", 
            "-File", str(self.script_path), 
            "-InputFile", str(pdf_path)
        ]
        
        # 執行並捕捉輸出
        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
        
        if result.returncode == 0:
            self.logger.info(f"翻譯成功！輸出至: {zh_md_path.name}")
        else:
            self.logger.error(f"翻譯腳本執行失敗:\n[STDOUT]:\n{result.stdout}\n[STDERR]:\n{result.stderr}")