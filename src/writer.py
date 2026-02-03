import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

class ObsidianWriter:
    """
    負責將推薦結果寫入 Obsidian 每日筆記
    """
    def __init__(self, config: Dict[str, Any]):
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        # 設定路徑
        self.vault_path = Path(config['obsidian']['vault_path'])
        self.daily_rel_path = config['obsidian']['daily_folder']
        self.daily_folder = self.vault_path / self.daily_rel_path
        
        # 確保資料夾存在
        self.daily_folder.mkdir(parents=True, exist_ok=True)

    def write_recommendations(self, papers: List[Dict[str, Any]]) -> bool:
        """
        將推薦結果寫入每日筆記
        回傳: True (寫入成功/新檔案), False (檔案已存在)
        """
        today_str = datetime.now().strftime("%Y-%m-%d")
        filename = f"{today_str}.md"
        file_path = self.daily_folder / filename
        
        # 冪等性檢查：如果檔案已存在，不覆蓋
        if file_path.exists():
            self.logger.info(f"{filename} 已包含推薦清單，跳過寫入。")
            return False

        # --- 1. YAML Metadata (保留這個新功能) ---
        yaml_frontmatter = f"""---
        type: "daily_recommendation"
        date: "{today_str}"
        created_time: "{datetime.now().strftime('%H:%M')}"
        paper_count: {len(papers)}
        tags: [daily_rec, paper_reading]
        status: "To Read"
---

        """

        # --- 2. 筆記內容 (還原為經典格式) ---
        content = [yaml_frontmatter]
        
        # 標題區塊
        content.append(f"# {today_str}\n")
        content.append(f"## 每日論文推薦 ({today_str})")
        
        # 狀態區塊
        content.append(f"> [!INFO] System Status")
        content.append(f"> **Mode**: Classic First | **Source**: Semantic Scholar | **Items**: {len(papers)}\n")
        
        # 論文清單 (還原 Checkbox 風格)
        for p in papers:
            title = p.get('title', 'Untitled')
            paper_id = p.get('paperId', '')
            
            # 還原 Semantic Scholar 連結
            if paper_id:
                s2_url = f"https://www.semanticscholar.org/paper/{paper_id}"
            else:
                s2_url = p.get('url', '#')
            
            # 分數與引用
            # 注意：這裡將分數轉為整數以符合舊版觀感，或者保留小數點視您的舊版資料而定
            # 舊版範例為 5169 (整數)，這裡做個轉換
            try:
                score = int(p.get('final_score', 0))
            except:
                score = 0
                
            # 標籤處理
            fields = p.get('fieldsOfStudy') or ["ComputerScience"]
            # 將欄位轉為 Hashtag 格式 (例如 Computer Science -> #ComputerScience)
            tags_str = " ".join([f"#{f.replace(' ', '')}" for f in fields])
            
            # 建立卡片
            card = f"- [ ] **{title}**\n"
            card += f"    - **Type**: CS-Core | **Impact**: 🔥 {score}\n"
            card += f"    - **Tags**: {tags_str}\n"
            card += f"    - [Open on Semantic Scholar]({s2_url})\n"
            card += f"    - **Rating**: ( )"
            
            content.append(card)

        # --- 3. 寫入檔案 ---
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write("\n".join(content))
            self.logger.info(f"已建立新筆記並寫入推薦: {filename}")
            return True
        except Exception as e:
            self.logger.error(f"寫入筆記失敗: {e}")
            return False

    def update_daily_link(self, paper_title: str, file_stem: str):
        """
        當論文下載/翻譯完成後，回頭更新每日筆記加上連結
        """
        import re
        today_str = datetime.now().strftime("%Y-%m-%d")
        filename = f"{today_str}.md"
        file_path = self.daily_folder / filename
        
        if not file_path.exists():
            return

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            new_lines = []
            modified = False
            target_found = False
            
            for line in lines:
                # 尋找目標論文的那一行 (Matching Title)
                # "- [ ] **Title**"
                if f"**{paper_title}**" in line:
                    target_found = True
                    new_lines.append(line)
                    continue
                
                # 如果找到了標題，接下來找 "Open on Semantic Scholar" 那一行加連結
                if target_found and "Open on Semantic Scholar" in line:
                    if "Read Paper (Local)" not in line:
                        link_text = f" | [[{file_stem}.zh.md|Read Paper (Local)]]"
                        line = line.rstrip() + link_text + "\n"
                        modified = True
                        self.logger.info(f"Added link for {paper_title}")
                    target_found = False # Reset
                
                new_lines.append(line)
                
            if modified:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.writelines(new_lines)
                    
        except Exception as e:
            self.logger.error(f"更新每日筆記連結失敗: {e}")