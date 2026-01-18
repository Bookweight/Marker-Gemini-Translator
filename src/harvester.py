import re
import json
import logging
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

# 引入之前的 Client，因為我們需要去抓取那些被評分論文的 Vector
from src.client import S2Client

class ProfileManager:
    """
    使用者畫像管理器
    功能: 管理 user_profile.json，執行 Rocchio 向量更新
    """
    def __init__(self, data_dir: str = "data"):
        self.logger = logging.getLogger(__name__)
        self.data_path = Path(data_dir)
        self.profile_file = self.data_path / "user_profile.json"
        self.data_path.mkdir(exist_ok=True)
        
        # 載入或初始化 Profile
        self.profile = self._load_profile()

    def _load_profile(self) -> Dict[str, Any]:
        if self.profile_file.exists():
            try:
                with open(self.profile_file, 'r') as f:
                    data = json.load(f)
                    # JSON存的是 list，轉回 numpy array
                    if data.get('user_vector'):
                        data['user_vector'] = np.array(data['user_vector'])
                    return data
            except Exception as e:
                self.logger.error(f"讀取 Profile 失敗: {e}，將重置。")
        
        # 預設 Profile
        return {
            "user_vector": None,       # 初始為 None (冷啟動)
            "rated_paper_ids": [],     # 紀錄已評分的 ID，避免重複訓練
            "total_ratings": 0
        }

    def save_profile(self):
        """將 Profile 寫回 JSON (Numpy array 需轉 list)"""
        data_to_save = self.profile.copy()
        if data_to_save['user_vector'] is not None:
            data_to_save['user_vector'] = data_to_save['user_vector'].tolist()
        
        with open(self.profile_file, 'w') as f:
            json.dump(data_to_save, f, indent=2)

    def update_vector(self, paper_vector: List[float], rating: int):
        """
        Rocchio 演算法核心 [cite: 76-78]
        Rating 映射邏輯:
          1-3 分 (負面): 推離 (-0.5 ~ -1.0)
          4-6 分 (忽略): 權重 0 (視為雜訊)
          7-10 分 (正面): 拉近 (+0.5 ~ +1.0)
        """
        # 1. 定義權重 w
        if rating <= 3:
            weight = -0.5  # 負面
        elif 4 <= rating <= 6:
            weight = 0.0   # 忽略
            self.logger.info(f"Rating {rating} 視為中立，跳過更新。")
            return
        else:
            weight = 1.0   # 正面 (7-10)

        paper_vec = np.array(paper_vector)
        user_vec = self.profile['user_vector']

        # 2. 冷啟動處理：如果是第一個評分，直接把論文向量當成使用者向量
        if user_vec is None:
            self.logger.info("冷啟動: 初始化使用者向量")
            self.profile['user_vector'] = paper_vec
            return

        # 3. Rocchio 更新公式: u_new = u_old + learning_rate * weight * (d - u_old)
        # 隨著評分次數增加，Learning Rate 逐漸降低 (0.1 -> 0.01) 以保持穩定
        n = self.profile['total_ratings']
        learning_rate = max(0.01, 0.1 * (0.95 ** n)) 
        
        # 向量加權移動
        new_vec = user_vec + learning_rate * weight * (paper_vec - user_vec)
        
        # 正規化 (Optional but recommended for Cosine Sim)
        norm = np.linalg.norm(new_vec)
        if norm > 0:
            new_vec = new_vec / norm
            
        self.profile['user_vector'] = new_vec
        self.profile['total_ratings'] += 1
        self.logger.info(f"向量已更新 (Rating: {rating}, Weight: {weight}, LR: {learning_rate:.4f})")


class NoteHarvester:
    """
    筆記收割者
    功能: 掃描 Obsidian 筆記，提取評分
    """
    def __init__(self, config: Dict[str, Any], client: S2Client, profile_manager: ProfileManager):
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.client = client
        self.pm = profile_manager
        
        vault_path = Path(self.config['obsidian']['vault_path'])
        daily_folder = self.config['obsidian'].get('daily_folder', '')
        self.search_path = vault_path / daily_folder

    def harvest(self, lookback_days: int = 7):
        """掃描過去 N 天的筆記"""
        self.logger.info(f"🌾 開始收割過去 {lookback_days} 天的評分...")
        
        today = datetime.now()
        found_ratings = []

        # 1. 遍歷日期檔案
        for i in range(lookback_days):
            date_str = (today - timedelta(days=i)).strftime("%Y-%m-%d")
            note_path = self.search_path / f"{date_str}.md"
            
            if note_path.exists():
                ratings = self._parse_note(note_path)
                found_ratings.extend(ratings)
        
        self.logger.info(f"共發現 {len(found_ratings)} 個評分標記")

        # 2. 過濾已處理過的評分
        processed_ids = set(self.pm.profile['rated_paper_ids'])
        new_ratings = [r for r in found_ratings if r['paper_id'] not in processed_ids]
        
        if not new_ratings:
            self.logger.info("沒有新的評分需要處理。")
            return

        self.logger.info(f"準備處理 {len(new_ratings)} 個新評分...")
        
        # 3. 獲取向量並更新
        # 為了節省 API，我們將 ID 收集起來一次抓取 (Batch)
        ids_to_fetch = [r['paper_id'] for r in new_ratings]
        paper_details = self.client.get_batch_details(ids_to_fetch)
        
        # 建立 ID -> Embedding 的查表
        embedding_map = {p['paperId']: p.get('embedding', {}).get('specter_v2') for p in paper_details}
        
        # 4. 執行更新
        for item in new_ratings:
            p_id = item['paper_id']
            score = item['score']
            vec = embedding_map.get(p_id)
            
            if vec:
                self.pm.update_vector(vec, score)
                self.pm.profile['rated_paper_ids'].append(p_id)
            else:
                self.logger.warning(f"無法獲取論文 {p_id} 的向量，跳過更新")

        # 5. 存檔
        self.pm.save_profile()
        self.logger.info("✅ 使用者畫像更新完成！")

    def _parse_note(self, file_path: Path) -> List[Dict]:
        """
        解析單一 Markdown 檔案
        尋找結構:
           - [Open on Semantic Scholar](.../paper/{paperId})
           - **Rating**: (9)
        """
        content = file_path.read_text(encoding='utf-8')
        results = []
        
        # 使用 Regex 捕捉：先抓 URL 裡的 ID，再往下找最近的 Rating
        # 注意：這個 Regex 假設 Link 和 Rating 在同一個區塊 (我們的 Writer 是這樣寫的)
        
        # 步驟 A: 將內容依據 "- [ ]" 分割成卡片區塊，避免跨論文誤判
        cards = re.split(r'- \[.\] \*\*', content)
        
        for card in cards:
            # 1. 提取 ID
            id_match = re.search(r'semanticscholar\.org/paper/([a-f0-9]+)', card)
            # 2. 提取分數 (支援 (9), ( 9 ), (10))
            score_match = re.search(r"Rating\**:\s*\(\s*(\d+)\s*\)", card)
            
            if id_match and score_match:
                paper_id = id_match.group(1)
                score = int(score_match.group(1))
                
                # 合理性檢查 (1-10分)
                if 1 <= score <= 10:
                    results.append({'paper_id': paper_id, 'score': score})
        
        return results