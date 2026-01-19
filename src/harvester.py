import re
import json
import logging
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

# 引入之前的 Client
from src.client import S2Client

class ProfileManager:
    """
    使用者畫像管理器 (升級版)
    功能: 
    1. 管理 user_vector (興趣向量)
    2. 管理 history_ids (已推薦過的論文，避免重複)
    """
    def __init__(self, data_dir: str = "data"):
        self.logger = logging.getLogger(__name__)
        self.data_path = Path(data_dir)
        self.profile_file = self.data_path / "user_profile.json"
        self.data_path.mkdir(exist_ok=True)
        
        # 載入或初始化 Profile
        self.profile = self._load_profile()

    def _load_profile(self) -> Dict[str, Any]:
        default_profile = {
            "user_vector": None,
            "rated_paper_ids": [],    # 已評分的 (用於計算向量)
            "history_ids": [],        # [新增] 已推薦過的 (用於去重)
            "total_ratings": 0
        }

        if self.profile_file.exists():
            try:
                with open(self.profile_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    # 合併預設值 (避免舊版 json 缺少欄位報錯)
                    for key, val in default_profile.items():
                        if key not in data:
                            data[key] = val

                    # JSON存的是 list，轉回 numpy array
                    if data.get('user_vector'):
                        data['user_vector'] = np.array(data['user_vector'])
                    return data
            except Exception as e:
                self.logger.error(f"讀取 Profile 失敗: {e}，將使用預設值。")
        
        return default_profile

    def save_profile(self):
        """將 Profile 寫回 JSON"""
        data_to_save = self.profile.copy()
        # Numpy array 轉 list
        if data_to_save['user_vector'] is not None:
            data_to_save['user_vector'] = data_to_save['user_vector'].tolist()
        
        # 確保去重 (雖然 logic 會擋，但存檔前再保險一次)
        data_to_save['history_ids'] = list(set(data_to_save['history_ids']))
        
        with open(self.profile_file, 'w', encoding='utf-8') as f:
            json.dump(data_to_save, f, indent=2, ensure_ascii=False)

    def add_recommendations(self, paper_ids: List[str]):
        """[新增] 記錄今日推薦的論文 ID"""
        current_history = set(self.profile['history_ids'])
        new_ids = set(paper_ids)
        
        # 更新歷史紀錄
        updated_history = current_history.union(new_ids)
        self.profile['history_ids'] = list(updated_history)
        
        self.save_profile()
        self.logger.info(f"已將 {len(new_ids)} 篇新論文加入歷史紀錄 (總計: {len(updated_history)})")

    def update_vector(self, paper_vector: List[float], rating: int):
        """Rocchio 演算法更新向量"""
        if rating <= 3:
            weight = -0.5
        elif 4 <= rating <= 6:
            weight = 0.0
            return
        else:
            weight = 1.0

        paper_vec = np.array(paper_vector)
        user_vec = self.profile['user_vector']

        if user_vec is None:
            self.logger.info("冷啟動: 初始化使用者向量")
            self.profile['user_vector'] = paper_vec
            return

        n = self.profile['total_ratings']
        learning_rate = max(0.01, 0.1 * (0.95 ** n)) 
        
        new_vec = user_vec + learning_rate * weight * (paper_vec - user_vec)
        
        norm = np.linalg.norm(new_vec)
        if norm > 0:
            new_vec = new_vec / norm
            
        self.profile['user_vector'] = new_vec
        self.profile['total_ratings'] += 1
        self.logger.info(f"向量已更新 (Rating: {rating})")


class NoteHarvester:
    """筆記收割者 (維持不變)"""
    def __init__(self, config: Dict[str, Any], client: S2Client, profile_manager: ProfileManager):
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.client = client
        self.pm = profile_manager
        
        vault_path = Path(self.config['obsidian']['vault_path'])
        daily_folder = self.config['obsidian'].get('daily_folder', '')
        self.search_path = vault_path / daily_folder

    def harvest(self, lookback_days: int = 7):
        self.logger.info(f"開始收割過去 {lookback_days} 天的評分...")
        today = datetime.now()
        found_ratings = []

        for i in range(lookback_days):
            date_str = (today - timedelta(days=i)).strftime("%Y-%m-%d")
            note_path = self.search_path / f"{date_str}.md"
            if note_path.exists():
                ratings = self._parse_note(note_path)
                found_ratings.extend(ratings)
        
        self.logger.info(f"共發現 {len(found_ratings)} 個評分標記")

        processed_ids = set(self.pm.profile['rated_paper_ids'])
        new_ratings = [r for r in found_ratings if r['paper_id'] not in processed_ids]
        
        if not new_ratings:
            self.logger.info("沒有新的評分需要處理。")
            return

        self.logger.info(f"準備處理 {len(new_ratings)} 個新評分...")
        
        ids_to_fetch = [r['paper_id'] for r in new_ratings]
        paper_details = self.client.get_batch_details(ids_to_fetch)
        embedding_map = {
            p['paperId']: (p.get('embedding') or {}).get('specter_v2') 
            for p in paper_details
        }
        
        for item in new_ratings:
            p_id = item['paper_id']
            score = item['score']
            vec = embedding_map.get(p_id)
            if vec:
                self.pm.update_vector(vec, score)
                self.pm.profile['rated_paper_ids'].append(p_id)
                
                # [關鍵] 評過分的論文也自動加入歷史清單 (如果尚未加入)
                if p_id not in self.pm.profile['history_ids']:
                    self.pm.profile['history_ids'].append(p_id)

        self.pm.save_profile()
        self.logger.info("使用者畫像更新完成！")

    def _parse_note(self, file_path: Path) -> List[Dict]:
        content = file_path.read_text(encoding='utf-8')
        results = []
        cards = re.split(r'- \[.\] \*\*', content)
        for card in cards:
            id_match = re.search(r'semanticscholar\.org/paper/([a-f0-9]+)', card)
            score_match = re.search(r"Rating\**:\s*\(\s*(\d+)\s*\)", card)
            if id_match and score_match:
                score = int(score_match.group(1))
                if 1 <= score <= 10:
                    results.append({'paper_id': id_match.group(1), 'score': score})
        return results