import json
import logging
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

import numpy as np
import yaml
from sklearn.cluster import KMeans

# 引入 Client Type Hint
# 由於循環引用風險 (client -> ranker -> client)，這裡使用 TYPE_CHECKING 或直接用 Any，
# 但 client.py 並沒有 import ranker，所以直接 import 是安全的。
# 不過為了保持 ranker 獨立性，也可以只用 Any。
from src.client import S2Client


class ProfileManager:
    """
    使用者畫像管理器
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
        default_profile: Dict[str, Any] = {
            "user_vector": None,
            "rated_paper_ids": [],  # 已評分的 (用於計算向量)
            "history_ids": [],  # 已推薦過的 (用於去重)
            "total_ratings": 0,
        }

        if self.profile_file.exists():
            try:
                with open(self.profile_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                    # 合併預設值 (避免舊版 json 缺少欄位報錯)
                    for key, val in default_profile.items():
                        if key not in data:
                            data[key] = val

                    # JSON存的是 list，轉回 numpy array
                    if data.get("user_vector"):
                        data["user_vector"] = np.array(data["user_vector"])
                    return cast(Dict[str, Any], data)
            except Exception as e:
                self.logger.error(f"讀取 Profile 失敗: {e}，將使用預設值。")

        return default_profile

    def save_profile(self):
        """將 Profile 寫回 JSON"""
        data_to_save = self.profile.copy()
        # Numpy array 轉 list
        if data_to_save["user_vector"] is not None:
            data_to_save["user_vector"] = data_to_save["user_vector"].tolist()

        # 確保去重
        data_to_save["history_ids"] = list(set(data_to_save["history_ids"]))

        with open(self.profile_file, "w", encoding="utf-8") as f:
            json.dump(data_to_save, f, indent=2, ensure_ascii=False)

    def add_recommendations(self, paper_ids: List[str]):
        """記錄今日推薦的論文 ID"""
        current_history = set(self.profile["history_ids"])
        new_ids = set(paper_ids)

        # 更新歷史紀錄
        updated_history = current_history.union(new_ids)
        self.profile["history_ids"] = list(updated_history)

        self.save_profile()
        self.logger.info(
            f"已將 {len(new_ids)} 篇新論文加入歷史紀錄 (總計: {len(updated_history)})"
        )

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
        user_vec = self.profile["user_vector"]

        if user_vec is None:
            self.logger.info("冷啟動: 初始化使用者向量")
            self.profile["user_vector"] = paper_vec
            return

        n = self.profile["total_ratings"]
        learning_rate = max(0.01, 0.1 * (0.95**n))

        new_vec = user_vec + learning_rate * weight * (paper_vec - user_vec)

        norm = np.linalg.norm(new_vec)
        if norm > 0:
            new_vec = new_vec / norm

        self.profile["user_vector"] = new_vec
        self.profile["total_ratings"] += 1
        self.logger.info(f"向量已更新 (Rating: {rating})")


class PaperRanker:
    """
    推薦與排序引擎 (Recommendation Engine)

    整合了:
    1. User Profiling (ProfileManager)
    2. Feedback Harvesting (NoteHarvester logic)
    3. Content-based Ranking (Vector Similarity + Weights)
    """

    def __init__(self, config: Dict[str, Any], client: Optional[S2Client] = None):
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.client = client

        # 初始化 User Profile Manager
        self.profile_manager = ProfileManager()

        # 讀取過濾參數
        filter_conf = self.config.get("filters", {})
        self.cross_domain_penalty = filter_conf.get("cross_domain_penalty", 0.6)
        self.whitelist = set(
            filter_conf.get("whitelist_fields", ["Computer Science", "Mathematics"])
        )
        self.cross_domain_tags = set(
            filter_conf.get("blacklist_tags", ["Biology", "Medicine", "Geology"])
        )

        # Load Venue Sources
        self.sources_config = self._load_sources_config()

    def _load_sources_config(self) -> Dict[str, Any]:
        """讀取 sources.yaml 設定"""
        try:
            with open("sources.yaml", "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            self.logger.warning(f"無法讀取 sources.yaml，將不啟用來源加權: {e}")
            return {}

    # --- Harvesting Logic ---
    def harvest_feedback(self, lookback_days: int = 7):
        """從 Obsidian 每日筆記中收割評分並更新 Profile"""
        if not self.client:
            self.logger.warning("未設定 Client，無法執行收割 (Harvesting skipped)")
            return

        self.logger.info(f"開始收割過去 {lookback_days} 天的評分...")

        vault_path = Path(self.config["obsidian"]["vault_path"])
        daily_folder = self.config["obsidian"].get("daily_folder", "")
        search_path = vault_path / daily_folder

        today = datetime.now()
        found_ratings = []

        for i in range(lookback_days):
            date_str = (today - timedelta(days=i)).strftime("%Y-%m-%d")
            note_path = search_path / f"{date_str}.md"
            if note_path.exists():
                ratings = self._parse_note(note_path)
                found_ratings.extend(ratings)

        self.logger.info(f"共發現 {len(found_ratings)} 個評分標記")

        processed_ids = set(self.profile_manager.profile["rated_paper_ids"])
        new_ratings = [r for r in found_ratings if r["paper_id"] not in processed_ids]

        if not new_ratings:
            self.logger.info("沒有新的評分需要處理。")
            return

        self.logger.info(f"準備處理 {len(new_ratings)} 個新評分...")

        ids_to_fetch = [r["paper_id"] for r in new_ratings]
        paper_details = self.client.get_batch_details(ids_to_fetch)
        # 建立 embedding 查找表
        embedding_map = {
            p["paperId"]: (p.get("embedding") or {}).get("specter_v2")
            for p in paper_details
        }

        for item in new_ratings:
            p_id = item["paper_id"]
            score = item["score"]
            vec = embedding_map.get(p_id)
            if vec:
                self.profile_manager.update_vector(vec, score)
                self.profile_manager.profile["rated_paper_ids"].append(p_id)

                # 評過分的論文也自動加入歷史清單
                if p_id not in self.profile_manager.profile["history_ids"]:
                    self.profile_manager.profile["history_ids"].append(p_id)

        self.profile_manager.save_profile()
        self.logger.info("使用者畫像更新完成！")

    def _parse_note(self, file_path: Path) -> List[Dict]:
        content = file_path.read_text(encoding="utf-8")
        results = []
        # 與 harvester.py 邏輯一致，解析 regex
        cards = re.split(r"- \[.\] \*\*", content)
        for card in cards:
            id_match = re.search(r"semanticscholar\.org/paper/([a-f0-9]+)", card)
            score_match = re.search(r"Rating\**:\s*\(\s*(\d+)\s*\)", card)
            if id_match and score_match:
                score = int(score_match.group(1))
                if 1 <= score <= 10:
                    results.append({"paper_id": id_match.group(1), "score": score})
        return results

    # --- Ranking Logic ---
    def rank_candidates(
        self, papers: List[Dict[str, Any]], top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        核心排序流程 (自動使用內部 Profile)
        """
        if not papers:
            self.logger.warning("候選列表為空，無法進行排序")
            return []

        self.logger.info(f"開始排序 {len(papers)} 篇候選論文...")

        user_vector = self.profile_manager.profile["user_vector"]

        # 1. 計算每篇論文的基礎分數
        scored_papers = []
        for paper in papers:
            if not paper.get("fieldsOfStudy"):
                paper["fieldsOfStudy"] = []

            score, is_cross = self._calculate_score(paper)

            # --- 個性化加分邏輯 ---
            similarity_bonus = 0.0
            if user_vector is not None and score > 0:
                embedding_data = paper.get("embedding") or {}
                paper_vec_data = embedding_data.get("specter_v2")

                if paper_vec_data:
                    try:
                        p_v = np.array(paper_vec_data)
                        norm_u = np.linalg.norm(user_vector)
                        norm_p = np.linalg.norm(p_v)

                        if norm_u > 0 and norm_p > 0:
                            cos_sim = np.dot(user_vector, p_v) / (norm_u * norm_p)
                            similarity_bonus = max(0, cos_sim)
                            paper["sim_score"] = similarity_bonus
                    except Exception as e:
                        self.logger.debug(
                            f"向量計算錯誤 (PaperID: {paper.get('paperId')}): {e}"
                        )

            if score > 0:
                paper["final_score"] = score * (1 + 1.0 * similarity_bonus)
                paper["is_cross_domain"] = is_cross
                scored_papers.append(paper)

        # 根據分數由高到低排序
        scored_papers.sort(key=lambda x: x["final_score"], reverse=True)
        self.logger.info(f"初步篩選後剩餘 {len(scored_papers)} 篇有效論文")

        # 2. 多樣性過濾
        if len(scored_papers) > top_k * 2:
            return self._apply_diversity_filter(scored_papers, top_k)
        else:
            self.logger.info("候選數量不足以進行多樣性聚類，直接回傳 Top K")
            return scored_papers[:top_k]

    def _calculate_score(self, paper: Dict[str, Any]) -> tuple[float, bool]:
        """計算單篇論文分數"""
        fields = set(paper.get("fieldsOfStudy", []))

        # A. 嚴格白名單檢查
        if not fields.intersection(self.whitelist):
            return 0.0, False

        # B. 判斷是否為跨領域
        is_cross_domain = not fields.isdisjoint(self.cross_domain_tags)

        # C. 取得影響力引用數 (加上時效性紅利)
        raw_citations = paper.get("influentialCitationCount") or paper.get(
            "citationCount", 0
        )

        # 時效性紅利 (Recency Bonus)
        year = paper.get("year")
        recency_bonus = 0
        if year:
            try:
                current_year = datetime.now().year
                age = current_year - int(year)
                # 從 config 讀取紅利基數，預設 50
                bonus_base = self.config.get("ranking", {}).get(
                    "recency_bonus_baseline", 50
                )

                if age <= 0:  # 當年或未來 (0-1年)
                    recency_bonus = bonus_base
                elif age == 1:  # 去年 (1-2年)
                    recency_bonus = bonus_base / 2
            except Exception:
                pass

        # 基礎分 = 實際引用 + 紅利
        citations = raw_citations + recency_bonus

        # D. 應用權重
        multiplier = self.cross_domain_penalty if is_cross_domain else 1.0

        # E. 來源加權
        venue_weight = self._get_venue_weight(paper)
        if venue_weight < 0:
            return -1.0, is_cross_domain

        final_score = citations * multiplier * venue_weight

        return final_score, is_cross_domain

    def _get_venue_weight(self, paper: Dict[str, Any]) -> float:
        """根據期刊/會議 Tier List 計算權重"""
        venue = paper.get("venue")
        pub_venue = paper.get("publicationVenue")
        if pub_venue:
            if isinstance(pub_venue, dict):
                venue = venue or pub_venue.get("name") or pub_venue.get("text")
            elif isinstance(pub_venue, str):
                venue = venue or pub_venue

        journal = paper.get("journal")
        if not venue and journal and isinstance(journal, dict):
            venue = journal.get("name")

        if not venue or not isinstance(venue, str):
            return 1.0

        venue_lower = venue.lower().strip()

        # 1. Check Blacklist
        blacklist = self.sources_config.get("blacklist", [])
        for bad in blacklist:
            if bad in venue_lower:
                return -10.0

        # 2. Check Tier 1
        tier1 = self.sources_config.get("tier_1", [])
        if any(t in venue_lower for t in tier1):
            return 1.5

        # 3. Check Tier 2
        tier2 = self.sources_config.get("tier_2", [])
        if any(t in venue_lower for t in tier2):
            return 1.2

        return 1.0

    def _apply_diversity_filter(
        self, ranked_papers: List[Dict[str, Any]], target_k: int
    ) -> List[Dict[str, Any]]:
        """使用 K-Means 確保推薦多樣性"""
        pool_size = min(len(ranked_papers), 20)
        candidate_pool = ranked_papers[:pool_size]

        embeddings = []
        valid_indices = []

        for idx, p in enumerate(candidate_pool):
            embedding_data = p.get("embedding") or {}
            emb = embedding_data.get("specter_v2")
            if emb:
                embeddings.append(emb)
                valid_indices.append(idx)

        if len(embeddings) < target_k:
            self.logger.warning("具有向量的論文不足，跳過多樣性過濾")
            return ranked_papers[:target_k]

        self.logger.info(
            f"執行多樣性聚類: 從 {len(embeddings)} 篇中選出 {target_k} 類代表作"
        )

        kmeans = KMeans(n_clusters=target_k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)

        final_selection = []
        cluster_found = set()

        # 優先挑選每個 Cluster 分數最高的 (這邊簡單遍歷，確保每個 label 至少一篇)
        # 因為 candidate_pool 已經是照分數排序的，所以第一次遇到某 label 就是該 label 最高分的
        for i in valid_indices:
            label = labels[valid_indices.index(i)]
            if label not in cluster_found:
                final_selection.append(candidate_pool[i])
                cluster_found.add(label)

        # 如果還不夠 (有些 cluster 可能沒被選到?), 補滿
        if len(final_selection) < target_k:
            remaining = [p for p in candidate_pool if p not in final_selection]
            final_selection.extend(remaining[: target_k - len(final_selection)])

        return final_selection
