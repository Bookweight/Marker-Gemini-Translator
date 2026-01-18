import logging
import numpy as np
from typing import List, Dict, Any, Optional
from sklearn.cluster import KMeans

class PaperRanker:
    """
    論文排序與過濾引擎 (Engineering Grade)
    
    功能:
    1. 計算論文權重分數 (考慮引用數、年份、跨領域懲罰、個人化向量)。
    2. 執行多樣性過濾 (K-Means Clustering)。
    """

    def __init__(self, config: Dict[str, Any]):
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        # 讀取設定參數
        filter_conf = self.config.get('filters', {})
        self.cross_domain_penalty = filter_conf.get('cross_domain_penalty', 0.6)
        self.whitelist = set(filter_conf.get('whitelist_fields', ["Computer Science", "Mathematics"]))
        self.cross_domain_tags = set(filter_conf.get('blacklist_tags', ["Biology", "Medicine", "Geology"]))
        
    def rank_candidates(self, papers: List[Dict[str, Any]], top_k: int = 5, user_vector: Optional[np.ndarray] = None) -> List[Dict[str, Any]]:
        """
        核心排序流程
        :param papers: 從 API 抓回來的論文列表
        :param top_k: 最終要推薦幾篇
        :param user_vector: 使用者偏好向量 (用於個性化加分)
        """
        if not papers:
            self.logger.warning("📭 候選列表為空，無法進行排序")
            return []

        self.logger.info(f"⚖️ 開始排序 {len(papers)} 篇候選論文...")

        # 1. 計算每篇論文的基礎分數
        scored_papers = []
        for paper in papers:
            if not paper.get('fieldsOfStudy'):
                paper['fieldsOfStudy'] = []
                
            score, is_cross = self._calculate_score(paper)
            
            # --- 個性化加分邏輯 (修復 None Type 錯誤) ---
            similarity_bonus = 0.0
            # 只有當分數大於 0 (未被剔除) 且有使用者向量時才計算
            if user_vector is not None and score > 0:
                # FIX: 使用 (get() or {}) 處理 embedding 為 None 的情況
                embedding_data = paper.get('embedding') or {}
                paper_vec_data = embedding_data.get('specter_v2')
                
                if paper_vec_data:
                    try:
                        p_v = np.array(paper_vec_data)
                        # 計算 Cosine Similarity
                        norm_u = np.linalg.norm(user_vector)
                        norm_p = np.linalg.norm(p_v)
                        
                        if norm_u > 0 and norm_p > 0:
                            cos_sim = np.dot(user_vector, p_v) / (norm_u * norm_p)
                            similarity_bonus = max(0, cos_sim)
                            paper['sim_score'] = similarity_bonus
                    except Exception as e:
                        self.logger.debug(f"向量計算錯誤 (PaperID: {paper.get('paperId')}): {e}")

            # 若分數為 0 代表被白名單過濾掉了
            if score > 0:
                # 最終分數 = 基礎分 * (1 + 相似度權重)
                paper['final_score'] = score * (1 + 1.0 * similarity_bonus)
                paper['is_cross_domain'] = is_cross
                scored_papers.append(paper)

        # 根據分數由高到低排序
        scored_papers.sort(key=lambda x: x['final_score'], reverse=True)
        self.logger.info(f"✅ 初步篩選後剩餘 {len(scored_papers)} 篇有效論文")

        # 2. 多樣性過濾 (Diversity Enforcement)
        if len(scored_papers) > top_k * 2:
            return self._apply_diversity_filter(scored_papers, top_k)
        else:
            self.logger.info("⚠️ 候選數量不足以進行多樣性聚類，直接回傳 Top K")
            return scored_papers[:top_k]

    def _calculate_score(self, paper: Dict[str, Any]) -> tuple[float, bool]:
        """計算單篇論文分數"""
        fields = set(paper.get('fieldsOfStudy', []))
        
        # A. 嚴格白名單檢查
        if not fields.intersection(self.whitelist):
            return 0.0, False

        # B. 判斷是否為跨領域
        is_cross_domain = not fields.isdisjoint(self.cross_domain_tags)
        
        # C. 取得影響力引用數
        citations = paper.get('influentialCitationCount') or paper.get('citationCount', 0)
        
        # D. 應用權重
        multiplier = self.cross_domain_penalty if is_cross_domain else 1.0
        final_score = citations * multiplier
        
        return final_score, is_cross_domain

    def _apply_diversity_filter(self, ranked_papers: List[Dict[str, Any]], target_k: int) -> List[Dict[str, Any]]:
        """使用 K-Means 確保推薦多樣性"""
        # 取出前 N 篇候選
        pool_size = min(len(ranked_papers), 20)
        candidate_pool = ranked_papers[:pool_size]
        
        embeddings = []
        valid_indices = []
        
        for idx, p in enumerate(candidate_pool):
            # FIX: 這裡同樣需要處理 embedding 為 None 的情況
            embedding_data = p.get('embedding') or {}
            emb = embedding_data.get('specter_v2')
            
            if emb:
                embeddings.append(emb)
                valid_indices.append(idx)
        
        if len(embeddings) < target_k:
            self.logger.warning("⚠️ 具有向量的論文不足，跳過多樣性過濾")
            return ranked_papers[:target_k]

        self.logger.info(f"🎨 執行多樣性聚類: 從 {len(embeddings)} 篇中選出 {target_k} 類代表作")
        
        # 執行 K-Means
        kmeans = KMeans(n_clusters=target_k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        
        final_selection = []
        cluster_found = set()
        
        # 依照分數高低選取每個 Cluster 的代表作
        for i in range(len(valid_indices)):
            original_idx = valid_indices[i]
            cluster_id = labels[i]
            
            if cluster_id not in cluster_found:
                final_selection.append(candidate_pool[original_idx])
                cluster_found.add(cluster_id)
                
            if len(final_selection) >= target_k:
                break
        
        # 補齊不足的數量
        if len(final_selection) < target_k:
            for p in candidate_pool:
                if p not in final_selection:
                    final_selection.append(p)
                    if len(final_selection) >= target_k:
                        break
        
        final_selection.sort(key=lambda x: x['final_score'], reverse=True)
        return final_selection