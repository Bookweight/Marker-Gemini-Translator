import os
import yaml
import logging
from dotenv import load_dotenv
from pathlib import Path

from src.client import S2Client
from src.ranker import PaperRanker
from src.writer import ObsidianWriter
from src.downloader import PaperDownloader

# 設定 Logging (同時輸出到 Console 和 File)
def setup_logging():
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / "app.log", encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

def load_config(path="config.yaml"):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def main():
    setup_logging()
    logger = logging.getLogger("Main")
    
    try:
        # 1. 初始化
        logger.info("系統啟動...")
        load_dotenv()
        config = load_config()
        api_key = os.getenv("S2_API_KEY")
        
        client = S2Client(api_key, config)
        # Refactored: Ranker now handles Profile and Harvesting
        ranker = PaperRanker(config, client)
        writer = ObsidianWriter(config)
        downloader = PaperDownloader(config, writer=writer)

        # 1.5 執行收割
        try:
            ranker.harvest_feedback(lookback_days=7)
        except Exception as e:
            logger.error(f"收割評分失敗，將使用舊有 Profile 繼續: {e}")
        
        # 2. 獲取候選
        keywords = config['search']['keywords']
        if isinstance(keywords, str):
            keywords = [keywords]
            
        years = config['search']['year_range']
        
        all_candidates = {} # 使用字典依 paperId 去重
        
        logger.info(f"🔍 啟動多領域搜尋: 包含 {len(keywords)} 個主題")
        
        for topic in keywords:
            logger.info(f"  - 正在搜尋領域: {topic}...")
            # Note: client usage remains same
            papers = client.search_papers(topic, years, limit=15)
            
            for p in papers:
                all_candidates[p['paperId']] = p
                
        # 轉回列表
        candidates = list(all_candidates.values())
        logger.info(f"✅ 多領域搜尋完成，合併後共 {len(candidates)} 篇候選論文")
        
        # 3. 過濾與排序
        whitelist = set(config['filters']['whitelist_fields'])
        # Access history from ranker's profile manager
        history_set = set(ranker.profile_manager.profile.get('history_ids', []))
        
        logger.info(f"目前歷史資料庫已有 {len(history_set)} 篇論文 (將被排除)")
        valid_ids = []
        for p in candidates:
            p_id = p['paperId']
            if p_id in history_set:
                continue
            fields = set(p.get('fieldsOfStudy') or [])
            if not fields.isdisjoint(whitelist):
                valid_ids.append(p['paperId'])
                
        logger.info(f"經過白名單過濾，準備抓取 {len(valid_ids)} 篇論文詳情")
        
        detailed_papers = client.get_batch_details(valid_ids)
        
        # Refactored: rank_candidates uses internal profile, no need to pass user_vector
        top_papers = ranker.rank_candidates(detailed_papers, top_k=5)
        
        # 4. 寫入介面
        if top_papers:
            if writer.write_recommendations(top_papers):
                # 成功寫入後，更新今日歷程 (避免重複推薦)
                recommended_ids = [p['paperId'] for p in top_papers]
                ranker.profile_manager.add_recommendations(recommended_ids)


            logger.info("進入檔案檢查流程：確認 PDF 與翻譯是否齊全...")
            downloader.process_papers(top_papers)
            
            logger.info("本次執行結束。")
        else:
            logger.warning("今日未能選出任何論文。")
            
    except Exception as e:
        logger.error(f"系統崩潰: {e}", exc_info=True)

if __name__ == "__main__":
    main()