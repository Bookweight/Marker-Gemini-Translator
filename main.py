import os
import yaml
import logging
from dotenv import load_dotenv
from pathlib import Path

from src.client import S2Client
from src.ranker import PaperRanker
from src.writer import ObsidianWriter
from src.harvester import NoteHarvester, ProfileManager
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
        logger.info("🚀 系統啟動...")
        load_dotenv()
        config = load_config()
        api_key = os.getenv("S2_API_KEY")
        
        client = S2Client(api_key, config)
        profile_manager = ProfileManager()
        harvester = NoteHarvester(config, client, profile_manager)
        ranker = PaperRanker(config)
        writer = ObsidianWriter(config)
        downloader = PaperDownloader(config)

        # 1.5 執行收割 (維持不變)
        try:
            harvester.harvest(lookback_days=7)
        except Exception as e:
            logger.error(f"收割評分失敗，將使用舊有 Profile 繼續: {e}")
        
        # 2. 獲取候選 (維持不變)
        query = config['search']['keywords']
        years = config['search']['year_range']
        candidates = client.search_papers(query, years)
        
        # 3. 過濾與排序 (維持不變)
        whitelist = set(config['filters']['whitelist_fields'])
        valid_ids = []
        for p in candidates:
            fields = set(p.get('fieldsOfStudy') or [])
            if not fields.isdisjoint(whitelist):
                valid_ids.append(p['paperId'])
                
        logger.info(f"經過白名單過濾，準備抓取 {len(valid_ids)} 篇論文詳情")
        
        user_vec = profile_manager.profile['user_vector']
        detailed_papers = client.get_batch_details(valid_ids)
        
        top_papers = ranker.rank_candidates(detailed_papers, top_k=5, user_vector=user_vec)
        
        # 4. 寫入介面 (修改核心邏輯)
        if top_papers:
            # 嘗試寫入筆記 (如果已存在，writer 會自動跳過並回傳 False，但這不重要)
            writer.write_recommendations(top_papers)
            
            # [修改點] 不論筆記是否是新建立的，都強制執行下載檢查
            # Downloader 內部本身就有檢查 "檔案是否存在" 的邏輯，所以這裡直接呼叫是安全的
            logger.info("🚀 進入檔案檢查流程：確認 PDF 與翻譯是否齊全...")
            downloader.process_papers(top_papers)
            
            logger.info("🎉 本次執行結束。")
        else:
            logger.warning("⚠️ 今日未能選出任何論文。")
            
    except Exception as e:
        logger.error(f"💥 系統崩潰: {e}", exc_info=True)

if __name__ == "__main__":
    main()