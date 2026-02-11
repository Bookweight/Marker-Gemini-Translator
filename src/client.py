import logging
import time
from typing import Any, Dict, List, Optional, cast

import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)


class S2Client:
    """
    Semantic Scholar API 客戶端 (Engineering Grade)

    特點:
    1. Type Hinting: 全面型別提示，提升開發體驗與除錯效率。
    2. Config Driven: 參數不寫死，依賴傳入的 config 字典。
    3. Logging: 使用 logger 取代 print，支援檔案紀錄。
    4. Resilience: 內建 Rate Limit 保護與指數退避重試機制。
    """

    BASE_URL = "https://api.semanticscholar.org/graph/v1"

    def __init__(self, api_key: Optional[str], config: Dict[str, Any]):
        """
        初始化客戶端

        :param api_key: 從環境變數讀取的 API Key
        :param config: 從 config.yaml 讀取的設定字典
        """
        self.logger = logging.getLogger(__name__)  # 取得當前模組的 logger
        self.api_key = api_key
        self.config = config

        # 設定 Headers
        self.headers = {}
        if self.api_key:
            self.headers["x-api-key"] = self.api_key
        else:
            self.logger.warning(
                "未偵測到 API Key，將以未認證模式運作 (Rate Limit 極低)"
            )

        # 從設定檔讀取參數 (設有預設值以防 config 缺漏)
        self.batch_size = self.config.get("api", {}).get("batch_size", 50)
        self.rate_limit_sleep = self.config.get("api", {}).get("sleep_seconds", 1.1)

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(requests.exceptions.RequestException),
        reraise=True,
    )
    def _make_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict] = None,
        json_data: Optional[Dict] = None,
    ) -> Any:
        """
        發送 HTTP 請求的底層函式 (Private Method)
        """
        url = f"{self.BASE_URL}{endpoint}"

        # 強制冷卻：遵守 Rate Limit
        time.sleep(self.rate_limit_sleep)

        try:
            self.logger.debug(f"發送請求: {method} {url}")

            if method == "GET":
                response = requests.get(url, headers=self.headers, params=params)
            elif method == "POST":
                response = requests.post(
                    url, headers=self.headers, params=params, json=json_data
                )
            else:
                raise ValueError(f"不支援的 HTTP 方法: {method}")

            # 處理 429 Too Many Requests
            if response.status_code == 429:
                self.logger.warning("⚠️ 觸發 Rate Limit (429)，Tenacity 將接手重試...")
                raise requests.exceptions.RequestException("Rate Limit Hit")

            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            self.logger.error(f"API 請求失敗: {e} | URL: {url}")
            raise  # 拋出讓 Tenacity 捕獲

    def search_papers(
        self, query: str, year_range: str, limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        搜尋候選論文 (Discovery Layer)

        :param query: 搜尋關鍵字 (如 "Deep Learning")
        :param year_range: 年份範圍 (如 "2020-2025")
        :param limit: 搜尋數量 (若未指定則讀取 config)
        :return: 包含基礎資訊的論文列表
        """
        endpoint = "/paper/search"

        # 若未指定 limit，則從 config 讀取，預設 20
        search_limit = limit or self.config.get("search", {}).get("limit", 20)

        params = {
            "query": query,
            "year": year_range,
            "limit": search_limit,
            "fields": "paperId,title,fieldsOfStudy,year",  # 只抓過濾需要的欄位
        }

        self.logger.info(
            f"執行搜尋: Query='{query}', Year='{year_range}', Limit={search_limit}"
        )

        data = self._make_request("GET", endpoint, params=params)
        papers = cast(List[Dict[str, Any]], data.get("data", []))

        self.logger.info(f"搜尋完成，共找到 {len(papers)} 篇候選論文")
        return papers

    def get_batch_details(self, paper_ids: List[str]) -> List[Dict[str, Any]]:
        """
        批量獲取詳細資料 (Enrichment Layer)

        :param paper_ids: 論文 ID 列表
        :return: 包含 Embedding 與引用數的詳細資料列表
        """
        if not paper_ids:
            return []

        endpoint = "/paper/batch"

        # 指定需要的欄位
        fields = "paperId,title,year,influentialCitationCount,citationCount,fieldsOfStudy,abstract,embedding.specter_v2,openAccessPdf,externalIds,venue,publicationVenue,journal"

        params = {"fields": fields}

        all_details = []
        total_batches = (len(paper_ids) + self.batch_size - 1) // self.batch_size

        self.logger.info(
            f"📥 開始批量下載詳情: {len(paper_ids)} 篇論文，分 {total_batches} 批次處理"
        )

        for i in range(0, len(paper_ids), self.batch_size):
            chunk = paper_ids[i : i + self.batch_size]
            payload = {"ids": chunk}

            try:
                self.logger.debug(
                    f"處理批次 {i // self.batch_size + 1}/{total_batches} (Size: {len(chunk)})"
                )
                result = self._make_request(
                    "POST", endpoint, params=params, json_data=payload
                )

                if result:
                    # 過濾 None (S2 有時會回傳找不到的 ID 為 None)
                    valid_items = [p for p in result if p is not None]
                    all_details.extend(valid_items)
            except Exception as e:
                self.logger.error(f"批次處理失敗 (Index {i}): {e}")
                # 選擇：這裡可以決定要中斷還是繼續 (目前策略是紀錄錯誤並繼續)
                continue

        self.logger.info(f"批量下載完成，成功獲取 {len(all_details)} 篇論文詳情")
        return all_details
