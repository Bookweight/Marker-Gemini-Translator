# 論文筆記知識庫 (Research Vault)

> [!INFO] 提示
> 此頁面透過 **Dataview** 自動彙整 `Papers` 資料夾下的所有翻譯內容。

---

## 領域分類彙整 (Fields Index)
```dataview
TABLE 
    rows.file.link as "論文名稱", 
    rows.status as "狀態"
FROM "Papers"
WHERE field != null
GROUP BY field
```


##  每日論文推薦

```dataview
TABLE date, paper_count, status 
FROM #daily_rec  
SORT date DESC
```
