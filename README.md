
# 論文筆記知識庫 (Research Vault)

> [!INFO] 提示
> 此頁面透過 **Dataview** 自動彙整 `Papers` 資料夾下的所有翻譯內容。

---

##  最近加入的論文 (最近 10 篇)

```dataview
TABLE 
    field as "領域", 
    status as "狀態", 
    created_date as "翻譯日期",
    pdf_link as "原始 PDF"
FROM "Papers"
SORT file.ctime DESC
LIMIT 10
```

## 領域分類彙整 (Fields Index)
```dataview
TABLE 
    rows.file.link as "論文名稱", 
    rows.status as "狀態"
FROM "Papers"
WHERE field != null
GROUP BY field
```
