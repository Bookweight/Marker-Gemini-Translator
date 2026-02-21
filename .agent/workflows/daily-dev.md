---
description: 每日論文推薦開發流程，含功能開發、CI 驗證到 commit
---

# Daily Development Workflow

此 workflow 適用於每日開發迭代，確保長時間開發的品質控管。

1. 同步最新依賴
```
uv sync
```

// turbo
2. 執行 pre-commit 安裝（首次或更新後執行）
```
uv run pre-commit install
```

3. 開發、修改程式碼（agent 依任務執行）

// turbo
4. 執行 lint/format 自動修復
```
uv run pre-commit run --all-files
```

// turbo
5. 執行所有測試
```
uv run pytest -v
```

6. 若測試通過，執行 git commit workflow
```
/git-commit
```
