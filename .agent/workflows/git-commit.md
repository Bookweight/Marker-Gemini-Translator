---
description: 完整 git commit 流程，包含 CI 驗證與 Conventional Commits
---

# Git Commit Workflow

此 workflow 確保每次 commit 前通過所有 CI 檢查。

// turbo-all

1. 執行 pre-commit 全量檢查（自動修復 lint/format 問題）
```
uv run pre-commit run --all-files
```

2. 執行 pytest 確認測試全部通過
```
uv run pytest
```

3. 暫存所有變更
```
git add -A
```

4. 使用 Conventional Commits 格式提交
```
git commit -m "feat: <描述本次變更的功能>"
```

## Commit 類型參考

| 類型 | 用途 |
|------|------|
| `feat` | 新功能 |
| `fix` | 修復 bug |
| `refactor` | 重構（非功能、非修復） |
| `test` | 測試相關 |
| `docs` | 文件更新 |
| `chore` | 依賴更新、CI 配置等 |
