@echo off
cd /d "C:\Users\User\Desktop\paper reading"
:: 使用 uv 執行，確保環境正確
uv run main.py >> logs/scheduler.log 2>&1