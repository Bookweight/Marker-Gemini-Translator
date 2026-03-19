
## 2026-03-19
- Arxiv API query script issues: Used -string quotes incorrectly inside python -c script, causing syntax errors. Also hit regex missing space issues (	itle = re.sub(...)). Fixed by ensuring correct imports (import re, import json) and quote escaping.
