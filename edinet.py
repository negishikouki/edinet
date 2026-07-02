import json

import requests

from edinet_constants import get_api_key

API_KEY = get_api_key()

url = "https://disclosure.edinet-fsa.go.jp/api/v2/documents.json"
params: dict[str, str | int] = {
    "date": "2025-06-16",
    "type": 2,
    "Subscription-Key": API_KEY
}

res = requests.get(url, params=params)

data = res.json()
print(data)

# JSONファイルとして保存
with open("edinet_documents_20250616.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
