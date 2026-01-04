import requests

API_KEY = "b053b9de6579485d94409e03743f16c4"

url = "https://disclosure.edinet-fsa.go.jp/api/v2/documents.json"
params = {
    "date": "2025-07-16",
    "type": 2, 
    "Subscription-Key": API_KEY
    }

res = requests.get(url, params=params)

print(res.json())
