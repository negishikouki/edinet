from enum import IntEnum

import requests


class EdinetType(IntEnum):
    MetaDataOnly = 1
    MetaDataAndDocumentList = 2


docTypeCode = {
    "120": "有価証券報告書",
    "140": "四半期報告書",
    "160": "半期報告書"
}

API_KEY = ""

url = "https://api.edinet-fsa.go.jp/api/v2/documents.json"
params: dict[str, str | int] = {
    "date": "2025-11-14",
    "type": int(EdinetType.MetaDataAndDocumentList),
    "Subscription-Key": API_KEY
}

res = requests.get(url, params=params)
res.raise_for_status()

data = res.json()

for doc in data.get("results", []):
    if doc["docTypeCode"] in docTypeCode.keys() and \
        doc["secCode"] is not None:
        print(
            doc["docID"],
            doc["edinetCode"],
            doc["filerName"],
            doc["docTypeCode"],
            doc["submitDateTime"]
        )
