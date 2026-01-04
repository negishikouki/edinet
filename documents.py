import requests
from doc_type_code import doc_name_to_code
from enum import IntEnum

class EdinetType(IntEnum):
    MetaDataOnly = 1
    MetaDataAndDocumentList = 2


docTypeCode = {
    "120": "有価証券報告書",
    "140": "四半期報告書",
    "160": "半期報告書"
}

API_KEY = "b053b9de6579485d94409e03743f16c4"

url = "https://api.edinet-fsa.go.jp/api/v2/documents.json"
params = {
    "date": "2025-11-14",
    "type": EdinetType.MetaDataAndDocumentList,
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