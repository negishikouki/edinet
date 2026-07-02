from enum import IntEnum

import requests

from doc_type_code import TARGET_DOC_TYPE_NAMES
from edinet_constants import get_api_key


class EdinetType(IntEnum):
    MetaDataOnly = 1
    MetaDataAndDocumentList = 2


API_KEY = get_api_key()

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
    if doc["docTypeCode"] in TARGET_DOC_TYPE_NAMES and \
        doc["secCode"] is not None:
        print(
            doc["docID"],
            doc["edinetCode"],
            doc["filerName"],
            doc["docTypeCode"],
            doc["submitDateTime"]
        )
