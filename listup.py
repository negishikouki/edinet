import requests
from datetime import date, timedelta
from enum import IntEnum
import pandas as pd
from tqdm import tqdm

class EdinetType(IntEnum):
    MetaDataOnly = 1
    MetaDataAndDocumentList = 2

docTypeCode = {
    "120": "有価証券報告書",
    "130": "訂正有価証券報告書",
    "140": "四半期報告書",
    "150": "訂正四半期報告書",
    "160": "半期報告書",
    "170": "訂正半期報告書"
}

API_KEY = ""
BASE_URL = "https://api.edinet-fsa.go.jp/api/v2/documents.json"

start_date = date(2015, 10, 19)
end_date = date.today()

total_days = (end_date - start_date).days + 1

listup = []

current = start_date
with tqdm(total=total_days, desc="EDINET fetching", unit="day") as pbar:
    while current <= end_date:
        params = {
            "date": current.isoformat(),
            "type": EdinetType.MetaDataAndDocumentList,
            "Subscription-Key": API_KEY
        }

        try:
            res = requests.get(BASE_URL, params=params, timeout=30)
            res.raise_for_status()
            data = res.json()

            for doc in data.get("results", []):
                if (
                    doc.get("docTypeCode") in docTypeCode
                    and doc.get("secCode") is not None
                ):
                    row = {
                        "date": current.isoformat(),
                        "docID": doc.get("docID"),
                        "edinetCode": doc.get("edinetCode"),
                        "secCode": doc.get("secCode"),
                        "filerName": doc.get("filerName"),
                        "docTypeCode": doc.get("docTypeCode"),
                        "submitDateTime": doc.get("submitDateTime"),
                    }
                    listup.append(row)

        except Exception as e:
            print(f"[ERROR] {current}: {e}")

        current += timedelta(days=1)
        pbar.update(1)

df = pd.DataFrame(listup)
df.to_csv("listup.csv")