import requests
from datetime import date, timedelta
from enum import IntEnum
from pathlib import Path
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
    "170": "訂正半期報告書",
}

API_KEY = ""
BASE_URL = "https://api.edinet-fsa.go.jp/api/v2/documents.json"
DEFAULT_START_DATE = date(2015, 10, 19)
LISTUP_COLUMNS = [
    "date",
    "docID",
    "edinetCode",
    "secCode",
    "filerName",
    "docTypeCode",
    "submitDateTime",
]
LISTUP_PATH = Path("listup.csv")
UPDATED_LISTUP_PATH = Path("updated_listup.csv")


def get_existing_listup() -> pd.DataFrame:
    if LISTUP_PATH.exists():
        return pd.read_csv(LISTUP_PATH, low_memory=False)
    return pd.DataFrame(columns=LISTUP_COLUMNS)


def determine_start_date(existing_df: pd.DataFrame) -> date:
    if "date" in existing_df.columns:
        dt = pd.to_datetime(existing_df["date"], errors="coerce")
        if dt.notna().any():
            return dt.max().date()
    return DEFAULT_START_DATE


def fetch_listup(start_date: date, end_date: date, existing_doc_ids: set[str]) -> pd.DataFrame:
    if start_date > end_date:
        return pd.DataFrame(columns=LISTUP_COLUMNS)

    total_days = (end_date - start_date).days + 1
    listup: list[dict] = []

    current = start_date
    with tqdm(total=total_days, desc="EDINET fetching", unit="day") as pbar:
        while current <= end_date:
            params = {
                "date": current.isoformat(),
                "type": EdinetType.MetaDataAndDocumentList,
                "Subscription-Key": API_KEY,
            }

            try:
                res = requests.get(BASE_URL, params=params, timeout=30)
                res.raise_for_status()
                data = res.json()

                for doc in data.get("results", []):
                    doc_id = doc.get("docID")
                    if (
                        doc.get("docTypeCode") in docTypeCode
                        and doc.get("secCode") is not None
                        and doc_id is not None
                        and doc_id not in existing_doc_ids
                    ):
                        row = {
                            "date": current.isoformat(),
                            "docID": doc_id,
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

    return pd.DataFrame(listup, columns=LISTUP_COLUMNS)


def main() -> bool:
    existing_df = get_existing_listup()
    start_date = determine_start_date(existing_df)
    end_date = date.today()

    existing_doc_ids = set(existing_df.get("docID", pd.Series(dtype=str)).astype(str).dropna())

    new_df = fetch_listup(start_date, end_date, existing_doc_ids)

    if new_df.empty:
        print("No new records found.")
        if UPDATED_LISTUP_PATH.exists():
            UPDATED_LISTUP_PATH.unlink()
        return False

    new_df.to_csv(UPDATED_LISTUP_PATH, index=False)
    print(f"Saved {len(new_df)} new records to {UPDATED_LISTUP_PATH}.")
    return True


if __name__ == "__main__":
    is_updated = main()
    print(f"updated={is_updated}")
