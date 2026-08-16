from datetime import date, datetime, timedelta, timezone
from enum import IntEnum
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

from doc_type_code import TARGET_DOC_TYPE_NAMES
from edinet_constants import LISTUP_COLUMNS, get_api_key
from normalization import normalize_sec_code_text


class EdinetType(IntEnum):
    MetaDataOnly = 1
    MetaDataAndDocumentList = 2


API_KEY = get_api_key()
BASE_URL = "https://api.edinet-fsa.go.jp/api/v2/documents.json"
DEFAULT_START_DATE = date(2015, 10, 19)
LISTUP_PATH = Path("listup.csv")
UPDATED_LISTUP_PATH = Path("updated_listup.csv")
STATE_ROOT = Path("state")
FAILED_LISTUP_DATES_PATH = STATE_ROOT / "failed_listup_dates.csv"
FAILED_LISTUP_COLUMNS = ["date", "error", "updatedAt"]
JST = timezone(timedelta(hours=9), name="JST")


def normalize_sec_code_column(df: pd.DataFrame) -> pd.DataFrame:
    if "secCode" not in df.columns:
        return df

    normalized = df.copy()
    normalized["secCode"] = normalized["secCode"].map(normalize_sec_code_text).astype("string")
    return normalized


def load_failed_listup_dates(path: Path = FAILED_LISTUP_DATES_PATH) -> dict[date, str]:
    if not path.exists():
        return {}

    df = pd.read_csv(path, dtype=str)
    failures: dict[date, str] = {}
    if "date" not in df.columns:
        return failures

    for row in df.to_dict("records"):
        value = row.get("date")
        if value is None or pd.isna(value):
            continue
        try:
            failed_date = date.fromisoformat(str(value))
        except ValueError:
            continue
        error = row.get("error")
        failures[failed_date] = "" if error is None or pd.isna(error) else str(error)
    return failures


def write_failed_listup_dates(
    failures: dict[date, str],
    path: Path = FAILED_LISTUP_DATES_PATH,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "date": failed_date.isoformat(),
            "error": error,
            "updatedAt": pd.Timestamp.utcnow().isoformat(),
        }
        for failed_date, error in sorted(failures.items())
    ]
    pd.DataFrame(rows, columns=FAILED_LISTUP_COLUMNS).to_csv(path, index=False)


def record_failed_listup_date(
    failed_date: date,
    error: str,
    path: Path = FAILED_LISTUP_DATES_PATH,
) -> None:
    failures = load_failed_listup_dates(path)
    failures[failed_date] = error
    write_failed_listup_dates(failures, path)


def clear_failed_listup_date(
    successful_date: date,
    path: Path = FAILED_LISTUP_DATES_PATH,
) -> None:
    failures = load_failed_listup_dates(path)
    if successful_date not in failures:
        return
    del failures[successful_date]
    write_failed_listup_dates(failures, path)


def build_listup_fetch_dates(
    start_date: date,
    end_date: date,
    failed_dates: set[date],
) -> list[date]:
    normal_dates: set[date] = set()
    if start_date <= end_date:
        total_days = (end_date - start_date).days + 1
        normal_dates = {
            start_date + timedelta(days=offset)
            for offset in range(total_days)
        }
    return sorted(normal_dates | failed_dates)


def get_existing_listup() -> pd.DataFrame:
    if LISTUP_PATH.exists():
        return normalize_sec_code_column(pd.read_csv(LISTUP_PATH, low_memory=False))
    return pd.DataFrame(columns=LISTUP_COLUMNS)


def determine_start_date(existing_df: pd.DataFrame) -> date:
    if "date" in existing_df.columns:
        dt = pd.to_datetime(existing_df["date"], errors="coerce")
        if dt.notna().any():
            return dt.max().date()
    return DEFAULT_START_DATE


def fetch_listup(start_date: date, end_date: date, existing_doc_ids: set[str]) -> pd.DataFrame:
    fetch_dates = build_listup_fetch_dates(
        start_date,
        end_date,
        set(load_failed_listup_dates()),
    )
    if not fetch_dates:
        return pd.DataFrame(columns=LISTUP_COLUMNS)

    listup: list[dict[str, object | None]] = []

    with tqdm(total=len(fetch_dates), desc="EDINET fetching", unit="day") as pbar:
        for current in fetch_dates:
            params: dict[str, str | int] = {
                "date": current.isoformat(),
                "type": int(EdinetType.MetaDataAndDocumentList),
                "Subscription-Key": API_KEY,
            }

            try:
                res = requests.get(BASE_URL, params=params, timeout=30)
                res.raise_for_status()
                data = res.json()

                for doc in data.get("results", []):
                    doc_id = doc.get("docID")
                    if (
                        doc.get("docTypeCode") in TARGET_DOC_TYPE_NAMES
                        and doc_id is not None
                        and doc_id not in existing_doc_ids
                    ):
                        row = {
                            "date": current.isoformat(),
                            "docID": doc_id,
                            "edinetCode": doc.get("edinetCode"),
                            "secCode": normalize_sec_code_text(doc.get("secCode")),
                            "filerName": doc.get("filerName"),
                            "docTypeCode": doc.get("docTypeCode"),
                            "submitDateTime": doc.get("submitDateTime"),
                        }
                        listup.append(row)
                clear_failed_listup_date(current)

            except Exception as e:
                print(f"[ERROR] {current}: {e}")
                record_failed_listup_date(current, str(e))

            pbar.update(1)

    return normalize_sec_code_column(pd.DataFrame(listup, columns=LISTUP_COLUMNS))


def main() -> bool:
    existing_df = get_existing_listup()
    start_date = determine_start_date(existing_df)
    end_date = datetime.now(JST).date()

    existing_doc_ids = (
        set(existing_df["docID"].astype(str).dropna())
        if "docID" in existing_df.columns
        else set()
    )

    new_df = normalize_sec_code_column(fetch_listup(start_date, end_date, existing_doc_ids))

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
