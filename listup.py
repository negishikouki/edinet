import json
from datetime import date, datetime, timedelta, timezone
from enum import IntEnum
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

from doc_type_code import TARGET_DOC_TYPE_NAMES
from edinet_constants import LISTUP_COLUMNS, get_api_key
from normalization import is_missing_scalar, normalize_sec_code_text
from provenance import file_identity, sha256_bytes, write_bytes_atomic, write_text_atomic


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
LIST_RESPONSE_ROOT = STATE_ROOT / "list_responses"
LAST_SUCCESSFUL_LISTUP_DATE_PATH = STATE_ROOT / "last_successful_listup_date.txt"
LIST_EVENT_COLUMNS = [
    "queryDate",
    "responseSha256",
    "docID",
    "edinetCode",
    "secCode",
    "filerName",
    "docTypeCode",
    "submitDateTime",
    "opeDateTime",
    "docInfoEditStatus",
    "withdrawalStatus",
    "disclosureStatus",
    "csvFlag",
    "xbrlFlag",
    "pdfFlag",
    "englishDocFlag",
]
JST = timezone(timedelta(hours=9), name="JST")


def ensure_listup_columns(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    for column in LISTUP_COLUMNS:
        if column not in result.columns:
            result[column] = pd.NA
    return pd.DataFrame(result.loc[:, list(LISTUP_COLUMNS)])


def _json_response_bytes(
    response: requests.Response,
    payload: dict[str, object],
) -> bytes:
    content = getattr(response, "content", b"")
    if isinstance(content, bytes) and content:
        return content
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def persist_list_response(
    query_date: date,
    response: requests.Response,
    payload: dict[str, object],
) -> Path:
    """Persist the exact response, its event fields, and SHA-256 identities."""

    content = _json_response_bytes(response, payload)
    response_sha256 = sha256_bytes(content)
    response_dir = (
        LIST_RESPONSE_ROOT
        / f"{query_date.year:04d}"
        / query_date.isoformat()
    )
    raw_path = response_dir / f"{response_sha256}.json"
    if not raw_path.exists():
        write_bytes_atomic(raw_path, content)
    write_text_atomic(
        raw_path.with_suffix(f"{raw_path.suffix}.sha256"),
        f"{response_sha256}  {raw_path.name}\n",
    )

    results = payload.get("results", [])
    documents = results if isinstance(results, list) else []
    event_rows: list[dict[str, object | None]] = []
    for value in documents:
        if not isinstance(value, dict):
            continue
        doc = {str(key): item for key, item in value.items()}
        event_rows.append(
            {
                "queryDate": query_date.isoformat(),
                "responseSha256": response_sha256,
                "docID": doc.get("docID"),
                "edinetCode": doc.get("edinetCode"),
                "secCode": normalize_sec_code_text(doc.get("secCode")),
                "filerName": doc.get("filerName"),
                "docTypeCode": doc.get("docTypeCode"),
                "submitDateTime": doc.get("submitDateTime"),
                "opeDateTime": doc.get("opeDateTime"),
                "docInfoEditStatus": doc.get("docInfoEditStatus"),
                "withdrawalStatus": doc.get("withdrawalStatus"),
                "disclosureStatus": doc.get("disclosureStatus"),
                "csvFlag": doc.get("csvFlag"),
                "xbrlFlag": doc.get("xbrlFlag"),
                "pdfFlag": doc.get("pdfFlag"),
                "englishDocFlag": doc.get("englishDocFlag"),
            }
        )
    events_path = response_dir / f"{response_sha256}.events.csv"
    events_content = pd.DataFrame(
        event_rows,
        columns=LIST_EVENT_COLUMNS,
    ).to_csv(index=False).encode("utf-8")
    write_bytes_atomic(events_path, events_content)
    manifest_path = response_dir / f"{response_sha256}.manifest.json"
    manifest = {
        "manifestSchemaVersion": 1,
        "queryDate": query_date.isoformat(),
        "responseSha256": response_sha256,
        "rawJson": file_identity(raw_path, root=response_dir),
        "eventsCsv": file_identity(events_path, root=response_dir),
    }
    write_text_atomic(
        manifest_path,
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
    )
    return raw_path


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
        if is_missing_scalar(value):
            continue
        try:
            failed_date = date.fromisoformat(str(value))
        except ValueError:
            continue
        error = row.get("error")
        failures[failed_date] = "" if is_missing_scalar(error) else str(error)
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
            "updatedAt": pd.Timestamp.now(tz="UTC").isoformat(),
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
    frames: list[pd.DataFrame] = []
    for path in (LISTUP_PATH, UPDATED_LISTUP_PATH):
        if path.exists():
            frames.append(
                ensure_listup_columns(pd.read_csv(path, low_memory=False))
            )
    if not frames:
        return pd.DataFrame(columns=LISTUP_COLUMNS)
    merged = pd.concat(frames, ignore_index=True)
    if "docID" in merged.columns:
        merged = merged.drop_duplicates("docID", keep="last")
    return normalize_sec_code_column(ensure_listup_columns(merged))


def load_last_successful_listup_date() -> date | None:
    if not LAST_SUCCESSFUL_LISTUP_DATE_PATH.is_file():
        return None
    try:
        return date.fromisoformat(
            LAST_SUCCESSFUL_LISTUP_DATE_PATH.read_text(encoding="utf-8").strip()
        )
    except ValueError:
        return None


def record_last_successful_listup_date(successful_date: date) -> None:
    write_text_atomic(
        LAST_SUCCESSFUL_LISTUP_DATE_PATH,
        f"{successful_date.isoformat()}\n",
    )


def determine_start_date(existing_df: pd.DataFrame) -> date:
    candidates: list[date] = []
    if "date" in existing_df.columns:
        dt = pd.to_datetime(existing_df["date"], errors="coerce")
        if dt.notna().any():
            candidates.append(dt.max().date())
    last_successful_date = load_last_successful_listup_date()
    if last_successful_date is not None:
        candidates.append(last_successful_date)
    return max(candidates) if candidates else DEFAULT_START_DATE


def build_listup_row(
    query_date: date,
    doc: dict[str, object],
) -> dict[str, object | None]:
    return {
        "date": query_date.isoformat(),
        "docID": doc.get("docID"),
        "edinetCode": doc.get("edinetCode"),
        "secCode": normalize_sec_code_text(doc.get("secCode")),
        "filerName": doc.get("filerName"),
        "docTypeCode": doc.get("docTypeCode"),
        "submitDateTime": doc.get("submitDateTime"),
        "listQueryDate": query_date.isoformat(),
        "opeDateTime": doc.get("opeDateTime"),
        "docInfoEditStatus": doc.get("docInfoEditStatus"),
        "withdrawalStatus": doc.get("withdrawalStatus"),
        "disclosureStatus": doc.get("disclosureStatus"),
        "csvFlag": doc.get("csvFlag"),
        "xbrlFlag": doc.get("xbrlFlag"),
        "pdfFlag": doc.get("pdfFlag"),
        "englishDocFlag": doc.get("englishDocFlag"),
    }


def fetch_listup(start_date: date, end_date: date, existing_doc_ids: set[str]) -> pd.DataFrame:
    del existing_doc_ids
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
                response_payload: object = res.json()
                if not isinstance(response_payload, dict):
                    raise TypeError("EDINET list response must be a JSON object")
                data = {
                    str(key): value for key, value in response_payload.items()
                }
                persist_list_response(current, res, data)

                results = data.get("results", [])
                documents = results if isinstance(results, list) else []
                for value in documents:
                    if not isinstance(value, dict):
                        continue
                    doc = {str(key): item for key, item in value.items()}
                    doc_id = doc.get("docID")
                    if (
                        doc.get("docTypeCode") in TARGET_DOC_TYPE_NAMES
                        and doc_id is not None
                    ):
                        listup.append(build_listup_row(current, doc))
                clear_failed_listup_date(current)
                record_last_successful_listup_date(current)

            except requests.exceptions.HTTPError as e:
                if e.response is not None and e.response.status_code == 403:
                    raise
                print(f"[ERROR] {current}: {e}")
                record_failed_listup_date(current, str(e))
            except (
                OSError,
                TypeError,
                ValueError,
                requests.exceptions.RequestException,
            ) as e:
                print(f"[ERROR] {current}: {e}")
                record_failed_listup_date(current, str(e))

            pbar.update(1)

    return normalize_sec_code_column(
        ensure_listup_columns(pd.DataFrame(listup, columns=LISTUP_COLUMNS))
    )


def _comparable_value(value: object) -> str:
    return "" if is_missing_scalar(value) else str(value)


def select_changed_listup_rows(
    fetched_df: pd.DataFrame,
    existing_df: pd.DataFrame,
) -> pd.DataFrame:
    if fetched_df.empty:
        return pd.DataFrame(columns=LISTUP_COLUMNS)
    existing_by_id = {
        str(row["docID"]): row
        for row in ensure_listup_columns(existing_df).to_dict("records")
        if not is_missing_scalar(row.get("docID"))
    }
    compared_columns = [
        column
        for column in LISTUP_COLUMNS
        if column not in {"date", "listQueryDate"}
    ]
    changed_rows: list[dict[str, object]] = []
    for row in ensure_listup_columns(fetched_df).to_dict("records"):
        doc_id_value = row.get("docID")
        if is_missing_scalar(doc_id_value):
            continue
        doc_id = str(doc_id_value)
        existing = existing_by_id.get(doc_id)
        if existing is None:
            changed_rows.append({str(key): value for key, value in row.items()})
            continue
        if any(
            _comparable_value(row.get(column))
            != _comparable_value(existing.get(column))
            for column in compared_columns
        ):
            updated = {str(key): value for key, value in row.items()}
            updated["date"] = existing.get("date")
            changed_rows.append(updated)
    return normalize_sec_code_column(
        ensure_listup_columns(pd.DataFrame(changed_rows))
    )


def main() -> bool:
    existing_df = get_existing_listup()
    start_date = determine_start_date(existing_df)
    end_date = datetime.now(JST).date()

    existing_doc_ids = (
        set(existing_df["docID"].astype(str).dropna())
        if "docID" in existing_df.columns
        else set()
    )

    fetched_df = normalize_sec_code_column(
        fetch_listup(start_date, end_date, existing_doc_ids)
    )
    new_df = select_changed_listup_rows(fetched_df, existing_df)

    if new_df.empty:
        print("No new records found.")
        return False

    pending_frames = [new_df]
    if UPDATED_LISTUP_PATH.exists():
        pending_frames.insert(
            0,
            ensure_listup_columns(
                pd.read_csv(UPDATED_LISTUP_PATH, low_memory=False)
            ),
        )
    pending = (
        pd.concat(pending_frames, ignore_index=True)
        .drop_duplicates("docID", keep="last")
        .reset_index(drop=True)
    )
    write_bytes_atomic(
        UPDATED_LISTUP_PATH,
        ensure_listup_columns(pending).to_csv(index=False).encode("utf-8"),
    )
    print(f"Saved {len(new_df)} changed records to {UPDATED_LISTUP_PATH}.")
    return True


if __name__ == "__main__":
    is_updated = main()
    print(f"updated={is_updated}")
