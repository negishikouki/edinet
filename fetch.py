import io
import os
import shutil
import time
import zipfile
from enum import IntEnum
from pathlib import Path
from typing import Mapping

import pandas as pd
import requests
from tqdm import tqdm

from doc_type_code import DOC_TYPE_PERIOD
from edinet_constants import get_api_key


# -----------------------------
# EDINET 定義
# -----------------------------
class EdinetDocType(IntEnum):
    CSV = 5


API_KEY = get_api_key()
BASE_URL = "https://api.edinet-fsa.go.jp/api/v2/documents"
LISTUP_PATH = Path("listup.csv")
UPDATED_LISTUP_PATH = Path("updated_listup.csv")
REQUEST_TIMEOUT_SECONDS = 30
COMMUNICATION_RETRY_COUNT = 3
RETRY_BACKOFF_SECONDS = 1
RETRYABLE_HTTP_STATUS_CODES = {429, 500, 502, 503, 504}
STATE_ROOT = Path("state")
FAILED_FETCH_DOCS_PATH = STATE_ROOT / "failed_fetch_docs.csv"
FAILED_FETCH_DOCS_COLUMNS = ["docID", "error", "updatedAt"]

doc_prefix_by_frequency = {
    "annual": {"prefix": ["jpcrp030000"], "role": "通期（12か月）"},
    "quarterly": {"prefix": ["jpcrp040300"], "role": "3か月（または累計）"},
    "semiannual": {"prefix": ["jpcrp050000", "jpcrp050200"], "role": "6か月（累計）"},
    "common": {"prefix": ["jpdei"], "role": "期間・連結区分などのキー"},
}

all_prefixes = sorted({p for v in doc_prefix_by_frequency.values() for p in v["prefix"]})
RequestParams = dict[str, str | int]


# -----------------------------
# CSV を取得する関数（prefix可変）
# -----------------------------
def get_document_response(docID: str, url: str, params: RequestParams) -> requests.Response:
    for retry_count in range(COMMUNICATION_RETRY_COUNT + 1):
        try:
            response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            if retry_count >= COMMUNICATION_RETRY_COUNT:
                raise

            wait_seconds = RETRY_BACKOFF_SECONDS * (2 ** retry_count)
            tqdm.write(
                f"[RETRY] {docID}: communication error: {e} "
                f"(retry {retry_count + 1}/{COMMUNICATION_RETRY_COUNT}, "
                f"wait {wait_seconds}s)"
            )
            time.sleep(wait_seconds)
            continue

        status_code = getattr(response, "status_code", None)
        if (
            isinstance(status_code, int)
            and status_code in RETRYABLE_HTTP_STATUS_CODES
            and retry_count < COMMUNICATION_RETRY_COUNT
        ):
            wait_seconds = RETRY_BACKOFF_SECONDS * (2 ** retry_count)
            tqdm.write(
                f"[RETRY] {docID}: HTTP {status_code} "
                f"(retry {retry_count + 1}/{COMMUNICATION_RETRY_COUNT}, "
                f"wait {wait_seconds}s)"
            )
            time.sleep(wait_seconds)
            continue

        return response

    raise RuntimeError("unreachable")


def fetch_all_prefix_dfs(docID: str, prefixes: list[str]) -> dict[str, pd.DataFrame]:
    url = f"{BASE_URL}/{docID}"
    params: RequestParams = {"type": int(EdinetDocType.CSV), "Subscription-Key": API_KEY}

    res = get_document_response(docID, url, params)
    res.raise_for_status()

    content_type = (res.headers.get("Content-Type") or "").lower()

    # --- JSONが返ってきたら、エラー理由を取り出してスキップ（or 例外）
    if "application/json" in content_type:
        try:
            j = res.json()
        except Exception:
            raise ValueError(f"{docID}: JSON returned but failed to parse. body={res.text[:500]}")

        # 代表的なキーを拾って分かりやすくする
        msg = j.get("message") or j.get("Message") or j.get("error") or j.get("Error") or str(j)
        raise ValueError(f"{docID}: EDINET returned JSON (not ZIP). message={msg}")

        # ※「止めずに続行」したいなら raise の代わりに:
        # return {}

    # --- ZIP（octet-stream）を期待
    if "octet-stream" not in content_type and "zip" not in content_type:
        raise ValueError(f"{docID}: unexpected Content-Type {res.headers.get('Content-Type')}")

    dfs: dict[str, pd.DataFrame] = {}

    with zipfile.ZipFile(io.BytesIO(res.content)) as zf:
        names = zf.namelist()

        for prefix in prefixes:
            prefix_path = f"XBRL_TO_CSV/{prefix}"
            target_files = [
                name for name in names
                if name.startswith(prefix_path) and name.endswith(".csv")
            ]
            if not target_files:
                continue

            with zf.open(target_files[0]) as f:
                dfs[prefix] = pd.read_csv(f, encoding="utf-16", sep="\t")

    return dfs


def build_fetched_docid_set(
    out_root: Path,
    doc_type_by_id: Mapping[str, str] | None = None,
) -> set[str]:
    """
    doc/配下の完了済み docID をまとめて取得する。
    """
    if not out_root.exists():
        return set()

    if doc_type_by_id is None:
        return {
            marker.stem
            for marker in (out_root / ".complete").glob("*.complete")
        }
    return build_completed_docid_set(out_root, doc_type_by_id)


def load_failed_fetch_doc_ids(path: Path = FAILED_FETCH_DOCS_PATH) -> set[str]:
    if not path.exists():
        return set()

    df = pd.read_csv(path, dtype=str)
    if "docID" not in df.columns:
        return set()
    return set(df["docID"].dropna().astype(str))


def write_failed_fetch_docs(
    failures: dict[str, str],
    path: Path = FAILED_FETCH_DOCS_PATH,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "docID": doc_id,
            "error": error,
            "updatedAt": pd.Timestamp.utcnow().isoformat(),
        }
        for doc_id, error in sorted(failures.items())
    ]
    pd.DataFrame(rows, columns=FAILED_FETCH_DOCS_COLUMNS).to_csv(path, index=False)


def load_failed_fetch_docs(path: Path = FAILED_FETCH_DOCS_PATH) -> dict[str, str]:
    if not path.exists():
        return {}

    df = pd.read_csv(path, dtype=str)
    failures: dict[str, str] = {}
    if "docID" not in df.columns:
        return failures
    for row in df.to_dict("records"):
        doc_id = row.get("docID")
        if doc_id is None or pd.isna(doc_id):
            continue
        error = row.get("error")
        failures[str(doc_id)] = "" if error is None or pd.isna(error) else str(error)
    return failures


def record_failed_fetch_doc(
    docID: str,
    error: str,
    path: Path = FAILED_FETCH_DOCS_PATH,
) -> None:
    failures = load_failed_fetch_docs(path)
    failures[docID] = error
    write_failed_fetch_docs(failures, path)


def clear_failed_fetch_doc(
    docID: str,
    path: Path = FAILED_FETCH_DOCS_PATH,
) -> None:
    failures = load_failed_fetch_docs(path)
    if docID not in failures:
        return
    del failures[docID]
    write_failed_fetch_docs(failures, path)


def complete_marker_path(out_root: Path, docID: str) -> Path:
    return out_root / ".complete" / f"{docID}.complete"


def required_prefix_groups(doc_type_code: str | int | None) -> list[tuple[str, ...]]:
    if doc_type_code is None or pd.isna(doc_type_code):
        return []
    period_type = DOC_TYPE_PERIOD.get(str(doc_type_code))
    if period_type is None:
        return []
    return [
        tuple(doc_prefix_by_frequency[period_type]["prefix"]),
        tuple(doc_prefix_by_frequency["common"]["prefix"]),
    ]


def doc_has_required_csvs(
    docID: str,
    doc_type_code: str | int | None,
    out_root: Path,
) -> bool:
    groups = required_prefix_groups(doc_type_code)
    if not groups:
        return False
    return all(
        any((out_root / prefix / f"{docID}.csv").exists() for prefix in group)
        for group in groups
    )


def build_completed_docid_set(
    out_root: Path,
    doc_type_by_id: Mapping[str, str],
) -> set[str]:
    if not out_root.exists():
        return set()

    completed = {
        marker.stem
        for marker in (out_root / ".complete").glob("*.complete")
    }
    for doc_id, doc_type_code in doc_type_by_id.items():
        if doc_has_required_csvs(doc_id, doc_type_code, out_root):
            completed.add(doc_id)
    return completed


def docid_already_fetched(docID: str, fetched_docids: set[str]) -> bool:
    """
    事前に収集した CSV ファイル一覧から docID の有無を判定する
    """
    return docID in fetched_docids


def persist_doc_csvs(
    docID: str,
    doc_type_code: str | int | None,
    dfs_by_prefix: dict[str, pd.DataFrame],
    out_root: Path,
) -> None:
    groups = required_prefix_groups(doc_type_code)
    missing_groups = [
        "/".join(group)
        for group in groups
        if not any(prefix in dfs_by_prefix for prefix in group)
    ]
    if missing_groups:
        raise ValueError(
            f"{docID}: required CSV prefixes are missing: {', '.join(missing_groups)}"
        )

    temp_root = out_root / ".tmp" / f"{docID}.{os.getpid()}.{time.time_ns()}"
    try:
        for prefix, df in dfs_by_prefix.items():
            temp_dir = temp_root / prefix
            temp_dir.mkdir(parents=True, exist_ok=True)
            df.to_csv(temp_dir / f"{docID}.csv", index=False)

        for prefix in dfs_by_prefix:
            final_dir = out_root / prefix
            final_dir.mkdir(parents=True, exist_ok=True)
            os.replace(
                temp_root / prefix / f"{docID}.csv",
                final_dir / f"{docID}.csv",
            )

        marker_path = complete_marker_path(out_root, docID)
        marker_path.parent.mkdir(parents=True, exist_ok=True)
        marker_path.write_text(pd.Timestamp.utcnow().isoformat(), encoding="utf-8")
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


# -----------------------------
# main
# -----------------------------
def main():
    listup_path = UPDATED_LISTUP_PATH if UPDATED_LISTUP_PATH.exists() else LISTUP_PATH
    out_root = Path("./doc")
    out_root.mkdir(parents=True, exist_ok=True)
    listup_df = pd.read_csv(listup_path)
    listup_df["docID"] = listup_df["docID"].astype(str)
    doc_type_by_id = {
        str(row["docID"]): str(row["docTypeCode"])
        for row in listup_df[["docID", "docTypeCode"]].dropna().to_dict("records")
    }
    fetched_docids = build_completed_docid_set(out_root, doc_type_by_id)
    docIDs = listup_df["docID"].dropna().astype(str).tolist()

    for docID in tqdm(docIDs, desc="Fetching per docID"):

        # 既存チェック
        if docid_already_fetched(docID, fetched_docids):
            clear_failed_fetch_doc(docID)
            tqdm.write(f"[SKIP] {docID}: already exists")
            continue

        try:
            dfs_by_prefix = fetch_all_prefix_dfs(docID, all_prefixes)
            persist_doc_csvs(
                docID,
                doc_type_by_id.get(docID),
                dfs_by_prefix,
                out_root,
            )
            fetched_docids.add(docID)
            clear_failed_fetch_doc(docID)

        except Exception as e:
            record_failed_fetch_doc(docID, str(e))
            tqdm.write(f"[WARN] {docID}: {e}")


if __name__ == "__main__":
    main()
