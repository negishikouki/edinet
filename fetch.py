import io
import json
import os
import shutil
import time
import zipfile
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Mapping

import pandas as pd
import requests
from tqdm import tqdm

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

RequestParams = dict[str, str | int]


@dataclass(frozen=True)
class FetchedCsv:
    source_name: str
    prefix: str
    is_body: bool
    dataframe: pd.DataFrame


@dataclass(frozen=True)
class FetchedDocument:
    csvs: tuple[FetchedCsv, ...]
    body_prefixes: tuple[str, ...]
    body_csv_files: tuple[str, ...]
    has_jpdei_cor: bool


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


def csv_prefix_from_source_name(source_name: str) -> str:
    return Path(source_name).stem.split("-", maxsplit=1)[0]


def _is_body_csv_prefix(prefix: str) -> bool:
    return not prefix.startswith("jpaud")


def _has_jpdei_cor(df: pd.DataFrame) -> bool:
    if "要素ID" not in df.columns:
        return False
    element_ids = df["要素ID"].astype("string")
    return bool(element_ids.str.startswith("jpdei_cor:", na=False).any())


def fetch_all_prefix_dfs(
    docID: str,
    prefixes: list[str] | None = None,
) -> FetchedDocument:
    """
    ZIP内のXBRL_TO_CSV直下にある全CSVを取得する。

    ``prefixes``は旧呼び出し側との互換性のために受け取るが、実際の
    CSV構成を失わないよう、docTypeCodeによる絞り込みには使用しない。
    """
    del prefixes
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

    fetched_csvs: list[FetchedCsv] = []

    with zipfile.ZipFile(io.BytesIO(res.content)) as zf:
        csv_names = [
            name
            for name in zf.namelist()
            if name.startswith("XBRL_TO_CSV/")
            and name.endswith(".csv")
            and "/" not in name.removeprefix("XBRL_TO_CSV/")
        ]

        for name in sorted(csv_names):
            source_name = Path(name).name
            prefix = csv_prefix_from_source_name(source_name)
            with zf.open(name) as f:
                dataframe = pd.read_csv(f, encoding="utf-16", sep="\t")
            fetched_csvs.append(
                FetchedCsv(
                    source_name=source_name,
                    prefix=prefix,
                    is_body=_is_body_csv_prefix(prefix),
                    dataframe=dataframe,
                )
            )

    body_csvs = tuple(csv for csv in fetched_csvs if csv.is_body)
    return FetchedDocument(
        csvs=tuple(fetched_csvs),
        body_prefixes=tuple(dict.fromkeys(csv.prefix for csv in body_csvs)),
        body_csv_files=tuple(csv.source_name for csv in body_csvs),
        has_jpdei_cor=any(_has_jpdei_cor(csv.dataframe) for csv in body_csvs),
    )


def build_fetched_docid_set(
    out_root: Path,
    doc_type_by_id: Mapping[str, str] | None = None,
) -> set[str]:
    """
    doc/配下の完了済み docID をまとめて取得する。
    """
    if not out_root.exists():
        return set()

    return build_completed_docid_set(out_root, doc_type_by_id or {})


def load_failed_fetch_doc_ids(path: Path = FAILED_FETCH_DOCS_PATH) -> set[str]:
    if not path.exists():
        return set()

    df = pd.read_csv(path, dtype=str)
    if "docID" not in df.columns:
        return set()
    return set(df["docID"].dropna().astype(str))


def is_permanent_fetch_error(error: str) -> bool:
    normalized = error.lower()
    return "404" in normalized and "not found" in normalized


def load_permanent_failed_fetch_doc_ids(
    path: Path = FAILED_FETCH_DOCS_PATH,
) -> set[str]:
    return {
        doc_id
        for doc_id, error in load_failed_fetch_docs(path).items()
        if is_permanent_fetch_error(error)
    }


def write_failed_fetch_docs(
    failures: dict[str, str],
    path: Path = FAILED_FETCH_DOCS_PATH,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "docID": doc_id,
            "error": error,
            "updatedAt": pd.Timestamp.now(tz="UTC").isoformat(),
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


def clear_failed_fetch_docs_for_completed(
    completed_doc_ids: set[str],
    path: Path = FAILED_FETCH_DOCS_PATH,
) -> int:
    failures = load_failed_fetch_docs(path)
    retained = {
        doc_id: error
        for doc_id, error in failures.items()
        if doc_id not in completed_doc_ids
    }
    removed_count = len(failures) - len(retained)
    if removed_count:
        write_failed_fetch_docs(retained, path)
    return removed_count


def complete_marker_path(out_root: Path, docID: str) -> Path:
    return out_root / ".complete" / f"{docID}.complete"


def document_manifest_path(out_root: Path, docID: str) -> Path:
    return out_root / ".complete" / f"{docID}.json"


def build_completed_docid_set(
    out_root: Path,
    doc_type_by_id: Mapping[str, str] | None = None,
) -> set[str]:
    del doc_type_by_id
    if not out_root.exists():
        return set()

    completed = {
        marker.stem
        for marker in (out_root / ".complete").glob("*.complete")
    }
    for manifest_path in (out_root / ".complete").glob("*.json"):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if manifest.get("complete") is True:
            completed.add(manifest_path.stem)
    # Existing CSVs are authoritative for download-state recovery. This also
    # covers files created before the completion-marker mechanism existed.
    completed.update(
        csv_path.stem.split("__", maxsplit=1)[0]
        for csv_path in out_root.glob("*/*.csv")
        if csv_path.parent.name != "jpaud"
    )
    return completed


def docid_already_fetched(docID: str, fetched_docids: set[str]) -> bool:
    """
    事前に収集した CSV ファイル一覧から docID の有無を判定する
    """
    return docID in fetched_docids


def persist_doc_csvs(
    docID: str,
    doc_type_code: str | int | None,
    fetched_document: FetchedDocument,
    out_root: Path,
) -> bool:
    del doc_type_code
    temp_root = out_root / ".tmp" / f"{docID}.{os.getpid()}.{time.time_ns()}"
    canonical_body_csv = (
        fetched_document.body_csv_files[0]
        if fetched_document.body_csv_files
        else None
    )
    saved_csv_metadata: list[dict[str, str | bool]] = []
    try:
        temp_root.mkdir(parents=True, exist_ok=True)
        for csv in fetched_document.csvs:
            destination_name = (
                f"{docID}.csv"
                if csv.source_name == canonical_body_csv
                else f"{docID}__{csv.source_name}"
            )
            temp_dir = temp_root / csv.prefix
            temp_dir.mkdir(parents=True, exist_ok=True)
            csv.dataframe.to_csv(temp_dir / destination_name, index=False)
            saved_csv_metadata.append(
                {
                    "sourceName": csv.source_name,
                    "prefix": csv.prefix,
                    "isBody": csv.is_body,
                    "path": f"{csv.prefix}/{destination_name}",
                }
            )

        for csv_metadata in saved_csv_metadata:
            relative_path = Path(str(csv_metadata["path"]))
            prefix = relative_path.parent.name
            destination_name = relative_path.name
            final_dir = out_root / prefix
            final_dir.mkdir(parents=True, exist_ok=True)
            os.replace(
                temp_root / relative_path,
                final_dir / destination_name,
            )

        manifest = {
            "docID": docID,
            "complete": bool(fetched_document.body_csv_files),
            "bodyPrefixes": list(fetched_document.body_prefixes),
            "bodyCsvFiles": list(fetched_document.body_csv_files),
            "hasJpdeiCor": fetched_document.has_jpdei_cor,
            "csvFiles": saved_csv_metadata,
            "updatedAt": pd.Timestamp.now(tz="UTC").isoformat(),
        }
        manifest_path = document_manifest_path(out_root, docID)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_temp_path = temp_root / f"{docID}.json"
        manifest_temp_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(manifest_temp_path, manifest_path)

        if not fetched_document.body_csv_files:
            raise ValueError(
                f"{docID}: ZIP contained no body CSV under XBRL_TO_CSV"
            )

        marker_path = complete_marker_path(out_root, docID)
        marker_path.parent.mkdir(parents=True, exist_ok=True)
        marker_path.write_text(
            pd.Timestamp.now(tz="UTC").isoformat(),
            encoding="utf-8",
        )
        return True
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


# -----------------------------
# main
# -----------------------------
def _load_fetch_listup() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in (LISTUP_PATH, UPDATED_LISTUP_PATH):
        if path.exists():
            frames.append(pd.read_csv(path, low_memory=False))
    if not frames:
        raise FileNotFoundError(
            f"Neither {LISTUP_PATH} nor {UPDATED_LISTUP_PATH} exists."
        )
    merged = pd.concat(frames, ignore_index=True)
    if "docID" not in merged.columns:
        raise ValueError("EDINET listup is missing docID.")
    return merged.drop_duplicates(subset=["docID"], keep="last")


def main() -> bool:
    """Fetch missing documents and return whether local document data changed."""
    out_root = Path("./doc")
    out_root.mkdir(parents=True, exist_ok=True)
    listup_df = _load_fetch_listup()
    listup_df["docID"] = listup_df["docID"].astype(str)
    fetched_docids = build_completed_docid_set(out_root, {})
    removed_completed_failures = clear_failed_fetch_docs_for_completed(
        fetched_docids
    )
    if removed_completed_failures:
        tqdm.write(
            f"[INFO] cleared {removed_completed_failures} stale failures "
            "for completed documents"
        )
    permanent_failed_doc_ids = load_permanent_failed_fetch_doc_ids()
    all_doc_ids = listup_df["docID"].dropna().astype(str).tolist()
    # A downloaded marker takes precedence over an old failure row. The
    # failure ledger remains an audit trail, but must not trigger a duplicate
    # download when data is already present locally.
    docIDs = [
        doc_id
        for doc_id in all_doc_ids
        if doc_id not in permanent_failed_doc_ids
        and doc_id not in fetched_docids
    ]

    # Keep the failure ledger in memory during the fetch loop. Re-reading and
    # rewriting the entire ledger after every successful document makes a
    # large run dominated by local CSV I/O. New failures are still persisted
    # immediately; successful-document cleanup is flushed on exit, including
    # KeyboardInterrupt, so resume behavior remains unchanged.
    failure_ledger = load_failed_fetch_docs()
    failure_ledger_dirty = False
    fetched_count = 0
    try:
        for docID in tqdm(docIDs, desc="Fetching per docID"):

            # 既存チェック
            if docid_already_fetched(docID, fetched_docids):
                if docID in failure_ledger:
                    del failure_ledger[docID]
                    failure_ledger_dirty = True
                tqdm.write(f"[SKIP] {docID}: already exists")
                continue

            try:
                fetched_document = fetch_all_prefix_dfs(docID)
                persist_doc_csvs(
                    docID,
                    None,
                    fetched_document,
                    out_root,
                )
                fetched_docids.add(docID)
                fetched_count += 1
                if docID in failure_ledger:
                    del failure_ledger[docID]
                    failure_ledger_dirty = True

            except Exception as e:
                error = str(e)
                if failure_ledger.get(docID) != error:
                    failure_ledger[docID] = error
                    write_failed_fetch_docs(failure_ledger)
                    failure_ledger_dirty = False
                tqdm.write(f"[WARN] {docID}: {e}")
    finally:
        if failure_ledger_dirty:
            write_failed_fetch_docs(failure_ledger)

    return fetched_count > 0


if __name__ == "__main__":
    main()
