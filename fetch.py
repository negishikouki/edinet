import io
import json
import os
import shutil
import time
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

from edinet_constants import get_api_key
from normalization import is_missing_scalar
from provenance import file_identity, sha256_bytes, sha256_file, write_bytes_atomic


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
FETCH_CLASSIFICATIONS_PATH = STATE_ROOT / "fetch_classifications.csv"
FETCH_CLASSIFICATION_COLUMNS = [
    "docID",
    "csvFlag",
    "xbrlFlag",
    "classification",
    "error",
    "updatedAt",
]

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
    raw_zip: bytes = b""


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
        except ValueError:
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
        raw_zip=res.content,
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


def normalize_availability_flag(value: object) -> str | None:
    if is_missing_scalar(value):
        return None
    normalized = str(value).strip()
    normalized = normalized.removesuffix(".0")
    return normalized if normalized in {"0", "1"} else None


def classify_document_capability(
    csv_flag: object,
    xbrl_flag: object,
) -> str:
    csv_value = normalize_availability_flag(csv_flag)
    xbrl_value = normalize_availability_flag(xbrl_flag)
    if csv_value == "1":
        return "csv_available"
    if csv_value == "0" and xbrl_value == "1":
        return "xbrl_only_no_csv"
    if csv_value == "0" and xbrl_value == "0":
        return "no_csv_or_xbrl"
    if csv_value == "0":
        return "csv_unavailable"
    return "capability_unknown"


def is_not_found_error(error: str) -> bool:
    normalized = error.lower()
    return "404" in normalized and "not found" in normalized


def is_permanent_fetch_error(
    error: str,
    *,
    csv_flag: object = None,
    xbrl_flag: object = None,
) -> bool:
    del xbrl_flag
    return is_not_found_error(error) and normalize_availability_flag(csv_flag) == "0"


def load_permanent_failed_fetch_doc_ids(
    path: Path = FAILED_FETCH_DOCS_PATH,
    capabilities: Mapping[str, tuple[object, object]] | None = None,
) -> set[str]:
    result: set[str] = set()
    for doc_id, error in load_failed_fetch_docs(path).items():
        csv_flag, xbrl_flag = (capabilities or {}).get(doc_id, (None, None))
        if is_permanent_fetch_error(
            error,
            csv_flag=csv_flag,
            xbrl_flag=xbrl_flag,
        ):
            result.add(doc_id)
    return result


def load_quarantined_failed_fetch_doc_ids(
    path: Path = FAILED_FETCH_DOCS_PATH,
    capabilities: Mapping[str, tuple[object, object]] | None = None,
) -> set[str]:
    result: set[str] = set()
    for doc_id, error in load_failed_fetch_docs(path).items():
        if not is_not_found_error(error):
            continue
        csv_flag, _ = (capabilities or {}).get(doc_id, (None, None))
        if normalize_availability_flag(csv_flag) != "1":
            result.add(doc_id)
    return result


def write_fetch_classifications(
    classifications: dict[str, dict[str, object]],
    path: Path = FETCH_CLASSIFICATIONS_PATH,
) -> None:
    rows = [
        {
            "docID": doc_id,
            "csvFlag": row.get("csvFlag"),
            "xbrlFlag": row.get("xbrlFlag"),
            "classification": row.get("classification"),
            "error": row.get("error"),
            "updatedAt": row.get("updatedAt"),
        }
        for doc_id, row in sorted(classifications.items())
    ]
    write_bytes_atomic(
        path,
        pd.DataFrame(rows, columns=FETCH_CLASSIFICATION_COLUMNS)
        .to_csv(index=False)
        .encode("utf-8"),
    )


def load_fetch_classifications(
    path: Path = FETCH_CLASSIFICATIONS_PATH,
) -> dict[str, dict[str, object]]:
    if not path.is_file():
        return {}
    frame = pd.read_csv(path, dtype=str)
    result: dict[str, dict[str, object]] = {}
    for row in frame.to_dict("records"):
        doc_id = row.get("docID")
        if is_missing_scalar(doc_id):
            continue
        result[str(doc_id)] = {str(key): value for key, value in row.items()}
    return result


def record_fetch_classification(
    doc_id: str,
    *,
    csv_flag: object,
    xbrl_flag: object,
    classification: str,
    error: str,
    classifications: dict[str, dict[str, object]],
) -> None:
    classifications[doc_id] = {
        "csvFlag": normalize_availability_flag(csv_flag),
        "xbrlFlag": normalize_availability_flag(xbrl_flag),
        "classification": classification,
        "error": error,
        "updatedAt": pd.Timestamp.now(tz="UTC").isoformat(),
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
        if is_missing_scalar(doc_id):
            continue
        error = row.get("error")
        failures[str(doc_id)] = "" if is_missing_scalar(error) else str(error)
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

    manifest_paths = list((out_root / ".complete").glob("*.json"))
    manifest_doc_ids = {path.stem for path in manifest_paths}
    completed = {
        marker.stem
        for marker in (out_root / ".complete").glob("*.complete")
        if marker.stem not in manifest_doc_ids
    }
    for manifest_path in manifest_paths:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if manifest.get("complete") is True and _document_manifest_hashes_match(
            out_root, manifest
        ):
            completed.add(manifest_path.stem)
    # Existing CSVs are authoritative for download-state recovery. This also
    # covers files created before the completion-marker mechanism existed.
    completed.update(
        doc_id
        for csv_path in out_root.glob("*/*.csv")
        if csv_path.parent.name != "jpaud"
        for doc_id in [csv_path.stem.split("__", maxsplit=1)[0]]
        if doc_id not in manifest_doc_ids
    )
    return completed


def _artifact_hash_matches(out_root: Path, identity: object) -> bool:
    if not isinstance(identity, dict):
        return False
    path_value = identity.get("path")
    size_value = identity.get("size")
    hash_value = identity.get("sha256")
    if not isinstance(path_value, str) or not isinstance(hash_value, str):
        return False
    path = out_root / Path(path_value)
    return (
        path.is_file()
        and path.stat().st_size == size_value
        and sha256_file(path) == hash_value
    )


def _document_manifest_hashes_match(
    out_root: Path,
    manifest: dict[str, object],
) -> bool:
    csv_files = manifest.get("csvFiles")
    if not isinstance(csv_files, list) or not csv_files:
        return False
    if not all(_artifact_hash_matches(out_root, identity) for identity in csv_files):
        return False
    raw_zip = manifest.get("rawZip")
    return raw_zip is None or _artifact_hash_matches(out_root, raw_zip)


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
    saved_csv_metadata: list[dict[str, object]] = []
    raw_zip_metadata: dict[str, object] | None = None
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
            temporary_csv_path = temp_dir / destination_name
            csv.dataframe.to_csv(temporary_csv_path, index=False)
            csv_identity = file_identity(temporary_csv_path, root=temp_root)
            saved_csv_metadata.append(
                {
                    "sourceName": csv.source_name,
                    "prefix": csv.prefix,
                    "isBody": csv.is_body,
                    "path": f"{csv.prefix}/{destination_name}",
                    "size": csv_identity["size"],
                    "sha256": csv_identity["sha256"],
                }
            )

        if fetched_document.raw_zip:
            temporary_raw_zip = temp_root / ".raw_zip" / f"{docID}.zip"
            temporary_raw_zip.parent.mkdir(parents=True, exist_ok=True)
            temporary_raw_zip.write_bytes(fetched_document.raw_zip)
            raw_zip_metadata = {
                "path": f".raw_zip/{docID}.zip",
                "size": len(fetched_document.raw_zip),
                "sha256": sha256_bytes(fetched_document.raw_zip),
            }

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

        if raw_zip_metadata is not None:
            relative_raw_zip = Path(str(raw_zip_metadata["path"]))
            final_raw_zip = out_root / relative_raw_zip
            final_raw_zip.parent.mkdir(parents=True, exist_ok=True)
            os.replace(temp_root / relative_raw_zip, final_raw_zip)

        manifest = {
            "docID": docID,
            "complete": bool(fetched_document.body_csv_files),
            "bodyPrefixes": list(fetched_document.body_prefixes),
            "bodyCsvFiles": list(fetched_document.body_csv_files),
            "hasJpdeiCor": fetched_document.has_jpdei_cor,
            "rawZip": raw_zip_metadata,
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
    capabilities: dict[str, tuple[object, object]] = {}
    for row in listup_df.to_dict("records"):
        doc_id = row.get("docID")
        if is_missing_scalar(doc_id):
            continue
        capabilities[str(doc_id)] = (
            row.get("csvFlag"),
            row.get("xbrlFlag"),
        )
    fetched_docids = build_completed_docid_set(out_root, {})
    removed_completed_failures = clear_failed_fetch_docs_for_completed(
        fetched_docids
    )
    if removed_completed_failures:
        tqdm.write(
            f"[INFO] cleared {removed_completed_failures} stale failures "
            "for completed documents"
        )
    quarantined_failed_doc_ids = load_quarantined_failed_fetch_doc_ids(
        capabilities=capabilities
    )
    all_doc_ids = listup_df["docID"].dropna().astype(str).tolist()
    metadata_unavailable_doc_ids = {
        doc_id
        for doc_id in all_doc_ids
        if normalize_availability_flag(capabilities.get(doc_id, (None, None))[0])
        == "0"
    }
    # A downloaded marker takes precedence over an old failure row. The
    # failure ledger remains an audit trail, but must not trigger a duplicate
    # download when data is already present locally.
    docIDs = [
        doc_id
        for doc_id in all_doc_ids
        if doc_id not in metadata_unavailable_doc_ids
        and doc_id not in quarantined_failed_doc_ids
        and doc_id not in fetched_docids
    ]

    # Keep the failure ledger in memory during the fetch loop. Re-reading and
    # rewriting the entire ledger after every successful document makes a
    # large run dominated by local CSV I/O. New failures are still persisted
    # immediately; successful-document cleanup is flushed on exit, including
    # KeyboardInterrupt, so resume behavior remains unchanged.
    failure_ledger = load_failed_fetch_docs()
    failure_ledger_dirty = False
    classifications = load_fetch_classifications()
    classifications_dirty = False
    for doc_id in sorted(metadata_unavailable_doc_ids - fetched_docids):
        csv_flag, xbrl_flag = capabilities.get(doc_id, (None, None))
        capability = classify_document_capability(csv_flag, xbrl_flag)
        prior = classifications.get(doc_id)
        if prior is None or prior.get("classification") != capability:
            record_fetch_classification(
                doc_id,
                csv_flag=csv_flag,
                xbrl_flag=xbrl_flag,
                classification=capability,
                error="CSV download skipped from EDINET list metadata",
                classifications=classifications,
            )
            classifications_dirty = True
        if doc_id in failure_ledger:
            del failure_ledger[doc_id]
            failure_ledger_dirty = True
    for doc_id in sorted(quarantined_failed_doc_ids - metadata_unavailable_doc_ids):
        csv_flag, xbrl_flag = capabilities.get(doc_id, (None, None))
        prior = classifications.get(doc_id)
        if prior is None or prior.get("classification") != "unknown_flag_404":
            record_fetch_classification(
                doc_id,
                csv_flag=csv_flag,
                xbrl_flag=xbrl_flag,
                classification="unknown_flag_404",
                error=failure_ledger.get(doc_id, "404 Not Found"),
                classifications=classifications,
            )
            classifications_dirty = True
    fetched_count = 0
    try:
        for docID in tqdm(docIDs, desc="Fetching per docID"):

            # 既存チェック
            if docid_already_fetched(docID, fetched_docids):
                if docID in failure_ledger:
                    del failure_ledger[docID]
                    failure_ledger_dirty = True
                if docID in classifications:
                    del classifications[docID]
                    classifications_dirty = True
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
                if docID in classifications:
                    del classifications[docID]
                    classifications_dirty = True

            except (
                OSError,
                UnicodeError,
                ValueError,
                zipfile.BadZipFile,
                requests.exceptions.RequestException,
            ) as e:
                error = str(e)
                if is_not_found_error(error):
                    csv_flag, xbrl_flag = capabilities.get(docID, (None, None))
                    record_fetch_classification(
                        docID,
                        csv_flag=csv_flag,
                        xbrl_flag=xbrl_flag,
                        classification=(
                            "unexpected_404_csv_available"
                            if normalize_availability_flag(csv_flag) == "1"
                            else "unknown_flag_404"
                        ),
                        error=error,
                        classifications=classifications,
                    )
                    write_fetch_classifications(classifications)
                    classifications_dirty = False
                if failure_ledger.get(docID) != error:
                    failure_ledger[docID] = error
                    write_failed_fetch_docs(failure_ledger)
                    failure_ledger_dirty = False
                tqdm.write(f"[WARN] {docID}: {e}")
    finally:
        if failure_ledger_dirty:
            write_failed_fetch_docs(failure_ledger)
        if classifications_dirty:
            write_fetch_classifications(classifications)

    return fetched_count > 0


if __name__ == "__main__":
    main()
