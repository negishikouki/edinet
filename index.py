from __future__ import annotations

import io
import csv
import os
import shutil
from collections import Counter
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import polars as pl
from tqdm import tqdm

from doc_type_code import DOC_TYPE_PERIOD
from edinet_constants import LISTUP_COLUMNS
from normalization import normalize_sec_code_expr

# =========================
# 設定
# =========================
LISTUP_CSV = Path("./listup.csv")
UPDATED_LISTUP_CSV = Path("./updated_listup.csv")
DOC_ROOT = Path("./doc")  # docID.csv を再帰探索するルート
OUT_ROOT = Path("./index")  # secCode別の出力先
LISTUP_SCHEMA = {
    "date": pl.Utf8,
    "docID": pl.Utf8,
    "edinetCode": pl.Utf8,
    "secCode": pl.Utf8,
    "filerName": pl.Utf8,
    "docTypeCode": pl.UInt32,
    "submitDateTime": pl.Utf8,
    "listQueryDate": pl.Utf8,
    "opeDateTime": pl.Utf8,
    "docInfoEditStatus": pl.Utf8,
    "withdrawalStatus": pl.Utf8,
    "disclosureStatus": pl.Utf8,
    "csvFlag": pl.Utf8,
    "xbrlFlag": pl.Utf8,
    "pdfFlag": pl.Utf8,
    "englishDocFlag": pl.Utf8,
}

# EDINET（jppfs_cor）最小セット
# EPS
EID_PROFIT_TO_OWNERS = (
    "jppfs_cor:ProfitLossAttributableToOwnersOfParent"  # 当期純利益（親会社株主帰属）
)
EID_AVG_SHARES = "jppfs_cor:AverageNumberOfShares"  # 期中平均株式数
EID_B_ELPS = "jpcrp_cor:BasicEarningsLossPerShareSummaryOfBusinessResults"  # １株当たり当期純利益又は当期純損失（△）、経営指標等
EID_D_ELPS = "jpcrp_cor:DilutedEarningsPerShareSummaryOfBusinessResults"  # 潜在株式調整後１株当たり当期純利益、経営指標等

# BPS
EID_NET_ASSETS_PER_SHARE = "jpcrp_cor:NetAssetsPerShareSummaryOfBusinessResults"  # １株当たり純資産額、経営指標等
EID_NET_ASSETS = "jppfs_cor:NetAssets"  # 純資産合計
EID_SHARES_OUT = "jppfs_cor:NumberOfSharesOutstanding"  # 期末発行済株式数

# 営業CF（J-GAAP / IFRS / JMIS / US GAAP）
EIDS_OCF = [
    "jpcrp_cor:NetCashProvidedByUsedInOperatingActivitiesSummaryOfBusinessResults",  # J-GAAP（経営指標等）
    "jpcrp_cor:CashFlowsFromUsedInOperatingActivitiesIFRSSummaryOfBusinessResults",  # IFRS（経営指標等）
    "jpcrp_cor:CashFlowsFromUsedInOperatingActivitiesJMISSummaryOfBusinessResults",  # JMIS（経営指標等）
    "jpcrp_cor:CashFlowsFromUsedInOperatingActivitiesUSGAAPSummaryOfBusinessResults",  # US GAAP（経営指標等）
    "jppfs_cor:NetCashProvidedByUsedInOperatingActivities",  # J-GAAP（CF本表）
    "jpigp_cor:NetCashProvidedByUsedInOperatingActivitiesIFRS",  # IFRS（CF本表）
]

# 投資CF（J-GAAP / IFRS / JMIS / US GAAP）
EIDS_ICF = [
    "jpcrp_cor:NetCashProvidedByUsedInInvestingActivitiesSummaryOfBusinessResults",  # J-GAAP（経営指標等）
    "jpcrp_cor:CashFlowsFromUsedInInvestingActivitiesIFRSSummaryOfBusinessResults",  # IFRS（経営指標等）
    "jpcrp_cor:CashFlowsFromUsedInInvestingActivitiesJMISSummaryOfBusinessResults",  # JMIS（経営指標等）
    "jpcrp_cor:CashFlowsFromUsedInInvestingActivitiesUSGAAPSummaryOfBusinessResults",  # US GAAP（経営指標等）
    "jppfs_cor:NetCashProvidedByUsedInInvestingActivities",  # J-GAAP（CF本表）
    "jpigp_cor:NetCashProvidedByUsedInInvestingActivitiesIFRS",  # IFRS（CF本表）
]

# 財務CF（FCFE向け）
EIDS_NET_INCREASE_DECREASE_ST_LOANS = [
    "jppfs_cor:NetIncreaseDecreaseInShortTermLoansPayableFinCF",  # J-GAAP
    "jpigp_cor:NetIncreaseDecreaseInShortTermBorrowingsFinCFIFRS",  # IFRS
]
EIDS_PROCEEDS_LT_LOANS = [
    "jppfs_cor:ProceedsFromLongTermLoansPayableFinCF",  # J-GAAP
    "jpigp_cor:ProceedsFromLongTermBorrowingsFinCFIFRS",  # IFRS
    "jpigp_cor:ProceedsFromIssuanceOfBondsAndLongTermBorrowingsFinCFIFRS",  # IFRS（代替）
]
EIDS_REPAYMENT_LT_LOANS = [
    "jppfs_cor:RepaymentOfLongTermLoansPayableFinCF",  # J-GAAP
    "jpigp_cor:RepaymentsOfLongTermBorrowingsFinCFIFRS",  # IFRS
    "jpigp_cor:RedemptionOfBondsAndRepaymentsOfLongTermBorrowingsFinCFIFRS",  # IFRS（代替）
]

# 投資（CAPEX）
EIDS_CAPEX_COMBINED = [
    "jpigp_cor:PurchaseOfPropertyPlantAndEquipmentAndIntangibleAssetsInvCFIFRS",  # IFRS（PPE+無形合算）
]
EIDS_PPE_PURCHASE = [
    "jppfs_cor:PurchaseOfPropertyPlantAndEquipmentInvCF",  # J-GAAP
    "jpigp_cor:PurchaseOfPropertyPlantAndEquipmentInvCFIFRS",  # IFRS
]
EIDS_INTANGIBLE_PURCHASE = [
    "jppfs_cor:PurchaseOfIntangibleAssetsInvCF",  # J-GAAP
    "jpigp_cor:PurchaseOfIntangibleAssetsInvCFIFRS",  # IFRS
]

# Issued Shares
EID_TNOIS = "jpcrp_cor:TotalNumberOfIssuedSharesSummaryOfBusinessResults"  # 発行済株式総数（普通株式）、経営指標等

# コンテキストID（期間）
CID_CurrentYTDDuration = "CurrentYTDDuration"  # 当四半期累計期間
CID_Prior1YTDDuration = "Prior1YTDDuration"  # 前年度同四半期累計期間
CID_Prior1YearDuration = "Prior1YearDuration"  # 前期
CID_CurrentYTDDuration_NonConsolidated = (
    "CurrentYTDDuration_NonConsolidated"  # 当四半期累計期間
)
CID_Prior1YTDDuration_NonConsolidated = (
    "Prior1YTDDuration_NonConsolidated"  # 前年度同四半期累計期間
)
CID_Prior1YearDuration_NonConsolidated = "Prior1YearDuration_NonConsolidated"  # 前期


# -----------------------------
# Instant（時点）
# -----------------------------
CID_CurrentYearInstant = "CurrentYearInstant"  # 当期末
CID_Prior4YearInstant = "Prior4YearInstant"  # 四期前時点
CID_Prior3YearInstant = "Prior3YearInstant"  # 三期前時点
CID_Prior2YearInstant = "Prior2YearInstant"  # 前々期末
CID_Prior1YearInstant = "Prior1YearInstant"  # 前期末
CID_InterimInstant = "InterimInstant"  # 当中間期末
CID_Prior4InterimInstant = "Prior4InterimInstant"  # 四期中間期末
CID_Prior3InterimInstant = "Prior3InterimInstant"  # 三期中間期末
CID_Prior2InterimInstant = "Prior2InterimInstant"  # 前々中間期末
CID_Prior1InterimInstant = "Prior1InterimInstant"  # 前中間期末


# -----------------------------
# Instant（時点・非連結）
# -----------------------------
CID_CurrentYearInstant_NonConsolidatedMember = (
    "CurrentYearInstant_NonConsolidatedMember"  # 当期末
)
CID_Prior4YearInstant_NonConsolidatedMember = (
    "Prior4YearInstant_NonConsolidatedMember"  # 四期前時点
)
CID_Prior3YearInstant_NonConsolidatedMember = (
    "Prior3YearInstant_NonConsolidatedMember"  # 三期前時点
)
CID_Prior2YearInstant_NonConsolidatedMember = (
    "Prior2YearInstant_NonConsolidatedMember"  # 前々期末
)
CID_Prior1YearInstant_NonConsolidatedMember = (
    "Prior1YearInstant_NonConsolidatedMember"  # 前期末
)
CID_InterimInstant_NonConsolidatedMember = (
    "InterimInstant_NonConsolidatedMember"  # 当中間期末
)
CID_Prior4InterimInstant_NonConsolidatedMember = (
    "Prior4InterimInstant_NonConsolidatedMember"  # 四期前期間末末
)
CID_Prior3InterimInstant_NonConsolidatedMember = (
    "Prior3InterimInstant_NonConsolidatedMember"  # 三期前期間末
)
CID_Prior2InterimInstant_NonConsolidatedMember = (
    "Prior2InterimInstant_NonConsolidatedMember"  # 前々中間期末
)
CID_Prior1InterimInstant_NonConsolidatedMember = (
    "Prior1InterimInstant_NonConsolidatedMember"  # 前中間期末
)


# -----------------------------
# Duration（期間・連結）
# -----------------------------
CID_Prior4YearDuration = "Prior4YearDuration"  # 四期前期間
CID_Prior3YearDuration = "Prior3YearDuration"  # 三期前期間
CID_Prior2YearDuration = "Prior2YearDuration"  # 前々期期間
CID_Prior1YearDuration = "Prior1YearDuration"  # 前期期間
CID_CurrentYearDuration = "CurrentYearDuration"  # 当期期間
CID_InterimDuration = "InterimDuration"  # 当中間期末
CID_Prior4InterimDuration = "Prior4InterimDuration"  # 四期中間期
CID_Prior3InterimDuration = "Prior3InterimDuration"  # 三期中間期
CID_Prior2InterimDuration = "Prior2InterimDuration"  # 前々中間期
CID_Prior1InterimDuration = "Prior1InterimDuration"  # 前中間期


# -----------------------------
# Duration（期間・非連結）
# -----------------------------
CID_Prior4YearDuration_NonConsolidated = (
    "Prior4YearDuration_NonConsolidatedMember"  # 四期前期間（非連結）
)
CID_Prior3YearDuration_NonConsolidated = (
    "Prior3YearDuration_NonConsolidatedMember"  # 三期前期間（非連結）
)
CID_Prior2YearDuration_NonConsolidated = (
    "Prior2YearDuration_NonConsolidatedMember"  # 前々期期間（非連結）
)
CID_Prior1YearDuration_NonConsolidated = (
    "Prior1YearDuration_NonConsolidatedMember"  # 前期期間（非連結）
)
CID_CurrentYearDuration_NonConsolidated = (
    "CurrentYearDuration_NonConsolidatedMember"  # 当期期間（非連結）
)
CID_InterimDuration_NonConsolidatedMember = (
    "InterimDuration_NonConsolidatedMember"  # 当中間期末
)
CID_Prior4InterimDuration_NonConsolidatedMember = (
    "Prior4InterimDuration_NonConsolidatedMember"  # 四期中間期
)
CID_Prior3InterimDuration_NonConsolidatedMember = (
    "Prior3InterimDuration_NonConsolidatedMember"  # 三期中間期
)
CID_Prior2InterimDuration_NonConsolidatedMember = (
    "Prior2InterimDuration_NonConsolidatedMember"  # 前々中間期
)
CID_Prior1InterimDuration_NonConsolidatedMember = (
    "Prior1InterimDuration_NonConsolidatedMember"  # 前中間期
)


# コンテキストID（時点・個別）
CID_Prior4YearInstant_NonConsolidated = (
    "Prior4YearInstant_NonConsolidatedMember"  # 四期前時点（個別）
)
CID_Prior3YearInstant_NonConsolidated = (
    "Prior3YearInstant_NonConsolidatedMember"  # 三期前時点（個別）
)
CID_Prior2YearInstant_NonConsolidated = (
    "Prior2YearInstant_NonConsolidatedMember"  # 前々期末（個別）
)
CID_Prior1YearInstant_NonConsolidated = (
    "Prior1YearInstant_NonConsolidatedMember"  # 前期末（個別）
)
CID_CurrentYearInstant_NonConsolidated = (
    "CurrentYearInstant_NonConsolidatedMember"  # 当期末（個別）
)


STATE_ROOT = Path("state")
FAILED_INDEX_DOCS_PATH = STATE_ROOT / "failed_index_docs.csv"
FAILED_INDEX_SEC_CODES_PATH = STATE_ROOT / "failed_index_sec_codes.csv"
FAILED_INDEX_DOC_COLUMNS = ["docID", "secCode", "path", "error", "updatedAt"]
FAILED_INDEX_SEC_CODE_COLUMNS = [
    "secCode",
    "phase",
    "error",
    "summaryPolicy",
    "updatedAt",
]


@dataclass(frozen=True)
class IndexDocReadFailure:
    doc_id: str
    sec_code: str
    path: str
    error: str


@dataclass(frozen=True)
class SecCodeProcessResult:
    sec_code: str
    doc_read_failures: list[IndexDocReadFailure]
    wrote_output: bool


# =========================
# ユーティリティ
# =========================
def build_doc_index(doc_root: Path) -> Dict[str, Path]:
    """
    ./doc 以下（サブディレクトリ含む）にある *.csv を再帰走査して、
    ファイル名（例: S100XXXX.csv） -> フルパス の辞書を作る。
    同名が複数ある場合は先に見つかったものを採用（必要なら変更可）。
    """
    index: Dict[str, Path] = {}
    if not doc_root.exists():
        return index

    for p in doc_root.rglob("*.csv"):
        name = p.name
        if name not in index:
            index[name] = p
    return index


def _write_csv_utf8sig_direct(df: pl.DataFrame, path: Path) -> None:
    try:
        df.write_csv(path, include_bom=True)
        return
    except TypeError:
        pass

    csv_bytes: bytes | None = None
    try:
        csv_output = df.write_csv()
        if isinstance(csv_output, bytes):
            csv_bytes = csv_output
        elif isinstance(csv_output, str):
            csv_bytes = csv_output.encode("utf-8")
    except TypeError:
        pass

    if csv_bytes is None:
        text_buf = io.StringIO()
        df.write_csv(text_buf)
        csv_bytes = text_buf.getvalue().encode("utf-8")

    with open(path, "wb") as f:
        f.write(b"\xef\xbb\xbf")
        f.write(csv_bytes)


def write_csv_utf8sig(df: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        _write_csv_utf8sig_direct(df, temp_path)
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _read_state_rows(path: Path, key_field: str) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}

    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows: dict[str, dict[str, str]] = {}
        for row in reader:
            key = row.get(key_field)
            if key:
                rows[key] = {k: "" if v is None else v for k, v in row.items()}
        return rows


def _write_state_rows(
    path: Path,
    fieldnames: list[str],
    rows: Iterable[dict[str, str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def record_failed_index_docs(
    failures: Iterable[IndexDocReadFailure],
    path: Path = FAILED_INDEX_DOCS_PATH,
) -> None:
    rows = _read_state_rows(path, "docID")
    updated_at = _utc_now_iso()
    for failure in failures:
        rows[failure.doc_id] = {
            "docID": failure.doc_id,
            "secCode": failure.sec_code,
            "path": failure.path,
            "error": failure.error,
            "updatedAt": updated_at,
        }
    _write_state_rows(path, FAILED_INDEX_DOC_COLUMNS, rows.values())


def clear_failed_index_docs_for_sec_code(
    sec_code: str,
    path: Path = FAILED_INDEX_DOCS_PATH,
) -> None:
    rows = _read_state_rows(path, "docID")
    kept = [
        row
        for row in rows.values()
        if row.get("secCode") != sec_code
    ]
    _write_state_rows(path, FAILED_INDEX_DOC_COLUMNS, kept)


def record_failed_index_sec_code(
    sec_code: str,
    phase: str,
    error: str,
    path: Path = FAILED_INDEX_SEC_CODES_PATH,
) -> None:
    rows = _read_state_rows(path, "secCode")
    rows[sec_code] = {
        "secCode": sec_code,
        "phase": phase,
        "error": error,
        "summaryPolicy": "leave_existing_index_for_summary",
        "updatedAt": _utc_now_iso(),
    }
    _write_state_rows(path, FAILED_INDEX_SEC_CODE_COLUMNS, rows.values())


def clear_failed_index_sec_code(
    sec_code: str,
    path: Path = FAILED_INDEX_SEC_CODES_PATH,
) -> None:
    rows = _read_state_rows(path, "secCode")
    if sec_code not in rows:
        return
    del rows[sec_code]
    _write_state_rows(path, FAILED_INDEX_SEC_CODE_COLUMNS, rows.values())


DOC_CSV_NULL_VALUES = ["", "-", "－", "―", "—"]

_NUMERIC_NULL_TOKENS = {"", "-", "nan", "null", "none", "n/a"}
_FULLWIDTH_NUMERIC_TRANS = str.maketrans(
    {
        "０": "0",
        "１": "1",
        "２": "2",
        "３": "3",
        "４": "4",
        "５": "5",
        "６": "6",
        "７": "7",
        "８": "8",
        "９": "9",
        "．": ".",
        "，": ",",
        "＋": "+",
        "－": "-",
        "−": "-",
        "―": "-",
        "—": "-",
    }
)


def read_listup_csv(path: Path) -> pl.DataFrame:
    frame = pl.read_csv(
        source=path,
        has_header=True,
        null_values=[""],
        infer_schema=False,
        infer_schema_length=0,
    )
    expressions: list[pl.Expr] = []
    for column in LISTUP_COLUMNS:
        dtype = LISTUP_SCHEMA[column]
        if column in frame.columns:
            expressions.append(
                pl.col(column).cast(dtype, strict=False).alias(column)
            )
        else:
            expressions.append(pl.lit(None, dtype=dtype).alias(column))
    return frame.select(expressions)


def load_merged_listup(
    listup_path: Path,
    updated_listup_path: Path,
) -> tuple[pl.DataFrame, pl.DataFrame | None]:
    if not listup_path.exists() and not updated_listup_path.exists():
        raise FileNotFoundError(
            f"{listup_path} と {updated_listup_path} のどちらも存在しません"
        )

    base = read_listup_csv(listup_path) if listup_path.exists() else None
    updated = (
        read_listup_csv(updated_listup_path)
        if updated_listup_path.exists()
        else None
    )
    frames = [frame for frame in (base, updated) if frame is not None]
    merged = pl.concat(frames, how="vertical_relaxed")
    if "docID" in merged.columns:
        merged = merged.unique(subset=["docID"], keep="last", maintain_order=True)
    return merged, updated


def prepare_listup(listup: pl.DataFrame) -> pl.DataFrame:
    required_cols = {"secCode", "date", "docID", "edinetCode", "docTypeCode"}
    missing = required_cols - set(listup.columns)
    if missing:
        raise ValueError(f"listup.csv に必要な列がありません: {sorted(missing)}")

    listup = listup.with_columns(
        normalize_sec_code_expr(pl.col("secCode")).alias("secCode"),
        pl.col("date")
        .cast(pl.Utf8)
        .str.strptime(pl.Datetime, strict=False)
        .alias("date_dt"),
    )

    non_null_sec = listup.filter(pl.col("secCode").is_not_null()).sort(
        ["date_dt", "date"], nulls_last=True
    )
    latest_sec_code_df = non_null_sec.group_by(
        "edinetCode", maintain_order=True
    ).agg(pl.col("secCode").last().alias("latest_secCode"))
    return (
        listup.join(latest_sec_code_df, on="edinetCode", how="left")
        .with_columns(
            pl.coalesce(
                [
                    pl.col("latest_secCode"),
                    pl.col("secCode"),
                ]
            ).alias("secCode")
        )
        .drop("latest_secCode")
        .sort(["secCode", "date_dt", "date"], nulls_last=True)
    )


def select_target_sec_codes(
    merged_listup: pl.DataFrame,
    updated_listup: pl.DataFrame | None,
) -> list[str]:
    if updated_listup is None:
        target_rows = merged_listup
    else:
        updated_edinet_codes = (
            updated_listup.get_column("edinetCode").drop_nulls().unique().to_list()
        )
        updated_sec_codes = (
            updated_listup.select(
                normalize_sec_code_expr(pl.col("secCode")).alias("secCode")
            )
            .get_column("secCode")
            .drop_nulls()
            .unique()
            .to_list()
        )
        target_rows = merged_listup.filter(
            pl.col("edinetCode").is_in(updated_edinet_codes)
            | pl.col("secCode").is_in(updated_sec_codes)
        )

    return [
        str(sec_code)
        for sec_code in target_rows.get_column("secCode")
        .drop_nulls()
        .unique(maintain_order=True)
        .to_list()
    ]


def validate_existing_history_is_preserved(
    out_df: pl.DataFrame,
    out_path: Path,
) -> None:
    if not out_path.exists():
        return

    existing = pl.read_csv(
        out_path,
        columns=["date"],
        schema_overrides={"date": pl.Utf8},
        infer_schema=False,
    )
    existing_dates = Counter(existing.get_column("date").drop_nulls().to_list())
    new_dates = Counter(
        out_df.get_column("date").cast(pl.Utf8).drop_nulls().to_list()
    )
    missing_dates = existing_dates - new_dates
    if missing_dates:
        samples = ", ".join(
            f"{date} ({count}件)" for date, count in missing_dates.most_common(5)
        )
        raise ValueError(
            f"{out_path} の既存履歴が欠落するため置換を中止します: {samples}"
        )


def _normalize_submit_datetime(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return None
    return text


def _fundamental_available_date(
    report_date: str, submit_datetime: str | None
) -> str:
    if submit_datetime:
        candidate = submit_datetime[:10]
        try:
            datetime.strptime(candidate, "%Y-%m-%d")
        except ValueError:
            pass
        else:
            return candidate
    return report_date


def read_doc_csv(path: str | Path) -> pl.DataFrame:
    """
    EDINET doc CSV を文字列優先で読む。
    「－」のような値が混在しても型推論エラーで落ちないようにする。
    """
    return pl.read_csv(
        path,
        infer_schema=False,
        null_values=DOC_CSV_NULL_VALUES,
    )


def _to_float(x) -> Optional[float]:
    """
    値をfloatに変換。空・NaN・変換不能はNone。
    """
    if x is None:
        return None

    if isinstance(x, (int, float)):
        if isinstance(x, float) and x != x:
            return None
        return float(x)

    s = str(x).strip()
    if not s:
        return None

    s = s.translate(_FULLWIDTH_NUMERIC_TRANS)
    s_lower = s.lower()
    if s_lower in _NUMERIC_NULL_TOKENS:
        return None

    negative = False
    if s.startswith(("△", "▲")):
        negative = True
        s = s[1:].strip()
    elif s.startswith("(") and s.endswith(")"):
        negative = True
        s = s[1:-1].strip()

    if s.endswith("%"):
        s = s[:-1].strip()

    s = s.replace(",", "").replace("_", "")
    if not s or s.lower() in _NUMERIC_NULL_TOKENS:
        return None

    try:
        value = float(s)
        return -value if negative else value
    except Exception:
        return None


def _sum_optional(values: Iterable[Optional[float]]) -> Optional[float]:
    found = False
    total = 0.0
    for value in values:
        if value is None:
            continue
        found = True
        total += value
    return total if found else None


def pick_value(
    df: pl.DataFrame,
    element_id: str,
    context_id: str | list | dict | None = None,
    period_type: str | None = None,
    aggregate: str | None = None,
    prefer_consolidated: bool = True,
    prefer_current_year: bool = True,
) -> Optional[float]:
    """
    EDINET CSV（要素ID/値などの列がある）から、指定element_idの値を1つ選んで返す。
    - 連結・個別（列名に "連結・個別" がある想定）: 連結を優先
    - 相対年度（列名に "相対年度" がある想定）: 当期(0)を優先
    条件列が存在しない場合は存在する行の先頭を採用。
    """
    if df.is_empty():
        return None

    # 要素ID列
    eid_col = df.columns[0]
    cid_col = df.columns[2]
    cid_col_expr = pl.col(cid_col).cast(pl.Utf8)

    if isinstance(context_id, dict):
        if period_type and period_type in context_id:
            context_id = context_id[period_type]
        else:
            context_id = context_id.get("default")

    def pick_single_value(sub_df: pl.DataFrame) -> Optional[float]:
        if sub_df.is_empty():
            return None

        if prefer_consolidated and "連結・個別" in sub_df.columns:
            con = sub_df.filter(pl.col("連結・個別").cast(pl.Utf8).str.contains("連結"))
            if not con.is_empty():
                sub_df = con

        if prefer_current_year and "相対年度" in sub_df.columns:
            cur = sub_df.filter(pl.col("相対年度").cast(pl.Utf8).is_in(["0", "当期"]))
            if not cur.is_empty():
                sub_df = cur

        value_col = None
        if "値" in sub_df.columns:
            value_col = "値"
        else:
            for c in reversed(sub_df.columns):
                if c == eid_col:
                    continue
                value_col = c
                break

        if value_col is None:
            return None

        return _to_float(sub_df.get_column(value_col)[0])

    def sum_contexts(contexts: Iterable[str]) -> Optional[float]:
        values: list[float] = []
        for cid in contexts:
            sub_df = df.filter(
                (pl.col(eid_col).cast(pl.Utf8) == element_id) & (cid_col_expr == cid)
            )
            value = pick_single_value(sub_df)
            if value is None:
                return None
            values.append(value)
        return sum(values) if values else None

    if aggregate == "sum" and context_id:
        if (
            isinstance(context_id, list)
            and context_id
            and isinstance(context_id[0], list)
        ):
            for option in context_id:
                value = sum_contexts(option)
                if value is not None:
                    return value
            return None

        if isinstance(context_id, list):
            return sum_contexts(context_id)

        if isinstance(context_id, str):
            return sum_contexts([context_id])

    if context_id:
        if isinstance(context_id, Iterable) and not isinstance(
            context_id, (str, bytes)
        ):
            sub = df.head(0)
            for cid in context_id:
                candidate_sub = df.filter(
                    (pl.col(eid_col).cast(pl.Utf8) == element_id)
                    & (cid_col_expr == cid)
                )
                if not candidate_sub.is_empty():
                    sub = candidate_sub
                    break
        else:
            sub = df.filter(
                (pl.col(eid_col).cast(pl.Utf8) == element_id)
                & (cid_col_expr == context_id)
            )
    else:
        sub = df.filter(pl.col(eid_col).cast(pl.Utf8) == element_id)

    return pick_single_value(sub)


def pick_value_any(
    df: pl.DataFrame,
    element_ids: Iterable[str] | str,
    context_id: str | list | dict | None = None,
    period_type: str | None = None,
    aggregate: str | None = None,
    prefer_consolidated: bool = True,
    prefer_current_year: bool = True,
) -> Optional[float]:
    """
    複数の要素ID候補を順番に評価し、最初に見つかった値を返す。
    """
    if isinstance(element_ids, (str, bytes)):
        element_ids = [element_ids]

    for element_id in element_ids:
        value = pick_value(
            df,
            element_id,
            context_id=context_id,
            period_type=period_type,
            aggregate=aggregate,
            prefer_consolidated=prefer_consolidated,
            prefer_current_year=prefer_current_year,
        )
        if value is not None:
            return value
    return None


def build_issued_shares_contexts() -> dict[str, list[str]]:
    return {
        "annual": [
            CID_CurrentYearInstant,
            CID_Prior1YearInstant,
            CID_Prior2YearInstant,
            CID_Prior3YearInstant,
            CID_Prior4YearInstant,
            CID_CurrentYearInstant_NonConsolidated,
            CID_Prior1YearInstant_NonConsolidated,
            CID_Prior2YearInstant_NonConsolidated,
            CID_Prior3YearInstant_NonConsolidated,
            CID_Prior4YearInstant_NonConsolidated,
        ],
        "semiannual": [
            CID_InterimInstant,
            CID_Prior1InterimInstant,
            CID_Prior2InterimInstant,
            CID_Prior3InterimInstant,
            CID_Prior4InterimInstant,
            CID_InterimInstant_NonConsolidatedMember,
            CID_Prior1InterimInstant_NonConsolidatedMember,
            CID_Prior2InterimInstant_NonConsolidatedMember,
            CID_Prior3InterimInstant_NonConsolidatedMember,
            CID_Prior4InterimInstant_NonConsolidatedMember,
        ],
        "quarterly": [
            CID_CurrentYearInstant,
            CID_InterimInstant,
            CID_Prior1YearInstant,
            CID_Prior1InterimInstant,
            CID_Prior2YearInstant,
            CID_Prior2InterimInstant,
            CID_Prior3YearInstant,
            CID_Prior3InterimInstant,
            CID_Prior4YearInstant,
            CID_Prior4InterimInstant,
            CID_CurrentYearInstant_NonConsolidated,
            CID_InterimInstant_NonConsolidatedMember,
            CID_Prior1YearInstant_NonConsolidated,
            CID_Prior1InterimInstant_NonConsolidatedMember,
            CID_Prior2YearInstant_NonConsolidated,
            CID_Prior2InterimInstant_NonConsolidatedMember,
            CID_Prior3YearInstant_NonConsolidated,
            CID_Prior3InterimInstant_NonConsolidatedMember,
            CID_Prior4YearInstant_NonConsolidated,
            CID_Prior4InterimInstant_NonConsolidatedMember,
        ],
    }


def calc_issued_shares(
    doc_df: pl.DataFrame, period_type: str
) -> Optional[float]:
    """
    doc CSV DataFrame から発行済株式数を取得して返す。
    欠損があれば None。
    """

    shares_context = build_issued_shares_contexts()
    issued_shares = pick_value(
        doc_df,
        EID_TNOIS,
        context_id=shares_context,
        period_type=period_type,
        prefer_consolidated=True,
        prefer_current_year=True,
    )

    if issued_shares is None:
        issued_shares = pick_value(
            doc_df,
            EID_SHARES_OUT,
            context_id=None,
            prefer_consolidated=True,
            prefer_current_year=True,
        )

    if issued_shares is None:
        # 'ユニットID' == 'shares' のなかで、element_idは不問として
        # context_id=shares_context,
        # period_type=period_type,
        # データを取得して、
        # 最大の値をissued_sharesに格納する
        context_ids = shares_context.get(period_type, [])
        if context_ids and len(doc_df.columns) > 6:
            unit_col = (
                "ユニットID" if "ユニットID" in doc_df.columns else doc_df.columns[6]
            )
            value_col = "値" if "値" in doc_df.columns else doc_df.columns[-1]
            cid_col = doc_df.columns[2]

            share_rows = doc_df.filter(
                pl.col(unit_col)
                .cast(pl.Utf8)
                .str.strip_chars()
                .str.to_lowercase()
                .is_in(["shares"])
                & pl.col(cid_col).cast(pl.Utf8).is_in(context_ids)
            )

            if not share_rows.is_empty():
                share_values = [
                    v
                    for v in (
                        _to_float(x) for x in share_rows.get_column(value_col).to_list()
                    )
                    if v is not None
                ]
                if share_values:
                    issued_shares = max(share_values)

    return issued_shares


def calc_eps_bps_cfps(
    doc_df: pl.DataFrame, period_type: str
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    doc CSV DataFrame から EPS, CFPS, BPS を計算して返す。
    欠損があれば None。
    """

    issued_shares = calc_issued_shares(doc_df, period_type)

    ttm_duration_contexts = {
        "annual": [
            [CID_CurrentYearDuration],
            [CID_CurrentYearDuration_NonConsolidated],
        ],
        "semiannual": [
            [CID_InterimDuration, CID_Prior1InterimDuration],
            [
                CID_InterimDuration_NonConsolidatedMember,
                CID_Prior1InterimDuration_NonConsolidatedMember,
            ],
        ],
        "quarterly": [
            [
                CID_CurrentYearDuration,
                CID_Prior1YearDuration,
                CID_Prior2YearDuration,
                CID_Prior3YearDuration,
            ],
            [
                CID_CurrentYearDuration_NonConsolidated,
                CID_Prior1YearDuration_NonConsolidated,
                CID_Prior2YearDuration_NonConsolidated,
                CID_Prior3YearDuration_NonConsolidated,
            ],
        ],
    }

    # -----------------------------
    # EPS
    # -----------------------------
    eps = pick_value(
        doc_df,
        EID_B_ELPS,
        context_id=ttm_duration_contexts,
        period_type=period_type,
        aggregate="sum",
        prefer_consolidated=True,
        prefer_current_year=True,
    )

    if eps is None:
        eps = pick_value(
            doc_df,
            EID_D_ELPS,
            context_id=ttm_duration_contexts,
            period_type=period_type,
            aggregate="sum",
            prefer_consolidated=True,
            prefer_current_year=True,
        )

    if eps is None:
        profit = pick_value(
            doc_df,
            EID_PROFIT_TO_OWNERS,
            context_id=ttm_duration_contexts,
            period_type=period_type,
            aggregate="sum",
            prefer_consolidated=True,
            prefer_current_year=True,
        )
        eps = (
            (profit / issued_shares)
            if (
                profit is not None
                and issued_shares is not None
                and issued_shares != 0.0
            )
            else None
        )

    # -----------------------------
    # BPS
    # -----------------------------
    bps = pick_value(
        doc_df,
        EID_NET_ASSETS_PER_SHARE,
        context_id=[
            CID_CurrentYearInstant,
            CID_Prior4YearInstant,
            CID_Prior3YearInstant,
            CID_Prior2YearInstant,
            CID_Prior1YearInstant,
            CID_CurrentYearInstant_NonConsolidated,
            CID_Prior4YearInstant_NonConsolidated,
            CID_Prior3YearInstant_NonConsolidated,
            CID_Prior2YearInstant_NonConsolidated,
            CID_Prior1YearInstant_NonConsolidated,
        ],
        prefer_consolidated=True,
        prefer_current_year=True,
    )

    if bps is None:
        net_assets = pick_value(
            doc_df,
            EID_NET_ASSETS,
            context_id=None,
            prefer_consolidated=True,
            prefer_current_year=True,
        )
        bps = (
            (net_assets / issued_shares)
            if (
                net_assets is not None
                and issued_shares is not None
                and issued_shares != 0.0
            )
            else None
        )

    # -----------------------------
    # CFPS（Operating + Investing Cash Flow）
    # -----------------------------
    # 営業CF
    ocf = pick_value_any(
        doc_df,
        EIDS_OCF,
        context_id=ttm_duration_contexts,
        period_type=period_type,
        aggregate="sum",
        prefer_consolidated=True,
        prefer_current_year=True,
    )

    # CAPEX（有形＋無形）
    capex = pick_value_any(
        doc_df,
        EIDS_CAPEX_COMBINED,
        context_id=None,
        prefer_consolidated=True,
        prefer_current_year=True,
    )

    if capex is None:
        capex = _sum_optional(
            [
                pick_value_any(
                    doc_df,
                    EIDS_PPE_PURCHASE,
                    context_id=None,
                    prefer_consolidated=True,
                    prefer_current_year=True,
                ),
                pick_value_any(
                    doc_df,
                    EIDS_INTANGIBLE_PURCHASE,
                    context_id=None,
                    prefer_consolidated=True,
                    prefer_current_year=True,
                ),
            ]
        )

    if capex is not None:
        capex = abs(capex)

    # 純借入額
    st_delta = pick_value_any(
        doc_df,
        EIDS_NET_INCREASE_DECREASE_ST_LOANS,
        context_id=None,
        prefer_consolidated=True,
        prefer_current_year=True,
    )

    lt_proceeds = pick_value_any(
        doc_df,
        EIDS_PROCEEDS_LT_LOANS,
        context_id=None,
        prefer_consolidated=True,
        prefer_current_year=True,
    )

    lt_repayment = pick_value_any(
        doc_df,
        EIDS_REPAYMENT_LT_LOANS,
        context_id=None,
        prefer_consolidated=True,
        prefer_current_year=True,
    )

    # 長期借入返済は必ずマイナスであることを要求
    if lt_repayment is not None and lt_repayment > 0:
        raise ValueError(
            f"Invalid sign: RepaymentOfLongTermLoansPayableFinCF must be negative, got {lt_repayment}"
        )

    net_borrowing = _sum_optional(
        [
            st_delta,
            lt_proceeds,
            lt_repayment,  # 既にマイナスで入っている前提
        ]
    )

    oicf = (
        # FCFE = ocf - capex + net_borrowingであるが、
        # oicf = ocf + capex + net_borrowingとする
        # キャッシュフローの大きさを指標化するため
        ocf + capex + net_borrowing
        if (ocf is not None and capex is not None and net_borrowing is not None)
        else None
    )

    cfps = (
        (oicf / issued_shares)
        if (
            oicf is not None
            and issued_shares is not None
            and issued_shares != 0.0
        )
        else None
    )

    if cfps is None:
        icf = pick_value_any(
            doc_df,
            EIDS_ICF,
            context_id=ttm_duration_contexts,
            period_type=period_type,
            aggregate="sum",
            prefer_consolidated=True,
            prefer_current_year=True,
        )

        fcff = _sum_optional([ocf, icf])
        cfps = (
            (fcff / issued_shares)
            if (
                fcff is not None
                and issued_shares is not None
                and issued_shares != 0.0
            )
            else None
        )

    return eps, cfps, bps


def _process_sec_code(
    sec_code: str,
    listup_for_sec_code: pl.DataFrame,
    doc_index: Dict[str, Path],
    out_root: Path,
    protect_existing_history: bool,
    fail_on_doc_read_error: bool = False,
) -> SecCodeProcessResult:
    records = []
    doc_read_failures: list[IndexDocReadFailure] = []
    for r in listup_for_sec_code.iter_rows(named=True):
        date_str = str(r["date"])
        date_dt = r["date_dt"]
        if date_dt is None:
            date_dt = datetime.min
        doc_id = str(r["docID"])
        submit_datetime = _normalize_submit_datetime(r.get("submitDateTime"))
        available_date = _fundamental_available_date(date_str, submit_datetime)
        doc_type_code = str(r["docTypeCode"])
        fname = f"{doc_id}.csv"

        period_type = DOC_TYPE_PERIOD.get(doc_type_code)
        if period_type is None:
            continue

        doc_path = doc_index.get(fname)
        if doc_path is None:
            continue

        try:
            doc_df = read_doc_csv(doc_path)
        except Exception as exc:
            doc_read_failures.append(
                IndexDocReadFailure(
                    doc_id=doc_id,
                    sec_code=sec_code,
                    path=str(doc_path),
                    error=str(exc),
                )
            )
            if fail_on_doc_read_error:
                return SecCodeProcessResult(
                    sec_code=sec_code,
                    doc_read_failures=doc_read_failures,
                    wrote_output=False,
                )
            continue

        issued_shares = calc_issued_shares(doc_df, period_type)
        eps, cfps, bps = calc_eps_bps_cfps(doc_df, period_type)
        records.append(
            {
                "date": date_str,
                "docID": doc_id,
                "submitDateTime": submit_datetime,
                "FundamentalAvailableDate": available_date,
                "date_dt": date_dt,
                "secCode": sec_code,
                "EPS": eps,
                "CFPS": cfps,
                "BPS": bps,
                "IssuedShares": issued_shares,
                "docTypeCode": doc_type_code,
                "period_type": period_type,
            }
        )

    if not records:
        return SecCodeProcessResult(
            sec_code=sec_code,
            doc_read_failures=doc_read_failures,
            wrote_output=False,
        )

    if doc_read_failures:
        return SecCodeProcessResult(
            sec_code=sec_code,
            doc_read_failures=doc_read_failures,
            wrote_output=False,
        )

    records.sort(
        key=lambda item: (
            item["FundamentalAvailableDate"],
            item["submitDateTime"] or "",
            item["docID"],
            item["date_dt"],
            item["date"],
        )
    )
    cfps_dates = [rec["date"] for rec in records if rec["CFPS"] is not None]
    edinet_history_start_date = min(cfps_dates) if cfps_dates else None

    rows = []
    for rec in records:
        rows.append(
            {
                "date": rec["date"],
                "docID": rec["docID"],
                "submitDateTime": rec["submitDateTime"],
                "FundamentalAvailableDate": rec["FundamentalAvailableDate"],
                "secCode": rec["secCode"],
                "EPS": rec["EPS"],
                "CFPS": rec["CFPS"],
                "BPS": rec["BPS"],  # BPSは時点値のためTTM化しない
                "IssuedShares": rec["IssuedShares"],
                "EdinetHistoryStartDate": edinet_history_start_date,
                "docTypeCode": rec["docTypeCode"],
                "periodType": rec["period_type"],
            }
        )

    out_df = pl.DataFrame(rows)

    date_parsed = pl.col("date").cast(pl.Utf8).str.strptime(pl.Date, strict=False)
    out_df = out_df.with_columns(
        pl.when(date_parsed.is_not_null())
        .then(date_parsed.dt.strftime("%Y-%m-%d"))
        .otherwise(pl.col("date").cast(pl.Utf8))
        .alias("date")
    )

    out_path = out_root / f"{sec_code}.csv"
    if protect_existing_history:
        validate_existing_history_is_preserved(out_df, out_path)
    write_csv_utf8sig(out_df, out_path)
    return SecCodeProcessResult(
        sec_code=sec_code,
        doc_read_failures=[],
        wrote_output=True,
    )


def build_full_rebuild_staging_root() -> Path:
    root = Path("tmp") / f"index_rebuild_{os.getpid()}_{datetime.now(UTC).strftime('%Y%m%d%H%M%S%f')}"
    root.mkdir(parents=True, exist_ok=False)
    return root


def replace_index_from_staging(staging_root: Path, out_root: Path) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    for existing_csv in out_root.glob("*.csv"):
        existing_csv.unlink()
    for staged_csv in staging_root.glob("*.csv"):
        shutil.move(str(staged_csv), out_root / staged_csv.name)


def run_index_rebuild(
    listup_path: Path = LISTUP_CSV,
    updated_listup_path: Path = UPDATED_LISTUP_CSV,
    doc_root: Path = DOC_ROOT,
    out_root: Path = OUT_ROOT,
    max_workers: int | None = None,
) -> None:
    listup, updated_listup = load_merged_listup(
        listup_path,
        updated_listup_path,
    )
    listup = prepare_listup(listup)
    target_sec_codes = select_target_sec_codes(listup, updated_listup)
    incremental = updated_listup is not None
    working_out_root = out_root
    staging_root: Path | None = None

    if incremental:
        print(
            f"updated_listup.csv の対象 {len(target_sec_codes)} 銘柄を"
            "完全履歴から再生成します。"
        )
    else:
        print(f"全 {len(target_sec_codes)} 銘柄を再生成します。")
        staging_root = build_full_rebuild_staging_root()
        working_out_root = staging_root

    # docディレクトリを走査してファイル名->パスのindexを作る（高速化）
    doc_index = build_doc_index(doc_root)

    # 出力先作成
    working_out_root.mkdir(parents=True, exist_ok=True)

    # secCodeごとに処理
    target_listup = listup.filter(pl.col("secCode").is_in(target_sec_codes))
    listup_by_sec_code = {
        str(sec_code): group
        for (sec_code,), group in target_listup.group_by(
            "secCode", maintain_order=True
        )
        if sec_code is not None
    }
    doc_read_failures: list[IndexDocReadFailure] = []

    def handle_result(sec_code: str, result: SecCodeProcessResult) -> None:
        if result.doc_read_failures:
            doc_read_failures.extend(result.doc_read_failures)
            message = (
                f"secCode={sec_code} で doc CSV 読み込み失敗が "
                f"{len(result.doc_read_failures)} 件ありました"
            )
            if incremental:
                tqdm.write(f"[WARN] {message}")
                record_failed_index_docs(result.doc_read_failures)
                record_failed_index_sec_code(sec_code, "doc_read", message)
            return

        if incremental:
            clear_failed_index_docs_for_sec_code(sec_code)
            clear_failed_index_sec_code(sec_code)

    def handle_exception(sec_code: str, exc: BaseException) -> None:
        if incremental:
            message = f"secCode={sec_code} の処理でエラーが発生しました: {exc}"
            tqdm.write(f"[WARN] {message}")
            record_failed_index_sec_code(sec_code, "sec_code", message)
            return
        raise RuntimeError(
            f"secCode={sec_code} の処理でエラーが発生しました"
        ) from exc

    try:
        if max_workers == 0:
            for sec_code in tqdm(target_sec_codes, total=len(target_sec_codes), desc="secCode"):
                try:
                    result = _process_sec_code(
                        sec_code,
                        listup_by_sec_code[sec_code],
                        doc_index,
                        working_out_root,
                        incremental,
                        not incremental,
                    )
                except Exception as exc:
                    handle_exception(sec_code, exc)
                    continue
                handle_result(sec_code, result)
        else:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                future_to_code = {
                    executor.submit(
                        _process_sec_code,
                        sec_code,
                        listup_by_sec_code[sec_code],
                        doc_index,
                        working_out_root,
                        incremental,
                        not incremental,
                    ): sec_code
                    for sec_code in target_sec_codes
                }
                for future in tqdm(
                    as_completed(future_to_code), total=len(future_to_code), desc="secCode"
                ):
                    sec_code = future_to_code[future]
                    try:
                        result = future.result()
                    except Exception as exc:
                        handle_exception(sec_code, exc)
                        continue
                    handle_result(sec_code, result)

        if doc_read_failures and not incremental:
            record_failed_index_docs(doc_read_failures)
            for failed_sec_code in sorted(
                {failure.sec_code for failure in doc_read_failures}
            ):
                record_failed_index_sec_code(
                    failed_sec_code,
                    "doc_read",
                    "doc CSV の読み込み失敗があるため、既存 index CSV は上書きしません",
                )
            samples = ", ".join(
                f"{failure.doc_id} ({failure.path})"
                for failure in doc_read_failures[:5]
            )
            raise RuntimeError(
                "doc CSV の読み込み失敗があるため、既存 index CSV は上書きしません: "
                f"{samples}"
            )

        if not incremental and staging_root is not None:
            replace_index_from_staging(staging_root, out_root)
            _write_state_rows(FAILED_INDEX_DOCS_PATH, FAILED_INDEX_DOC_COLUMNS, [])
            _write_state_rows(
                FAILED_INDEX_SEC_CODES_PATH,
                FAILED_INDEX_SEC_CODE_COLUMNS,
                [],
            )
    finally:
        if staging_root is not None:
            shutil.rmtree(staging_root, ignore_errors=True)

    print("done.")


# =========================
# メイン処理
# =========================
def main() -> None:
    run_index_rebuild()


def test_S10078N0():
    doc_path = r"D:\source\repos\edinet\doc\jpcrp030000\S100W5M9.csv"
    doc_df = read_doc_csv(doc_path)
    eps, cfps, bps = calc_eps_bps_cfps(doc_df, "annual")
    print(eps, cfps, bps)


if __name__ == "__main__":
    main()
    # test_S10078N0()
