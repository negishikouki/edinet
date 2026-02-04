from __future__ import annotations

import io
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple
from collections.abc import Iterable

import polars as pl
from tqdm import tqdm

# =========================
# 設定
# =========================
LISTUP_CSV = Path("./listup.csv")
UPDATED_LISTUP_CSV = Path("./updated_listup.csv")
DOC_ROOT = Path("./doc")      # docID.csv を再帰探索するルート
OUT_ROOT = Path("./index")    # secCode別の出力先

# EDINET（jppfs_cor）最小セット
# EPS
EID_PROFIT_TO_OWNERS = "jppfs_cor:ProfitLossAttributableToOwnersOfParent"  # 当期純利益（親会社株主帰属）
EID_AVG_SHARES = "jppfs_cor:AverageNumberOfShares"                         # 期中平均株式数
EID_B_ELPS = "jpcrp_cor:BasicEarningsLossPerShareSummaryOfBusinessResults" # １株当たり当期純利益又は当期純損失（△）、経営指標等
EID_D_ELPS = "jpcrp_cor:DilutedEarningsPerShareSummaryOfBusinessResults"   # 潜在株式調整後１株当たり当期純利益、経営指標等

# BPS
EID_NET_ASSETS_PER_SHARE = "jpcrp_cor:NetAssetsPerShareSummaryOfBusinessResults" # １株当たり純資産額、経営指標等
EID_NET_ASSETS = "jppfs_cor:NetAssets"                                     # 純資産合計
EID_SHARES_OUT = "jppfs_cor:NumberOfSharesOutstanding"                     # 期末発行済株式数

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
CID_Prior1YTDDuration  = "Prior1YTDDuration"   # 前年度同四半期累計期間
CID_Prior1YearDuration = "Prior1YearDuration"  # 前期
CID_CurrentYTDDuration_NonConsolidated = "CurrentYTDDuration_NonConsolidated" # 当四半期累計期間
CID_Prior1YTDDuration_NonConsolidated  = "Prior1YTDDuration_NonConsolidated"   # 前年度同四半期累計期間
CID_Prior1YearDuration_NonConsolidated = "Prior1YearDuration_NonConsolidated"  # 前期


# -----------------------------
# Instant（時点）
# -----------------------------
CID_CurrentYearInstant = "CurrentYearInstant"   # 当期末
CID_Prior4YearInstant  = "Prior4YearInstant"    # 四期前時点
CID_Prior3YearInstant  = "Prior3YearInstant"    # 三期前時点
CID_Prior2YearInstant  = "Prior2YearInstant"    # 前々期末
CID_Prior1YearInstant  = "Prior1YearInstant"    # 前期末
CID_InterimInstant = "InterimInstant" # 当中間期末
CID_Prior4InterimInstant = "Prior4InterimInstant" # 四期中間期末
CID_Prior3InterimInstant = "Prior3InterimInstant" # 三期中間期末
CID_Prior2InterimInstant = "Prior2InterimInstant" # 前々中間期末
CID_Prior1InterimInstant = "Prior1InterimInstant" # 前中間期末


# -----------------------------
# Instant（時点・非連結）
# -----------------------------
CID_CurrentYearInstant_NonConsolidatedMember = "CurrentYearInstant_NonConsolidatedMember"   # 当期末
CID_Prior4YearInstant_NonConsolidatedMember  = "Prior4YearInstant_NonConsolidatedMember"    # 四期前時点
CID_Prior3YearInstant_NonConsolidatedMember  = "Prior3YearInstant_NonConsolidatedMember"    # 三期前時点
CID_Prior2YearInstant_NonConsolidatedMember  = "Prior2YearInstant_NonConsolidatedMember"    # 前々期末
CID_Prior1YearInstant_NonConsolidatedMember  = "Prior1YearInstant_NonConsolidatedMember"    # 前期末
CID_InterimInstant_NonConsolidatedMember = "InterimInstant_NonConsolidatedMember" # 当中間期末
CID_Prior4InterimInstant_NonConsolidatedMember = "Prior4InterimInstant_NonConsolidatedMember" # 四期前期間末末
CID_Prior3InterimInstant_NonConsolidatedMember = "Prior3InterimInstant_NonConsolidatedMember" # 三期前期間末
CID_Prior2InterimInstant_NonConsolidatedMember = "Prior2InterimInstant_NonConsolidatedMember" # 前々中間期末
CID_Prior1InterimInstant_NonConsolidatedMember = "Prior1InterimInstant_NonConsolidatedMember" # 前中間期末


# -----------------------------
# Duration（期間・連結）
# -----------------------------
CID_Prior4YearDuration = "Prior4YearDuration"  # 四期前期間
CID_Prior3YearDuration = "Prior3YearDuration"  # 三期前期間
CID_Prior2YearDuration = "Prior2YearDuration"  # 前々期期間
CID_Prior1YearDuration = "Prior1YearDuration"  # 前期期間
CID_CurrentYearDuration = "CurrentYearDuration"  # 当期期間
CID_InterimDuration = "InterimDuration"  # 当中間期末
CID_Prior4InterimDuration = "Prior4InterimDuration" # 四期中間期
CID_Prior3InterimDuration = "Prior3InterimDuration" # 三期中間期
CID_Prior2InterimDuration = "Prior2InterimDuration" # 前々中間期
CID_Prior1InterimDuration = "Prior1InterimDuration"	# 前中間期


# -----------------------------
# Duration（期間・非連結）
# -----------------------------
CID_Prior4YearDuration_NonConsolidated = "Prior4YearDuration_NonConsolidatedMember"  # 四期前期間（非連結）
CID_Prior3YearDuration_NonConsolidated = "Prior3YearDuration_NonConsolidatedMember"  # 三期前期間（非連結）
CID_Prior2YearDuration_NonConsolidated = "Prior2YearDuration_NonConsolidatedMember"  # 前々期期間（非連結）
CID_Prior1YearDuration_NonConsolidated = "Prior1YearDuration_NonConsolidatedMember"  # 前期期間（非連結）
CID_CurrentYearDuration_NonConsolidated = "CurrentYearDuration_NonConsolidatedMember"  # 当期期間（非連結）
CID_InterimDuration_NonConsolidatedMember = "InterimDuration_NonConsolidatedMember"  # 当中間期末
CID_Prior4InterimDuration_NonConsolidatedMember = "Prior4InterimDuration_NonConsolidatedMember" # 四期中間期
CID_Prior3InterimDuration_NonConsolidatedMember = "Prior3InterimDuration_NonConsolidatedMember" # 三期中間期
CID_Prior2InterimDuration_NonConsolidatedMember = "Prior2InterimDuration_NonConsolidatedMember" # 前々中間期
CID_Prior1InterimDuration_NonConsolidatedMember = "Prior1InterimDuration_NonConsolidatedMember"	# 前中間期


# コンテキストID（時点・個別）
CID_Prior4YearInstant_NonConsolidated = "Prior4YearInstant_NonConsolidatedMember"  # 四期前時点（個別）
CID_Prior3YearInstant_NonConsolidated = "Prior3YearInstant_NonConsolidatedMember"  # 三期前時点（個別）
CID_Prior2YearInstant_NonConsolidated = "Prior2YearInstant_NonConsolidatedMember"  # 前々期末（個別）
CID_Prior1YearInstant_NonConsolidated = "Prior1YearInstant_NonConsolidatedMember"  # 前期末（個別）
CID_CurrentYearInstant_NonConsolidated = "CurrentYearInstant_NonConsolidatedMember"  # 当期末（個別）


# =========================
# 期間種別（docTypeCode）
# =========================
DOC_TYPE_PERIOD = {
    "120": "annual",
    "130": "annual",
    "160": "semiannual",
    "170": "semiannual",
    "140": "quarterly",
    "150": "quarterly",
}


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


def write_csv_utf8sig(df: pl.DataFrame, path: Path) -> None:
    try:
        df.write_csv(path, include_bom=True)
        return
    except TypeError:
        pass

    csv_bytes = None
    try:
        csv_text = df.write_csv()
        if isinstance(csv_text, bytes):
            csv_bytes = csv_text
        elif isinstance(csv_text, str):
            csv_bytes = csv_text.encode("utf-8")
    except TypeError:
        pass

    if csv_bytes is None:
        try:
            buf = io.StringIO()
            df.write_csv(buf)
            csv_bytes = buf.getvalue().encode("utf-8")
        except Exception:
            buf = io.BytesIO()
            df.write_csv(buf)
            csv_bytes = buf.getvalue()

    with open(path, "wb") as f:
        f.write(b"\xef\xbb\xbf")
        f.write(csv_bytes)


def _to_float(x) -> Optional[float]:
    """
    値をfloatに変換。空・NaN・変換不能はNone。
    """
    if x is None:
        return None
    try:
        if isinstance(x, float) and x != x:
            return None
        # 文字列のカンマ等を除去
        if isinstance(x, str):
            x = x.strip().replace(",", "")
            if x == "":
                return None
        return float(x)
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
            con = sub_df.filter(
                pl.col("連結・個別").cast(pl.Utf8).str.contains("連結")
            )
            if not con.is_empty():
                sub_df = con

        if prefer_current_year and "相対年度" in sub_df.columns:
            cur = sub_df.filter(
                pl.col("相対年度").cast(pl.Utf8).is_in(["0", "当期"])
            )
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
                (pl.col(eid_col).cast(pl.Utf8) == element_id)
                & (cid_col_expr == cid)
            )
            value = pick_single_value(sub_df)
            if value is None:
                return None
            values.append(value)
        return sum(values) if values else None

    if aggregate == "sum" and context_id:
        if isinstance(context_id, list) and context_id and isinstance(context_id[0], list):
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
        if isinstance(context_id, Iterable) and not isinstance(context_id, (str, bytes)):
            for cid in context_id:
                sub = df.filter(
                    (pl.col(eid_col).cast(pl.Utf8) == element_id)
                    & (cid_col_expr == cid)
                )
                if not sub.is_empty():
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


def calc_eps_bps_cfps(
    doc_df: pl.DataFrame, period_type: str
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    doc CSV DataFrame から EPS, CFPS, BPS を計算して返す。
    欠損があれば None。
    """

    # -----------------------------
    # 発行済株式数
    # -----------------------------
    shares_context = {
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

    ttm_duration_contexts = {
        "annual": [
            [CID_CurrentYearDuration],
            [CID_CurrentYearDuration_NonConsolidated],
        ],
        "semiannual": [
            [CID_InterimDuration, CID_Prior1InterimDuration],
            [CID_InterimDuration_NonConsolidatedMember, CID_Prior1InterimDuration_NonConsolidatedMember],
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
        eps = (profit / issued_shares) if (profit is not None and issued_shares not in (None, 0.0)) else None

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
        bps = (net_assets / issued_shares) if (net_assets is not None and issued_shares not in (None, 0.0)) else None

    
    # -----------------------------
    # CFPS（FCFEベース：厳密版）
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
        capex = _sum_optional([
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
        ])

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

    net_borrowing = _sum_optional([
        st_delta,
        lt_proceeds,
        lt_repayment,  # 既にマイナスで入っている前提
    ])

    fcfe = (
        ocf - capex + net_borrowing
        if (ocf is not None and capex is not None and net_borrowing is not None)
        else None
    )

    cfps = (fcfe / issued_shares) if (fcfe is not None and issued_shares not in (None, 0.0)) else None

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
        cfps = (fcff / issued_shares) if (fcff is not None and issued_shares not in (None, 0.0)) else None

    return eps, cfps, bps



# =========================
# メイン処理
# =========================
def main() -> None:
    # listup.csv 読み込み
    listup_path = UPDATED_LISTUP_CSV if UPDATED_LISTUP_CSV.exists() else LISTUP_CSV
    # secCode can contain alphanumeric values (e.g. 138A0), so read it as string
    listup = pl.read_csv(
        listup_path,
        schema_overrides={"secCode": pl.Utf8},
    )

    required_cols = {"secCode", "date", "docID", "edinetCode", "docTypeCode"}
    missing = required_cols - set(listup.columns)
    if missing:
        raise ValueError(f"listup.csv に必要な列がありません: {sorted(missing)}")

    listup = listup.with_columns(
        pl.col("date")
        .cast(pl.Utf8)
        .str.strptime(pl.Datetime, strict=False)
        .alias("date_dt")
    )

    # edinetCodeで最新のsecCodeを取得し、全件に反映
    non_null_sec = (
        listup.filter(pl.col("secCode").is_not_null())
        .sort(["date_dt", "date"], nulls_last=True)
    )
    latest_sec_code_df = non_null_sec.group_by("edinetCode", maintain_order=True).agg(
        pl.col("secCode").last().alias("latest_secCode")
    )
    listup = (
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
    )

    # secCodeでGroupBy、dateでOrderBy
    # dateは文字列でも良いが、並びが崩れそうならdatetime化
    # ここでは datetime化してから出力時は YYYY-MM-DD に戻す
    # dateが壊れている行は最後に回す
    listup = listup.sort(["secCode", "date_dt", "date"], nulls_last=True)

    # docディレクトリを走査してファイル名->パスのindexを作る（高速化）
    doc_index = build_doc_index(DOC_ROOT)

    # 出力先作成
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    # secCodeごとに処理
    sec_codes = (
        listup.get_column("secCode")
        .drop_nulls()
        .unique(maintain_order=True)
        .to_list()
    )
    for sec_code, g in tqdm(
            ((code, listup.filter(pl.col("secCode") == code)) for code in sec_codes),
            total=len(sec_codes),
            desc="secCode"
        ):

        records = []
        for r in g.iter_rows(named=True):
            date_str = str(r["date"])
            date_dt = r["date_dt"]
            if date_dt is None:
                date_dt = datetime.min
            doc_id = str(r["docID"])
            doc_type_code = str(r["docTypeCode"])
            fname = f"{doc_id}.csv"

            period_type = DOC_TYPE_PERIOD.get(doc_type_code)
            if period_type is None:
                continue

            doc_path = doc_index.get(fname)
            if doc_path is None:
                # 見つからなければ continue
                continue

            try:
                doc_df = pl.read_csv(doc_path)
            except Exception:
                # 読めない場合もスキップ
                continue

            eps, cfps, bps = calc_eps_bps_cfps(doc_df, period_type)

            records.append(
                {
                    "date": date_str,
                    "date_dt": date_dt,
                    "secCode": sec_code,
                    "EPS": eps,
                    "CFPS": cfps,
                    "BPS": bps,
                    "docTypeCode": doc_type_code,
                    "period_type": period_type,
                }
            )

        if not records:
            # 1件も作れない場合は出力しない（必要なら空CSVを出すよう変更可）
            continue

        records.sort(key=lambda item: (item["date_dt"], item["date"]))

        rows = []
        for rec in records:
            rows.append(
                {
                    "date": rec["date"],
                    "secCode": rec["secCode"],
                    "EPS": rec["EPS"],
                    "CFPS": rec["CFPS"],
                    "BPS": rec["BPS"],  # BPSは時点値のためTTM化しない
                    "docTypeCode": rec["docTypeCode"],
                    "periodType": rec["period_type"],
                }
            )

        out_df = pl.DataFrame(rows)

        # dateを可能なら整形（YYYY-MM-DD）
        date_parsed = pl.col("date").cast(pl.Utf8).str.strptime(pl.Date, strict=False)
        out_df = out_df.with_columns(
            pl.when(date_parsed.is_not_null())
            .then(date_parsed.dt.strftime("%Y-%m-%d"))
            .otherwise(pl.col("date").cast(pl.Utf8))
            .alias("date")
        )

        # secCodeごとにCSV出力
        out_path = OUT_ROOT / f"{sec_code}.csv"
        write_csv_utf8sig(out_df, out_path)

    print("done.")


def test_S10078N0():
    doc_path = r"D:\source\repos\edinet\doc\jpcrp030000\S100VUG5.csv"
    doc_df = pl.read_csv(doc_path)
    eps, cfps, bps = calc_eps_bps_cfps(doc_df, "annual")
    print(eps, cfps, bps)

if __name__ == "__main__":
    main()
    #test_S10078N0()
