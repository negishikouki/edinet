from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple
from collections.abc import Iterable

import pandas as pd
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

# 営業CF
EID_OCF = "jppfs_cor:NetCashProvidedByUsedInOperatingActivities"           # 営業CF
EID_NCPBUIOA = "jpcrp_cor:NetCashProvidedByUsedInOperatingActivitiesSummaryOfBusinessResults" # 営業活動によるキャッシュ・フロー、経営指標等

# 投資CF
EID_ICF = "jppfs_cor:NetCashProvidedByUsedInInvestingActivities"           # 投資CF
EID_NCPBUIIA = "jpcrp_cor:NetCashProvidedByUsedInInvestingActivitiesSummaryOfBusinessResults" # 投資活動によるキャッシュ・フロー、経営指標等

# 財務CF（FCFE向け）
EID_NET_INCREASE_DECREASE_ST_LOANS = "jppfs_cor:NetIncreaseDecreaseInShortTermLoansPayableFinCF"  # 短期借入金の純増減額
EID_PROCEEDS_LT_LOANS = "jppfs_cor:ProceedsFromLongTermLoansPayableFinCF"  # 長期借入れによる収入
EID_REPAYMENT_LT_LOANS = "jppfs_cor:RepaymentOfLongTermLoansPayableFinCF"  # 長期借入金の返済による支出

# 投資（CAPEX）
EID_PPE_PURCHASE = "jppfs_cor:PurchaseOfPropertyPlantAndEquipmentInvCF"  # 有形固定資産の取得
EID_INTANGIBLE_PURCHASE = "jppfs_cor:PurchaseOfIntangibleAssetsInvCF"    # 無形固定資産の取得

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

# -----------------------------
# Duration（期間・非連結）
# -----------------------------
CID_Prior4YearDuration = "Prior4YearDuration"  # 四期前期間
CID_Prior3YearDuration = "Prior3YearDuration"  # 三期前期間
CID_Prior2YearDuration = "Prior2YearDuration"  # 前々期期間
CID_Prior1YearDuration = "Prior1YearDuration"  # 前期期間
CID_CurrentYearDuration = "CurrentYearDuration"  # 当期期間
# -----------------------------
# Duration（期間・非連結）
# -----------------------------
CID_Prior4YearDuration_NonConsolidated = "Prior4YearDuration_NonConsolidatedMember"  # 四期前期間（非連結）
CID_Prior3YearDuration_NonConsolidated = "Prior3YearDuration_NonConsolidatedMember"  # 三期前期間（非連結）
CID_Prior2YearDuration_NonConsolidated = "Prior2YearDuration_NonConsolidatedMember"  # 前々期期間（非連結）
CID_Prior1YearDuration_NonConsolidated = "Prior1YearDuration_NonConsolidatedMember"  # 前期期間（非連結）
CID_CurrentYearDuration_NonConsolidated = "CurrentYearDuration_NonConsolidatedMember"  # 当期期間（非連結）


# コンテキストID（時点・個別）
CID_Prior4YearInstant_NonConsolidated = "Prior4YearInstant_NonConsolidatedMember"  # 四期前時点（個別）
CID_Prior3YearInstant_NonConsolidated = "Prior3YearInstant_NonConsolidatedMember"  # 三期前時点（個別）
CID_Prior2YearInstant_NonConsolidated = "Prior2YearInstant_NonConsolidatedMember"  # 前々期末（個別）
CID_Prior1YearInstant_NonConsolidated = "Prior1YearInstant_NonConsolidatedMember"  # 前期末（個別）
CID_CurrentYearInstant_NonConsolidated = "CurrentYearInstant_NonConsolidatedMember"  # 当期末（個別）


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


def _to_float(x) -> Optional[float]:
    """
    値をfloatに変換。空・NaN・変換不能はNone。
    """
    if x is None:
        return None
    try:
        if pd.isna(x):
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
    df: pd.DataFrame,
    element_id: str,
    context_id: str | list = None,
    prefer_consolidated: bool = True,
    prefer_current_year: bool = True,
) -> Optional[float]:
    """
    EDINET CSV（要素ID/値などの列がある）から、指定element_idの値を1つ選んで返す。
    - 連結・個別（列名に "連結・個別" がある想定）: 連結を優先
    - 相対年度（列名に "相対年度" がある想定）: 当期(0)を優先
    条件列が存在しない場合は存在する行の先頭を採用。
    """
    if df.empty:
        return None

    # 要素ID列
    eid_col = df.columns[0]
    
    if context_id:
        # コンテキストID列
        cid_col = df.columns[2]
        if isinstance(context_id, Iterable):
            for cid in context_id:
                sub = df[(df[eid_col].astype(str) == element_id) & (df[cid_col].astype(str) == cid)].copy()
                if not sub.empty:
                    break
        else:
            sub = df[(df[eid_col].astype(str) == element_id) & (df[cid_col].astype(str) == context_id)].copy()
    else:
        sub = df[df[eid_col].astype(str) == element_id].copy()
    if sub.empty:
        return None

    # フィルタ優先
    if prefer_consolidated and "連結・個別" in sub.columns:
        # よくある表記例: "連結" / "個別" / "連結財務諸表" 等
        # "連結" を含むものを優先
        con = sub[sub["連結・個別"].astype(str).str.contains("連結", na=False)]
        if not con.empty:
            sub = con

    if prefer_current_year and "相対年度" in sub.columns:
        # 当期=0 を優先（もし文字列なら0相当を拾う）
        # 例: 0, "0", "当期" など揺れがある可能性に軽く対応
        s = sub["相対年度"].astype(str)
        cur = sub[s.isin(["0", "当期"])].copy()
        if not cur.empty:
            sub = cur

    # 値列推定：よくある「値」列を優先、なければ末尾列から探す
    value_col = None
    if "値" in sub.columns:
        value_col = "値"
    else:
        # 値っぽい列を後ろから探す
        for c in reversed(sub.columns):
            if c == eid_col:
                continue
            # 単位やテキストを避けたいが完全には無理なので、最後の列を優先
            value_col = c
            break

    if value_col is None:
        return None

    v = _to_float(sub.iloc[0][value_col])
    return v


def calc_eps_bps_cfps(doc_df: pd.DataFrame) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    doc CSV DataFrame から EPS, CFPS, BPS を計算して返す。
    欠損があれば None。
    """

    # -----------------------------
    # 発行済株式数
    # -----------------------------
    issued_shares = pick_value(
        doc_df,
        EID_TNOIS,
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

    if issued_shares is None:
        issued_shares = pick_value(
            doc_df,
            EID_SHARES_OUT,
            context_id=None,
            prefer_consolidated=True,
            prefer_current_year=True,
        )

    # -----------------------------
    # EPS
    # -----------------------------
    eps = pick_value(
        doc_df,
        EID_B_ELPS,
            context_id=[
                CID_CurrentYTDDuration,
                CID_CurrentYearDuration,
                CID_CurrentYearDuration_NonConsolidated,
                CID_CurrentYearInstant_NonConsolidated,
            ],
        prefer_consolidated=True,
        prefer_current_year=True,
    )

    if eps is None:
        eps = pick_value(
            doc_df,
            EID_D_ELPS,
            context_id=[
                CID_CurrentYTDDuration,
                CID_CurrentYearDuration,
                CID_CurrentYearDuration_NonConsolidated,
                CID_CurrentYearInstant_NonConsolidated,
            ],
            prefer_consolidated=True,
            prefer_current_year=True,
        )

    if eps is None:
        profit = pick_value(
            doc_df,
            EID_PROFIT_TO_OWNERS,
            context_id=None,
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
    ocf = pick_value(
        doc_df,
        EID_NCPBUIOA,
        context_id=[
            CID_CurrentYTDDuration,
            CID_CurrentYearDuration,
            CID_CurrentYearDuration_NonConsolidated,
        ],
        prefer_consolidated=True,
        prefer_current_year=True,
    )

    if ocf is None:
        ocf = pick_value(
            doc_df,
            EID_OCF,
            context_id=None,
            prefer_consolidated=True,
            prefer_current_year=True,
        )

    # CAPEX（有形＋無形）
    capex = _sum_optional([
        pick_value(
            doc_df,
            EID_PPE_PURCHASE,
            context_id=None,
            prefer_consolidated=True,
            prefer_current_year=True,
        ),
        pick_value(
            doc_df,
            EID_INTANGIBLE_PURCHASE,
            context_id=None,
            prefer_consolidated=True,
            prefer_current_year=True,
        ),
    ])

    # 純借入額
    st_delta = pick_value(
        doc_df,
        EID_NET_INCREASE_DECREASE_ST_LOANS,
        context_id=None,
        prefer_consolidated=True,
        prefer_current_year=True,
    )

    lt_proceeds = pick_value(
        doc_df,
        EID_PROCEEDS_LT_LOANS,
        context_id=None,
        prefer_consolidated=True,
        prefer_current_year=True,
    )

    lt_repayment = pick_value(
        doc_df,
        EID_REPAYMENT_LT_LOANS,
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
        icf = pick_value(
            doc_df,
            EID_NCPBUIIA,
            context_id=[
                CID_CurrentYTDDuration,
                CID_CurrentYearDuration,
                CID_CurrentYearDuration_NonConsolidated,
            ],
            prefer_consolidated=True,
            prefer_current_year=True,
        )

        if icf is None:
            icf = pick_value(
                doc_df,
                EID_ICF,
                context_id=None,
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
    listup = pd.read_csv(listup_path, low_memory=False)

    required_cols = {"secCode", "date", "docID"}
    missing = required_cols - set(listup.columns)
    if missing:
        raise ValueError(f"listup.csv に必要な列がありません: {sorted(missing)}")

    # secCodeでGroupBy、dateでOrderBy
    # dateは文字列でも良いが、並びが崩れそうならdatetime化
    # ここでは datetime化してから出力時は YYYY-MM-DD に戻す
    listup = listup.copy()
    listup["date_dt"] = pd.to_datetime(listup["date"], errors="coerce")
    # dateが壊れている行は最後に回す
    listup = listup.sort_values(["secCode", "date_dt", "date"], ascending=[True, True, True])

    # docディレクトリを走査してファイル名->パスのindexを作る（高速化）
    doc_index = build_doc_index(DOC_ROOT)

    # 出力先作成
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    # secCodeごとに処理
    for sec_code, g in tqdm(
            listup.groupby("secCode", sort=False),
            total=listup["secCode"].nunique(),
            desc="secCode"
        ):
        rows = []

        for _, r in g.iterrows():
            date_str = str(r["date"])
            doc_id = str(r["docID"])
            fname = f"{doc_id}.csv"

            doc_path = doc_index.get(fname)
            if doc_path is None:
                # 見つからなければ continue
                continue

            try:
                doc_df = pd.read_csv(doc_path)
            except Exception:
                # 読めない場合もスキップ
                continue

            eps, cfps, bps = calc_eps_bps_cfps(doc_df)

            rows.append(
                {
                    "date": date_str,
                    "secCode": sec_code,
                    "EPS": eps,
                    "CFPS": cfps,
                    "BPS": bps,
                }
            )

        if not rows:
            # 1件も作れない場合は出力しない（必要なら空CSVを出すよう変更可）
            continue

        out_df = pd.DataFrame(rows)

        # dateを可能なら整形（YYYY-MM-DD）
        dt = pd.to_datetime(out_df["date"], errors="coerce")
        out_df.loc[dt.notna(), "date"] = dt[dt.notna()].dt.strftime("%Y-%m-%d")

        # secCodeごとにCSV出力
        out_path = OUT_ROOT / f"{sec_code}.csv"
        out_df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print("done.")


def test_S10078N0():
    doc_path = "./doc/jpcrp030000/S10078N0.csv"
    doc_df = pd.read_csv(doc_path)
    eps, cfps, bps = calc_eps_bps_cfps(doc_df)
    print(eps, cfps, bps)

if __name__ == "__main__":
    main()
    #test_S10078N0()
