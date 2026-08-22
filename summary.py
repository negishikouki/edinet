import os
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from edinet_constants import LISTUP_COLUMNS
from normalization import normalize_sec_code_text

INDEX_DIR = Path("index")
SUMMARY_DIR = Path("summary")
SUMMARY_DIR.mkdir(exist_ok=True)
LISTUP_PATH = Path("listup.csv")
UPDATED_LISTUP_PATH = Path("updated_listup.csv")
SUMMARY_PATH = SUMMARY_DIR / "edinet_summary.csv"
TARGET_COLS = ["EPS", "CFPS", "BPS"]


def normalize_sec_code_column(df: pd.DataFrame) -> pd.DataFrame:
    if "secCode" not in df.columns:
        return df

    normalized = df.copy()
    normalized["secCode"] = normalized["secCode"].map(normalize_sec_code_text).astype("string")
    return normalized


def write_dataframe_atomic(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        df.to_csv(temp_path, index=False)
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def load_filter_names(listup_df: pd.DataFrame) -> dict:
    if listup_df.empty or "secCode" not in listup_df.columns:
        return {}

    name_column = "filterName" if "filterName" in listup_df.columns else "filerName"
    return listup_df.dropna(subset=["secCode"]).set_index("secCode")[name_column].to_dict()


def pick_latest_values(df: pd.DataFrame) -> tuple[dict[str, object], dict[str, object]]:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df_sorted = df.sort_values("date")

    values: dict[str, object] = {}
    dates: dict[str, object] = {}

    for col in TARGET_COLS:
        non_null = df_sorted.dropna(subset=["date", col])
        if non_null.empty:
            values[col] = None
            dates[col] = None
        else:
            latest_row = non_null.iloc[-1]
            values[col] = latest_row[col]
            dates[col] = latest_row["date"].date().isoformat()

    return values, dates


def merge_listups(listup_path: Path, updated_path: Path) -> pd.DataFrame:
    frames = []

    if listup_path.exists():
        base_df = normalize_sec_code_column(pd.read_csv(listup_path, low_memory=False))
        frames.append(base_df)

    if updated_path.exists():
        updated_df = normalize_sec_code_column(pd.read_csv(updated_path, low_memory=False))
        frames.append(updated_df)

    if not frames:
        return pd.DataFrame(columns=LISTUP_COLUMNS)

    merged = pd.concat(frames, ignore_index=True)
    if "docID" in merged.columns:
        merged = merged.drop_duplicates(subset=["docID"], keep="last")

    extra_columns = sorted(set(merged.columns).difference(LISTUP_COLUMNS))
    merged = merged.reindex(columns=[*LISTUP_COLUMNS, *extra_columns])

    return normalize_sec_code_column(merged)


def main() -> None:
    listup_df = merge_listups(LISTUP_PATH, UPDATED_LISTUP_PATH)
    filter_names = load_filter_names(listup_df)
    records = []

    index_files_by_code: dict[str, Path] = {}
    for csv_path in sorted(INDEX_DIR.glob("*.csv")):
        sec_code = normalize_sec_code_text(csv_path.stem)
        if sec_code is None:
            continue

        selected = index_files_by_code.get(sec_code)
        if selected is None:
            index_files_by_code[sec_code] = csv_path
            continue

        selected_is_canonical = selected.stem == sec_code
        current_is_canonical = csv_path.stem == sec_code
        if current_is_canonical and not selected_is_canonical:
            index_files_by_code[sec_code] = csv_path

    for sec_code, csv_path in tqdm(
        sorted(index_files_by_code.items()),
        desc="Summarizing index",
        unit="file",
    ):
        df = pd.read_csv(csv_path)

        if "date" not in df.columns:
            continue

        values, dates = pick_latest_values(df)

        record = {
            "secCode": sec_code,
            "filterName": filter_names.get(sec_code, pd.NA),
            "EPS": values["EPS"],
            "CFPS": values["CFPS"],
            "BPS": values["BPS"],
            "date_EPS": dates["EPS"],
            "date_CFPS": dates["CFPS"],
            "date_BPS": dates["BPS"],
        }
        records.append(record)

    summary_df = pd.DataFrame(
        records,
        columns=[
            "secCode",
            "filterName",
            "EPS",
            "CFPS",
            "BPS",
            "date_EPS",
            "date_CFPS",
            "date_BPS",
        ],
    )
    write_dataframe_atomic(summary_df, SUMMARY_PATH)

    if UPDATED_LISTUP_PATH.exists():
        write_dataframe_atomic(listup_df, LISTUP_PATH)
        UPDATED_LISTUP_PATH.unlink()


if __name__ == "__main__":
    main()
