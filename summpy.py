import pandas as pd
from pathlib import Path
from tqdm import tqdm

INDEX_DIR = Path("index")
SUMMARY_DIR = Path("summary")
SUMMARY_DIR.mkdir(exist_ok=True)
LISTUP_PATH = Path("listup.csv")
SUMMARY_PATH = SUMMARY_DIR / "summary.csv"
TARGET_COLS = ["EPS", "CFPS", "BPS"]


def load_filter_names(listup_path: Path) -> dict:
    df = pd.read_csv(listup_path)
    name_column = "filterName" if "filterName" in df.columns else "filerName"
    return df.set_index("secCode")[name_column].to_dict()


def pick_latest_values(df: pd.DataFrame) -> tuple[dict, dict]:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df_sorted = df.sort_values("date")

    values: dict[str, float | pd._libs.missing.NAType] = {}
    dates: dict[str, str | pd._libs.missing.NAType] = {}

    for col in TARGET_COLS:
        non_null = df_sorted.dropna(subset=["date", col])
        if non_null.empty:
            values[col] = pd.NA
            dates[col] = pd.NA
        else:
            latest_row = non_null.iloc[-1]
            values[col] = latest_row[col]
            dates[col] = latest_row["date"].date().isoformat()

    return values, dates


def main():
    filter_names = load_filter_names(LISTUP_PATH) if LISTUP_PATH.exists() else {}
    records = []

    for csv_path in tqdm(sorted(INDEX_DIR.glob("*.csv")), desc="Summarizing index", unit="file"):
        sec_code = csv_path.stem
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
    summary_df.to_csv(SUMMARY_PATH, index=False)


if __name__ == "__main__":
    main()
