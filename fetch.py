import requests
import pandas as pd
import zipfile
import io
from pathlib import Path
from enum import IntEnum
from tqdm import tqdm

# -----------------------------
# EDINET 定義
# -----------------------------
class EdinetDocType(IntEnum):
    CSV = 5

API_KEY = ""
BASE_URL = "https://api.edinet-fsa.go.jp/api/v2/documents"

doc_prefix_by_frequency = {
    "annual": {"prefix": ["jpcrp030000"], "role": "通期（12か月）"},
    "quarterly": {"prefix": ["jpcrp040300"], "role": "3か月（または累計）"},
    "semiannual": {"prefix": ["jpcrp050000", "jpcrp050200"], "role": "6か月（累計）"},
    "common": {"prefix": ["jpdei"], "role": "期間・連結区分などのキー"},
}

all_prefixes = sorted({p for v in doc_prefix_by_frequency.values() for p in v["prefix"]})


# -----------------------------
# CSV を取得する関数（prefix可変）
# -----------------------------
def fetch_all_prefix_dfs(docID: str, prefixes: list[str]) -> dict[str, pd.DataFrame]:
    url = f"{BASE_URL}/{docID}"
    params = {"type": EdinetDocType.CSV, "Subscription-Key": API_KEY}

    res = requests.get(url, params=params)
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


def docid_already_fetched(out_root: Path, docID: str) -> bool:
    """
    doc/配下のどこかに {docID}.csv が存在すれば True
    """
    return any(out_root.glob(f"*/{docID}.csv"))


# -----------------------------
# main
# -----------------------------
def main():
    listup_path = Path("listup.csv")
    out_root = Path("./doc")
    out_root.mkdir(parents=True, exist_ok=True)

    listup_df = pd.read_csv(listup_path)
    docIDs = listup_df["docID"].dropna().astype(str).tolist()

    for docID in tqdm(docIDs, desc="Fetching per docID"):

        # 既存チェック
        if docid_already_fetched(out_root, docID):
            tqdm.write(f"[SKIP] {docID}: already exists")
            continue

        try:
            dfs_by_prefix = fetch_all_prefix_dfs(docID, all_prefixes)

            for prefix, df in dfs_by_prefix.items():
                out_dir = out_root / prefix
                out_dir.mkdir(parents=True, exist_ok=True)
                df.to_csv(out_dir / f"{docID}.csv", index=False)

        except Exception as e:
            tqdm.write(f"[WARN] {docID}: {e}")



if __name__ == "__main__":
    main()
