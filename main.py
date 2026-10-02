import argparse
import csv
import sys
from pathlib import Path

from fetch import main as fetch_main
from listup import main as listup_main
from recalc import refresh_recalc

UPDATED_LISTUP_PATH = Path("updated_listup.csv")


def write_repair_status_report(
    path: Path,
    statuses: list[dict[str, str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    with temporary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["secCode", "status"])
        writer.writeheader()
        writer.writerows(statuses)
    temporary_path.replace(path)


def main(argv: list[str] | None = None) -> list[dict[str, str]]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repair-sec-codes", nargs="*", default=[])
    parser.add_argument("--repair-sec-codes-file", type=Path)
    parser.add_argument("--repair-status-report", type=Path)
    args = parser.parse_args(argv if argv is not None else [])
    repair_sec_codes = [str(code).strip() for code in args.repair_sec_codes]
    if args.repair_sec_codes_file is not None:
        repair_sec_codes.extend(
            line.strip()
            for line in args.repair_sec_codes_file.read_text(encoding="utf-8").splitlines()
        )
    repair_sec_codes = sorted({code for code in repair_sec_codes if code})

    listup_changed = listup_main()
    documents_changed = fetch_main()

    # Keep an interrupted or otherwise pending listup update from being
    # skipped even when all of its documents are already present locally.
    has_pending_listup = UPDATED_LISTUP_PATH.exists()
    if repair_sec_codes:
        statuses = refresh_recalc(repair_sec_codes)
    elif listup_changed or documents_changed or has_pending_listup:
        statuses = refresh_recalc()
    else:
        print("変更がないため再計算をスキップします。")
        statuses = []

    if args.repair_status_report is not None:
        write_repair_status_report(args.repair_status_report, statuses)
    return statuses


if __name__ == "__main__":
    main(sys.argv[1:])
