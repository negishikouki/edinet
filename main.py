from pathlib import Path

from fetch import main as fetch_main
from listup import main as listup_main
from recalc import refresh_recalc

UPDATED_LISTUP_PATH = Path("updated_listup.csv")


def main() -> None:
    listup_changed = listup_main()
    documents_changed = fetch_main()

    # Keep an interrupted or otherwise pending listup update from being
    # skipped even when all of its documents are already present locally.
    has_pending_listup = UPDATED_LISTUP_PATH.exists()
    if not (listup_changed or documents_changed or has_pending_listup):
        print("変更がないため再計算をスキップします。")
        return

    refresh_recalc()


if __name__ == "__main__":
    main()
