from index import main as index_main
from summary import main as summary_main


def refresh_recalc(
    repair_sec_codes: list[str] | tuple[str, ...] = (),
) -> list[dict[str, str]]:
    if repair_sec_codes:
        statuses = index_main(forced_sec_codes=repair_sec_codes)
    else:
        statuses = index_main()
    summary_main()
    return statuses

if __name__ == "__main__":
    refresh_recalc()

