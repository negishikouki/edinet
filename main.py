from listup import main as listup_main
from fetch import main as fetch_main
from recalc import refresh_recalc


def main():
    is_updated = listup_main()
    if is_updated:
        fetch_main()
        refresh_recalc()


if __name__ == "__main__":
    main()
