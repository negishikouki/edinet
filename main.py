from fetch import main as fetch_main
from listup import main as listup_main
from recalc import refresh_recalc


def main():
    listup_main()
    fetch_main()
    refresh_recalc()


if __name__ == "__main__":
    main()
