from listup import main as listup_main
from fetch import main as fetch_main
from index import main as index_main
from summary import main as summary_main


def main():
    is_updated = listup_main()
    if is_updated:
        fetch_main()
        index_main()
        summary_main()


if __name__ == "__main__":
    main()
