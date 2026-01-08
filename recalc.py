from index import main as index_main
from summary import main as summary_main

def refresh_recalc():
    index_main()
    summary_main()

if __name__ == "__main__":
    refresh_recalc()

