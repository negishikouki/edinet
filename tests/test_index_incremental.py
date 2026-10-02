import os
import unittest
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import polars as pl
import index as index_module

from index import (_process_sec_code, calc_issued_shares, load_merged_listup,
                   prepare_listup, select_target_sec_codes,
                   run_index_rebuild, validate_existing_history_is_preserved,
                   write_csv_utf8sig)

LISTUP_COLUMNS = [
    "date",
    "docID",
    "edinetCode",
    "secCode",
    "filerName",
    "docTypeCode",
    "submitDateTime",
]


def write_listup(path: Path, rows: list[dict[str, object]]) -> None:
    pl.DataFrame(rows).select(LISTUP_COLUMNS).write_csv(path)


def write_minimal_doc_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        [
            {
                "要素ID": "dummy",
                "項目名": "dummy",
                "コンテキストID": "dummy",
                "相対年度": "0",
                "連結・個別": "連結",
                "期間・時点": "時点",
                "ユニットID": "",
                "単位": "",
                "値": "",
            }
        ]
    ).write_csv(path)


@contextmanager
def temporary_cwd(path: Path):
    original = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(original)


class IncrementalIndexTests(unittest.TestCase):
    def test_repair_status_distinguishes_available_failed_and_no_history(self) -> None:
        status_builder = getattr(index_module, "build_sec_code_repair_statuses", None)
        self.assertTrue(callable(status_builder), "repair status builder must be implemented")
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "10000.csv").write_text("date,secCode\n2024-01-01,10000\n", encoding="utf-8")
            statuses = status_builder(
                ["10000", "20000", "30000"],
                {"10000", "20000"},
                root,
                {"20000"},
            )

        self.assertEqual(
            statuses,
            [
                {"secCode": "10000", "status": "index_available"},
                {"secCode": "20000", "status": "index_rebuild_failed"},
                {"secCode": "30000", "status": "no_edinet_history"},
            ],
        )


    def test_forced_missing_codes_are_added_to_incremental_targets(self) -> None:
        merged = pl.DataFrame(
            [
                {"date": "2024-01-01", "docID": "D1", "edinetCode": "E1", "secCode": "10000", "filerName": "A", "docTypeCode": "120", "submitDateTime": "2024-01-01 09:00:00"},
                {"date": "2024-01-01", "docID": "D2", "edinetCode": "E2", "secCode": "20000", "filerName": "B", "docTypeCode": "120", "submitDateTime": "2024-01-01 09:00:00"},
            ]
        ).select(LISTUP_COLUMNS)
        updated = pl.DataFrame(
            [
                {"date": "2024-02-01", "docID": "D3", "edinetCode": "E1", "secCode": "10000", "filerName": "A", "docTypeCode": "140", "submitDateTime": "2024-02-01 09:00:00"},
            ]
        ).select(LISTUP_COLUMNS)

        targets = select_target_sec_codes(
            prepare_listup(merged),
            prepare_listup(updated),
            forced_sec_codes=["20000", "99999"],
        )

        self.assertEqual(targets, ["10000", "20000"])

    def test_calc_issued_shares_uses_current_year_instant(self) -> None:
        doc_df = pl.DataFrame(
            [
                {
                    "要素ID": "jpcrp_cor:TotalNumberOfIssuedSharesSummaryOfBusinessResults",
                    "項目名": "発行済株式総数（普通株式）、経営指標等",
                    "コンテキストID": "Prior1YearInstant_NonConsolidatedMember",
                    "相対年度": "前期末",
                    "連結・個別": "その他",
                    "期間・時点": "時点",
                    "ユニットID": "shares",
                    "単位": "",
                    "値": "100",
                },
                {
                    "要素ID": "jpcrp_cor:TotalNumberOfIssuedSharesSummaryOfBusinessResults",
                    "項目名": "発行済株式総数（普通株式）、経営指標等",
                    "コンテキストID": "CurrentYearInstant_NonConsolidatedMember",
                    "相対年度": "当期末",
                    "連結・個別": "その他",
                    "期間・時点": "時点",
                    "ユニットID": "shares",
                    "単位": "",
                    "値": "123",
                },
            ]
        )

        self.assertEqual(calc_issued_shares(doc_df, "annual"), 123.0)

    def test_updated_codes_are_rebuilt_from_full_history(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            base_path = root / "listup.csv"
            updated_path = root / "updated_listup.csv"
            write_listup(
                base_path,
                [
                    {
                        "date": "2024-01-01",
                        "docID": "D1",
                        "edinetCode": "E1",
                        "secCode": "10000",
                        "filerName": "A",
                        "docTypeCode": 120,
                        "submitDateTime": "2024-01-01 09:00:00",
                    },
                    {
                        "date": "2024-02-01",
                        "docID": "D2",
                        "edinetCode": "E1",
                        "secCode": "10000",
                        "filerName": "A",
                        "docTypeCode": 140,
                        "submitDateTime": "2024-02-01 09:00:00",
                    },
                    {
                        "date": "2024-01-01",
                        "docID": "D3",
                        "edinetCode": "E2",
                        "secCode": "20000",
                        "filerName": "B",
                        "docTypeCode": 120,
                        "submitDateTime": "2024-01-01 09:00:00",
                    },
                ],
            )
            write_listup(
                updated_path,
                [
                    {
                        "date": "2024-03-01",
                        "docID": "D4",
                        "edinetCode": "E1",
                        "secCode": "10000",
                        "filerName": "A",
                        "docTypeCode": 140,
                        "submitDateTime": "2024-03-01 09:00:00",
                    }
                ],
            )

            merged, updated = load_merged_listup(base_path, updated_path)
            prepared = prepare_listup(merged)
            targets = select_target_sec_codes(prepared, updated)
            target_history = prepared.filter(pl.col("secCode") == targets[0])

            self.assertEqual(targets, ["10000"])
            self.assertEqual(target_history.get_column("docID").to_list(), ["D1", "D2", "D4"])

    def test_latest_sec_code_moves_full_company_history_to_new_code(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            base_path = root / "listup.csv"
            updated_path = root / "updated_listup.csv"
            write_listup(
                base_path,
                [
                    {
                        "date": "2024-01-01",
                        "docID": "D1",
                        "edinetCode": "E1",
                        "secCode": "11110",
                        "filerName": "A",
                        "docTypeCode": 120,
                        "submitDateTime": "2024-01-01 09:00:00",
                    },
                    {
                        "date": "2024-02-01",
                        "docID": "D2",
                        "edinetCode": "E1",
                        "secCode": "11110",
                        "filerName": "A",
                        "docTypeCode": 140,
                        "submitDateTime": "2024-02-01 09:00:00",
                    },
                ],
            )
            write_listup(
                updated_path,
                [
                    {
                        "date": "2024-03-01",
                        "docID": "D3",
                        "edinetCode": "E1",
                        "secCode": "22220",
                        "filerName": "A",
                        "docTypeCode": 140,
                        "submitDateTime": "2024-03-01 09:00:00",
                    }
                ],
            )

            merged, updated = load_merged_listup(base_path, updated_path)
            prepared = prepare_listup(merged)

            self.assertEqual(select_target_sec_codes(prepared, updated), ["22220"])
            self.assertEqual(prepared.get_column("secCode").unique().to_list(), ["22220"])

    def test_existing_history_guard_rejects_missing_dates(self) -> None:
        with TemporaryDirectory() as temp_dir:
            out_path = Path(temp_dir) / "10000.csv"
            write_csv_utf8sig(
                pl.DataFrame(
                    {
                        "date": ["2024-01-01", "2024-02-01"],
                        "secCode": ["10000", "10000"],
                    }
                ),
                out_path,
            )

            with self.assertRaisesRegex(ValueError, "既存履歴が欠落"):
                validate_existing_history_is_preserved(
                    pl.DataFrame(
                        {
                            "date": ["2024-02-01", "2024-03-01"],
                            "secCode": ["10000", "10000"],
                        }
                    ),
                    out_path,
                )

    def test_existing_history_guard_accepts_appended_history(self) -> None:
        with TemporaryDirectory() as temp_dir:
            out_path = Path(temp_dir) / "10000.csv"
            write_csv_utf8sig(
                pl.DataFrame(
                    {
                        "date": ["2024-01-01", "2024-02-01"],
                        "secCode": ["10000", "10000"],
                    }
                ),
                out_path,
            )

            validate_existing_history_is_preserved(
                pl.DataFrame(
                    {
                        "date": ["2024-01-01", "2024-02-01", "2024-03-01"],
                        "secCode": ["10000", "10000", "10000"],
                    }
                ),
                out_path,
            )

    def test_process_sec_code_writes_all_rows_passed_to_worker(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            listup_for_code = prepare_listup(
                pl.DataFrame(
                    [
                        {
                            "date": "2024-01-01",
                            "docID": "D1",
                            "edinetCode": "E1",
                            "secCode": "10000",
                            "filerName": "A",
                            "docTypeCode": 120,
                            "submitDateTime": "2024-01-03 09:00:00",
                        },
                        {
                            "date": "2024-02-01",
                            "docID": "D2",
                            "edinetCode": "E1",
                            "secCode": "10000",
                            "filerName": "A",
                            "docTypeCode": 140,
                            "submitDateTime": "2024-02-01 09:00:00",
                        },
                        {
                            "date": "2024-03-01",
                            "docID": "D3",
                            "edinetCode": "E1",
                            "secCode": "10000",
                            "filerName": "A",
                            "docTypeCode": 140,
                            "submitDateTime": "2024-03-01 09:00:00",
                        },
                    ]
                )
            )
            doc_index = {
                "D1.csv": root / "D1.csv",
                "D2.csv": root / "D2.csv",
                "D3.csv": root / "D3.csv",
            }

            with (
                patch("index.read_doc_csv", return_value=pl.DataFrame()),
                patch("index.calc_issued_shares", return_value=1000.0),
                patch("index.calc_eps_bps_cfps", return_value=(1.0, 2.0, 3.0)),
            ):
                _process_sec_code(
                    "10000",
                    listup_for_code,
                    doc_index,
                    root,
                    protect_existing_history=True,
                )

            result = pl.read_csv(root / "10000.csv")
            self.assertEqual(result.height, 3)
            self.assertEqual(
                result.get_column("date").cast(pl.Utf8).to_list(),
                ["2024-01-01", "2024-02-01", "2024-03-01"],
            )
            self.assertEqual(
                result.get_column("docID").cast(pl.Utf8).to_list(),
                ["D1", "D2", "D3"],
            )
            self.assertEqual(
                result.get_column("submitDateTime").cast(pl.Utf8).to_list(),
                [
                    "2024-01-03 09:00:00",
                    "2024-02-01 09:00:00",
                    "2024-03-01 09:00:00",
                ],
            )
            self.assertEqual(
                result.get_column("FundamentalAvailableDate").cast(pl.Utf8).to_list(),
                ["2024-01-03", "2024-02-01", "2024-03-01"],
            )
            self.assertEqual(
                result.get_column("IssuedShares").cast(pl.Float64).to_list(),
                [1000.0, 1000.0, 1000.0],
            )
            self.assertEqual(
                result.get_column("EdinetHistoryStartDate").cast(pl.Utf8).to_list(),
                ["2024-01-01", "2024-01-01", "2024-01-01"],
            )

    def test_process_sec_code_writes_first_non_null_cfps_history_start_date(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            listup_for_code = prepare_listup(
                pl.DataFrame(
                    [
                        {
                            "date": "2024-01-01",
                            "docID": "D1",
                            "edinetCode": "E1",
                            "secCode": "10000",
                            "filerName": "A",
                            "docTypeCode": 120,
                            "submitDateTime": "2024-01-01 09:00:00",
                        },
                        {
                            "date": "2024-02-01",
                            "docID": "D2",
                            "edinetCode": "E1",
                            "secCode": "10000",
                            "filerName": "A",
                            "docTypeCode": 140,
                            "submitDateTime": "2024-02-01 09:00:00",
                        },
                        {
                            "date": "2024-03-01",
                            "docID": "D3",
                            "edinetCode": "E1",
                            "secCode": "10000",
                            "filerName": "A",
                            "docTypeCode": 140,
                            "submitDateTime": "2024-03-01 09:00:00",
                        },
                    ]
                )
            )
            doc_index = {
                "D1.csv": root / "D1.csv",
                "D2.csv": root / "D2.csv",
                "D3.csv": root / "D3.csv",
            }

            with (
                patch("index.read_doc_csv", return_value=pl.DataFrame()),
                patch("index.calc_issued_shares", return_value=1000.0),
                patch(
                    "index.calc_eps_bps_cfps",
                    side_effect=[
                        (1.0, None, 3.0),
                        (1.0, 2.0, 3.0),
                        (1.0, 4.0, 3.0),
                    ],
                ),
            ):
                _process_sec_code(
                    "10000",
                    listup_for_code,
                    doc_index,
                    root,
                    protect_existing_history=True,
                )

            result = pl.read_csv(root / "10000.csv")
            self.assertEqual(
                result.get_column("EdinetHistoryStartDate").cast(pl.Utf8).to_list(),
                ["2024-02-01", "2024-02-01", "2024-02-01"],
            )

    def test_full_rebuild_doc_read_failure_does_not_overwrite_existing_index(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            listup_path = root / "listup.csv"
            updated_path = root / "updated_listup.csv"
            doc_root = root / "doc"
            out_root = root / "index"
            write_listup(
                listup_path,
                [
                    {
                        "date": "2024-01-01",
                        "docID": "D1",
                        "edinetCode": "E1",
                        "secCode": "10000",
                        "filerName": "A",
                        "docTypeCode": 120,
                        "submitDateTime": "2024-01-01 09:00:00",
                    }
                ],
            )
            write_minimal_doc_csv(doc_root / "D1.csv")
            write_csv_utf8sig(
                pl.DataFrame(
                    {
                        "date": ["2023-01-01"],
                        "secCode": ["10000"],
                    }
                ),
                out_root / "10000.csv",
            )

            with temporary_cwd(root):
                with patch("index.read_doc_csv", side_effect=ValueError("bad csv")):
                    with self.assertRaisesRegex(RuntimeError, "上書きしません"):
                        run_index_rebuild(
                            listup_path=listup_path,
                            updated_listup_path=updated_path,
                            doc_root=doc_root,
                            out_root=out_root,
                            max_workers=0,
                        )

            result = pl.read_csv(out_root / "10000.csv")
            self.assertEqual(result.get_column("date").to_list(), ["2023-01-01"])

    def test_one_sec_code_history_guard_failure_does_not_stop_other_sec_codes(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            listup_path = root / "listup.csv"
            updated_path = root / "updated_listup.csv"
            doc_root = root / "doc"
            out_root = root / "index"
            write_listup(
                listup_path,
                [
                    {
                        "date": "2024-02-01",
                        "docID": "D_BAD_1",
                        "edinetCode": "E_BAD",
                        "secCode": "10000",
                        "filerName": "Bad",
                        "docTypeCode": 120,
                        "submitDateTime": "2024-02-01 09:00:00",
                    },
                    {
                        "date": "2024-01-01",
                        "docID": "D_GOOD_1",
                        "edinetCode": "E_GOOD",
                        "secCode": "20000",
                        "filerName": "Good",
                        "docTypeCode": 120,
                        "submitDateTime": "2024-01-01 09:00:00",
                    },
                ],
            )
            write_listup(
                updated_path,
                [
                    {
                        "date": "2024-03-01",
                        "docID": "D_BAD_2",
                        "edinetCode": "E_BAD",
                        "secCode": "10000",
                        "filerName": "Bad",
                        "docTypeCode": 120,
                        "submitDateTime": "2024-03-01 09:00:00",
                    },
                    {
                        "date": "2024-02-01",
                        "docID": "D_GOOD_2",
                        "edinetCode": "E_GOOD",
                        "secCode": "20000",
                        "filerName": "Good",
                        "docTypeCode": 120,
                        "submitDateTime": "2024-02-01 09:00:00",
                    },
                ],
            )
            for doc_id in ("D_BAD_1", "D_BAD_2", "D_GOOD_1", "D_GOOD_2"):
                write_minimal_doc_csv(doc_root / f"{doc_id}.csv")
            write_csv_utf8sig(
                pl.DataFrame(
                    {
                        "date": ["2024-01-01"],
                        "secCode": ["10000"],
                    }
                ),
                out_root / "10000.csv",
            )
            write_csv_utf8sig(
                pl.DataFrame(
                    {
                        "date": ["2024-01-01"],
                        "secCode": ["20000"],
                    }
                ),
                out_root / "20000.csv",
            )

            with temporary_cwd(root):
                run_index_rebuild(
                    listup_path=listup_path,
                    updated_listup_path=updated_path,
                    doc_root=doc_root,
                    out_root=out_root,
                    max_workers=0,
                )

                failed = pl.read_csv("state/failed_index_sec_codes.csv")
            self.assertEqual(
                failed.get_column("secCode").cast(pl.Utf8).to_list(),
                ["10000"],
            )
            self.assertEqual(
                pl.read_csv(out_root / "10000.csv").get_column("date").to_list(),
                ["2024-01-01"],
            )
            self.assertEqual(
                pl.read_csv(out_root / "20000.csv").get_column("date").to_list(),
                ["2024-01-01", "2024-02-01"],
            )


if __name__ == "__main__":
    unittest.main()
