import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import polars as pl

from index import (_process_sec_code, calc_issued_shares, load_merged_listup,
                   prepare_listup, select_target_sec_codes,
                   validate_existing_history_is_preserved, write_csv_utf8sig)

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


class IncrementalIndexTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
