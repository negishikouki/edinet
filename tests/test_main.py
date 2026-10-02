import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import main


class MainPipelineTests(unittest.TestCase):
    def test_main_rebuilds_requested_codes_and_writes_status_report(self) -> None:
        with TemporaryDirectory() as temp_dir:
            original_cwd = Path.cwd()
            try:
                os.chdir(temp_dir)
                report_path = Path(temp_dir) / "repair-status.csv"
                codes_path = Path(temp_dir) / "repair-codes.txt"
                codes_path.write_text("20000\n", encoding="utf-8")
                statuses = [{"secCode": "20000", "status": "index_available"}]
                with (
                    patch("main.listup_main", return_value=False),
                    patch("main.fetch_main", return_value=False),
                    patch("main.refresh_recalc", return_value=statuses) as refresh_recalc,
                ):
                    main.main(
                        [
                            "--repair-sec-codes-file",
                            str(codes_path),
                            "--repair-status-report",
                            str(report_path),
                        ]
                    )
                self.assertEqual(
                    report_path.read_text(encoding="utf-8").strip().splitlines(),
                    ["secCode,status", "20000,index_available"],
                )
            finally:
                os.chdir(original_cwd)

        refresh_recalc.assert_called_once_with(["20000"])

    def test_main_skips_recalc_when_no_inputs_changed(self) -> None:
        with TemporaryDirectory() as temp_dir:
            original_cwd = Path.cwd()
            try:
                os.chdir(temp_dir)
                with (
                    patch("main.listup_main", return_value=False),
                    patch("main.fetch_main", return_value=False),
                    patch("main.refresh_recalc") as refresh_recalc,
                ):
                    main.main()
            finally:
                os.chdir(original_cwd)

        refresh_recalc.assert_not_called()

    def test_main_recalculates_when_a_pending_listup_exists(self) -> None:
        with TemporaryDirectory() as temp_dir:
            original_cwd = Path.cwd()
            try:
                os.chdir(temp_dir)
                main.UPDATED_LISTUP_PATH.write_text("pending", encoding="utf-8")
                with (
                    patch("main.listup_main", return_value=False),
                    patch("main.fetch_main", return_value=False),
                    patch("main.refresh_recalc") as refresh_recalc,
                ):
                    main.main()
            finally:
                os.chdir(original_cwd)

        refresh_recalc.assert_called_once_with()

    def test_main_recalculates_when_documents_changed(self) -> None:
        with TemporaryDirectory() as temp_dir:
            original_cwd = Path.cwd()
            try:
                os.chdir(temp_dir)
                with (
                    patch("main.listup_main", return_value=False),
                    patch("main.fetch_main", return_value=True),
                    patch("main.refresh_recalc") as refresh_recalc,
                ):
                    main.main()
            finally:
                os.chdir(original_cwd)

        refresh_recalc.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
