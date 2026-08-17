import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import main


class MainPipelineTests(unittest.TestCase):
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
