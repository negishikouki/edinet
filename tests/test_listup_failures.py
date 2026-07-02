import os
import unittest
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import requests

import listup


@contextmanager
def temporary_cwd(path: Path):
    original = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(original)


class ListupFailureStateTests(unittest.TestCase):
    def test_api_failure_date_is_persisted(self) -> None:
        with TemporaryDirectory() as temp_dir:
            with temporary_cwd(Path(temp_dir)):
                with patch(
                    "listup.requests.get",
                    side_effect=requests.exceptions.Timeout("timeout"),
                ):
                    result = listup.fetch_listup(
                        date(2024, 1, 1),
                        date(2024, 1, 1),
                        set(),
                    )

                self.assertTrue(result.empty)
                self.assertEqual(
                    listup.load_failed_listup_dates(),
                    {date(2024, 1, 1): "timeout"},
                )

    def test_failed_dates_are_added_to_next_fetch_dates(self) -> None:
        with TemporaryDirectory() as temp_dir:
            with temporary_cwd(Path(temp_dir)):
                listup.record_failed_listup_date(date(2024, 1, 1), "timeout")

                dates = listup.build_listup_fetch_dates(
                    date(2024, 1, 2),
                    date(2024, 1, 2),
                    set(listup.load_failed_listup_dates()),
                )

                self.assertEqual(dates, [date(2024, 1, 1), date(2024, 1, 2)])

    def test_successful_retry_clears_failed_date(self) -> None:
        response = Mock(spec=requests.Response)
        response.raise_for_status.return_value = None
        response.json.return_value = {"results": []}

        with TemporaryDirectory() as temp_dir:
            with temporary_cwd(Path(temp_dir)):
                listup.record_failed_listup_date(date(2024, 1, 1), "timeout")
                with patch("listup.requests.get", return_value=response):
                    result = listup.fetch_listup(
                        date(2024, 1, 2),
                        date(2024, 1, 2),
                        set(),
                    )

                self.assertTrue(result.empty)
                self.assertEqual(listup.load_failed_listup_dates(), {})


if __name__ == "__main__":
    unittest.main()
