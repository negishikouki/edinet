import os
import json
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

    def test_forbidden_error_is_raised_and_not_persisted(self) -> None:
        response = Mock(spec=requests.Response)
        response.status_code = 403
        error = requests.exceptions.HTTPError(
            "403 Client Error: Forbidden",
            response=response,
        )
        response.raise_for_status.side_effect = error

        with TemporaryDirectory() as temp_dir:
            with temporary_cwd(Path(temp_dir)):
                with patch("listup.requests.get", return_value=response):
                    with self.assertRaisesRegex(
                        requests.exceptions.HTTPError,
                        "403 Client Error: Forbidden",
                    ):
                        listup.fetch_listup(
                            date(2024, 1, 1),
                            date(2024, 1, 2),
                            set(),
                        )

                self.assertFalse(Path("state/failed_listup_dates.csv").exists())

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

    def test_success_persists_raw_json_hash_and_event_fields(self) -> None:
        payload = {
            "results": [
                {
                    "docID": "S100TEST",
                    "edinetCode": "E1",
                    "secCode": "13010",
                    "filerName": "A",
                    "docTypeCode": "120",
                    "submitDateTime": "2024-01-02 09:00:00",
                    "opeDateTime": "2024-01-02 09:05:00",
                    "docInfoEditStatus": "1",
                    "withdrawalStatus": "0",
                    "disclosureStatus": "0",
                    "csvFlag": "1",
                    "xbrlFlag": "1",
                    "pdfFlag": "1",
                    "englishDocFlag": "0",
                }
            ]
        }
        response = Mock(spec=requests.Response)
        response.raise_for_status.return_value = None
        response.json.return_value = payload
        response.content = json.dumps(payload, ensure_ascii=False).encode("utf-8")

        with TemporaryDirectory() as temp_dir:
            with temporary_cwd(Path(temp_dir)):
                with patch("listup.requests.get", return_value=response):
                    result = listup.fetch_listup(
                        date(2024, 1, 2),
                        date(2024, 1, 2),
                        set(),
                    )

                self.assertEqual(result["csvFlag"].tolist(), ["1"])
                self.assertEqual(result["docInfoEditStatus"].tolist(), ["1"])
                response_dirs = list(
                    (Path("state") / "list_responses" / "2024" / "2024-01-02").glob(
                        "*.manifest.json"
                    )
                )
                self.assertEqual(len(response_dirs), 1)
                manifest = json.loads(response_dirs[0].read_text(encoding="utf-8"))
                raw_path = response_dirs[0].parent / manifest["rawJson"]["path"]
                self.assertEqual(
                    manifest["rawJson"]["sha256"],
                    listup.sha256_bytes(raw_path.read_bytes()),
                )
                events_path = response_dirs[0].parent / manifest["eventsCsv"]["path"]
                events = listup.pd.read_csv(events_path, dtype=str)
                self.assertEqual(events["opeDateTime"].tolist(), ["2024-01-02 09:05:00"])

    def test_no_changes_preserves_pending_updated_listup(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            pending = root / "updated_listup.csv"
            pending.write_text(
                ",".join(listup.LISTUP_COLUMNS)
                + "\n"
                + "2024-01-02,S100TEST,E1,13010,A,120,2024-01-02 09:00:00,,,,,,,,,\n",
                encoding="utf-8",
            )
            with temporary_cwd(root):
                with patch(
                    "listup.fetch_listup",
                    return_value=listup.pd.DataFrame(columns=listup.LISTUP_COLUMNS),
                ):
                    changed = listup.main()

                self.assertFalse(changed)
                self.assertTrue(pending.exists())


if __name__ == "__main__":
    unittest.main()
