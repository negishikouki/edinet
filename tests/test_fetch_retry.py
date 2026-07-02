import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import requests

import fetch


class GetDocumentResponseTests(unittest.TestCase):
    @patch("fetch.time.sleep")
    @patch("fetch.requests.get")
    def test_retries_communication_errors_up_to_three_times(
        self,
        mock_get: Mock,
        mock_sleep: Mock,
    ) -> None:
        response = Mock(spec=requests.Response)
        mock_get.side_effect = [
            requests.exceptions.ConnectionError("first"),
            requests.exceptions.Timeout("second"),
            requests.exceptions.ConnectionError("third"),
            response,
        ]

        actual = fetch.get_document_response("S100TEST", "https://example.test", {})

        self.assertIs(actual, response)
        self.assertEqual(mock_get.call_count, 4)
        self.assertEqual(
            [call.args[0] for call in mock_sleep.call_args_list],
            [1, 2, 4],
        )

    @patch("fetch.time.sleep")
    @patch("fetch.requests.get")
    def test_raises_after_three_retries(
        self,
        mock_get: Mock,
        mock_sleep: Mock,
    ) -> None:
        mock_get.side_effect = requests.exceptions.ConnectionError("failed")

        with self.assertRaises(requests.exceptions.ConnectionError):
            fetch.get_document_response("S100TEST", "https://example.test", {})

        self.assertEqual(mock_get.call_count, 4)
        self.assertEqual(mock_sleep.call_count, 3)

    @patch("fetch.time.sleep")
    @patch("fetch.requests.get")
    def test_does_not_retry_non_communication_errors(
        self,
        mock_get: Mock,
        mock_sleep: Mock,
    ) -> None:
        mock_get.side_effect = ValueError("invalid response")

        with self.assertRaises(ValueError):
            fetch.get_document_response("S100TEST", "https://example.test", {})

        mock_get.assert_called_once()
        mock_sleep.assert_not_called()

    @patch("fetch.time.sleep")
    @patch("fetch.requests.get")
    def test_retries_retryable_http_status_codes(
        self,
        mock_get: Mock,
        mock_sleep: Mock,
    ) -> None:
        retry_response = Mock(spec=requests.Response)
        retry_response.status_code = 503
        success_response = Mock(spec=requests.Response)
        success_response.status_code = 200
        mock_get.side_effect = [retry_response, success_response]

        actual = fetch.get_document_response("S100TEST", "https://example.test", {})

        self.assertIs(actual, success_response)
        self.assertEqual(mock_get.call_count, 2)
        mock_sleep.assert_called_once_with(1)

    @patch("fetch.time.sleep")
    @patch("fetch.requests.get")
    def test_returns_retryable_http_status_after_retries_are_exhausted(
        self,
        mock_get: Mock,
        mock_sleep: Mock,
    ) -> None:
        response = Mock(spec=requests.Response)
        response.status_code = 429
        mock_get.return_value = response

        actual = fetch.get_document_response("S100TEST", "https://example.test", {})

        self.assertIs(actual, response)
        self.assertEqual(mock_get.call_count, 4)
        self.assertEqual(mock_sleep.call_count, 3)


class FetchCompletionStateTests(unittest.TestCase):
    def test_incomplete_doc_with_one_csv_is_not_completed(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            out_dir = root / "jpcrp030000"
            out_dir.mkdir()
            (out_dir / "S100TEST.csv").write_text("a,b\n1,2\n", encoding="utf-8")

            completed = fetch.build_completed_docid_set(root, {"S100TEST": "120"})
            fetched = fetch.build_fetched_docid_set(root, {"S100TEST": "120"})

            self.assertNotIn("S100TEST", completed)
            self.assertNotIn("S100TEST", fetched)

    def test_doc_without_complete_marker_or_required_csvs_is_refetched(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)

            completed = fetch.build_completed_docid_set(root, {"S100TEST": "120"})

            self.assertFalse(fetch.docid_already_fetched("S100TEST", completed))

    def test_successful_persist_creates_complete_marker(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            fetch.persist_doc_csvs(
                "S100TEST",
                "120",
                {
                    "jpcrp030000": fetch.pd.DataFrame({"a": [1]}),
                    "jpdei": fetch.pd.DataFrame({"a": [2]}),
                },
                root,
            )

            completed = fetch.build_completed_docid_set(root, {"S100TEST": "120"})

            self.assertIn("S100TEST", completed)
            self.assertTrue(fetch.complete_marker_path(root, "S100TEST").exists())

    def test_failed_fetch_doc_is_recorded(self) -> None:
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "failed_fetch_docs.csv"

            fetch.record_failed_fetch_doc("S100TEST", "HTTP 503", path)

            self.assertEqual(fetch.load_failed_fetch_docs(path), {"S100TEST": "HTTP 503"})


if __name__ == "__main__":
    unittest.main()
