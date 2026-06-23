import unittest
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


if __name__ == "__main__":
    unittest.main()
