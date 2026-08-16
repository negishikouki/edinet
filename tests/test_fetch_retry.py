import io
import json
import unittest
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch
import zipfile

import requests

import fetch


def fetched_document_for_test(
    doc_id: str = "S100TEST",
    prefix: str = "jpcrp030000",
) -> fetch.FetchedDocument:
    source_name = f"{prefix}-asr-test.csv"
    dataframe = fetch.pd.DataFrame(
        {
            "要素ID": ["jpdei_cor:EDINETCodeDEI"],
            "値": ["E1"],
        }
    )
    return fetch.FetchedDocument(
        csvs=(
            fetch.FetchedCsv(
                source_name=source_name,
                prefix=prefix,
                is_body=True,
                dataframe=dataframe,
            ),
            fetch.FetchedCsv(
                source_name=f"jpaud-aai-test-{doc_id}.csv",
                prefix="jpaud",
                is_body=False,
                dataframe=dataframe,
            ),
        ),
        body_prefixes=(prefix,),
        body_csv_files=(source_name,),
        has_jpdei_cor=True,
    )


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
    @patch("fetch.get_document_response")
    def test_fetches_all_csvs_and_records_actual_body_prefix(
        self,
        mock_get_response: Mock,
    ) -> None:
        buffer = io.BytesIO()
        body_csv = fetch.pd.DataFrame(
            {
                "要素ID": ["jpdei_cor:EDINETCodeDEI"],
                "値": ["E1"],
            }
        ).to_csv(index=False, sep="\t").encode("utf-16")
        audit_csv = fetch.pd.DataFrame(
            {"要素ID": ["jpaud_cor:AuditExample"], "値": ["x"]}
        ).to_csv(index=False, sep="\t").encode("utf-16")
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr(
                "XBRL_TO_CSV/jpcrp080000-asr-test.csv",
                body_csv,
            )
            archive.writestr("XBRL_TO_CSV/jpaud-aai-test.csv", audit_csv)

        response = Mock()
        response.headers = {"Content-Type": "application/octet-stream"}
        response.content = buffer.getvalue()
        response.raise_for_status = Mock()
        mock_get_response.return_value = response

        fetched = fetch.fetch_all_prefix_dfs("S100TEST", ["jpcrp030000"])

        self.assertEqual(
            [csv.source_name for csv in fetched.csvs],
            ["jpaud-aai-test.csv", "jpcrp080000-asr-test.csv"],
        )
        self.assertEqual(fetched.body_prefixes, ("jpcrp080000",))
        self.assertTrue(fetched.has_jpdei_cor)

    def test_existing_csv_is_treated_as_downloaded(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            out_dir = root / "jpcrp030000"
            out_dir.mkdir()
            (out_dir / "S100TEST.csv").write_text("a,b\n1,2\n", encoding="utf-8")

            completed = fetch.build_completed_docid_set(root, {"S100TEST": "120"})
            fetched = fetch.build_fetched_docid_set(root, {"S100TEST": "120"})

            self.assertIn("S100TEST", completed)
            self.assertIn("S100TEST", fetched)

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
                fetched_document_for_test(),
                root,
            )

            completed = fetch.build_completed_docid_set(root, {"S100TEST": "120"})

            self.assertIn("S100TEST", completed)
            self.assertTrue(fetch.complete_marker_path(root, "S100TEST").exists())
            self.assertTrue((root / "jpcrp030000" / "S100TEST.csv").exists())
            self.assertTrue(
                (root / "jpaud" / "S100TEST__jpaud-aai-test-S100TEST.csv").exists()
            )
            manifest = json.loads(
                fetch.document_manifest_path(root, "S100TEST").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(manifest["bodyPrefixes"], ["jpcrp030000"])
            self.assertTrue(manifest["hasJpdeiCor"])

    def test_persist_partial_zip_creates_complete_marker_but_empty_zip_does_not(
        self,
    ) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)

            fetch.persist_doc_csvs(
                "S100PARTIAL",
                "120",
                fetched_document_for_test("S100PARTIAL"),
                root,
            )
            with self.assertRaisesRegex(ValueError, "no body CSV"):
                fetch.persist_doc_csvs(
                    "S100EMPTY",
                    "120",
                    fetch.FetchedDocument((), (), (), False),
                    root,
                )

            completed = fetch.build_fetched_docid_set(root)

            self.assertIn("S100PARTIAL", completed)
            self.assertNotIn("S100EMPTY", completed)
            self.assertFalse(fetch.complete_marker_path(root, "S100EMPTY").exists())

    def test_failed_fetch_doc_is_recorded(self) -> None:
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "failed_fetch_docs.csv"

            fetch.record_failed_fetch_doc("S100TEST", "HTTP 503", path)

            self.assertEqual(fetch.load_failed_fetch_docs(path), {"S100TEST": "HTTP 503"})

    def test_completed_failures_are_removed_from_failure_ledger(self) -> None:
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "failed_fetch_docs.csv"
            fetch.record_failed_fetch_doc("S100DONE", "old validation error", path)
            fetch.record_failed_fetch_doc("S100RETRY", "HTTP 503", path)

            removed = fetch.clear_failed_fetch_docs_for_completed(
                {"S100DONE"},
                path,
            )

            self.assertEqual(removed, 1)
            self.assertEqual(
                fetch.load_failed_fetch_docs(path),
                {"S100RETRY": "HTTP 503"},
            )

    def test_not_found_failure_is_classified_as_permanent(self) -> None:
        self.assertTrue(
            fetch.is_permanent_fetch_error(
                "EDINET returned JSON (not ZIP). message="
                "{'metadata': {'status': '404', 'message': 'Not Found'}}"
            )
        )
        self.assertFalse(fetch.is_permanent_fetch_error("HTTP 503"))

    @patch("fetch.fetch_all_prefix_dfs")
    def test_main_skips_permanent_not_found_failure(
        self,
        mock_fetch: Mock,
    ) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "listup.csv").write_text(
                "date,docID,edinetCode,secCode,filerName,docTypeCode,submitDateTime\n"
                "2026-08-14,S100404,E1,10000,A,120,2026-08-14 09:00:00\n",
                encoding="utf-8",
            )
            failed_path = root / "state" / "failed_fetch_docs.csv"
            fetch.record_failed_fetch_doc(
                "S100404",
                "S100404: EDINET returned JSON (not ZIP). message="
                "{'metadata': {'status': '404', 'message': 'Not Found'}}",
                failed_path,
            )
            original_cwd = Path.cwd()
            try:
                os.chdir(root)
                fetch.main()
            finally:
                os.chdir(original_cwd)

            mock_fetch.assert_not_called()

    @patch("fetch.persist_doc_csvs")
    @patch("fetch.fetch_all_prefix_dfs")
    def test_main_retries_recorded_failure_without_new_listup(
        self,
        mock_fetch: Mock,
        mock_persist: Mock,
    ) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "listup.csv").write_text(
                "date,docID,edinetCode,secCode,filerName,docTypeCode,submitDateTime\n"
                "2026-08-14,S100TEST,E1,10000,A,120,2026-08-14 09:00:00\n",
                encoding="utf-8",
            )
            failed_path = root / "state" / "failed_fetch_docs.csv"
            fetch.record_failed_fetch_doc("S100TEST", "HTTP 503", failed_path)
            mock_fetch.return_value = fetched_document_for_test()
            original_cwd = Path.cwd()
            try:
                os.chdir(root)
                fetch.main()
            finally:
                os.chdir(original_cwd)

            mock_fetch.assert_called_once_with("S100TEST")
            mock_persist.assert_called_once()
            self.assertEqual(fetch.load_failed_fetch_docs(failed_path), {})

    @patch("fetch.persist_doc_csvs")
    @patch("fetch.fetch_all_prefix_dfs")
    def test_main_flushes_failure_cleanup_on_keyboard_interrupt(
        self,
        mock_fetch: Mock,
        mock_persist: Mock,
    ) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "listup.csv").write_text(
                "date,docID,edinetCode,secCode,filerName,docTypeCode,submitDateTime\n"
                "2026-08-14,S100DONE,E1,10000,A,120,2026-08-14 09:00:00\n"
                "2026-08-14,S100STOP,E1,10000,A,120,2026-08-14 09:00:00\n",
                encoding="utf-8",
            )
            failed_path = root / "state" / "failed_fetch_docs.csv"
            fetch.record_failed_fetch_doc("S100DONE", "old validation error", failed_path)
            mock_fetch.side_effect = [fetched_document_for_test(), KeyboardInterrupt()]
            original_cwd = Path.cwd()
            try:
                os.chdir(root)
                with self.assertRaises(KeyboardInterrupt):
                    fetch.main()
            finally:
                os.chdir(original_cwd)

            mock_persist.assert_called_once()
            self.assertEqual(fetch.load_failed_fetch_docs(failed_path), {})

    @patch("fetch.fetch_all_prefix_dfs")
    def test_main_skips_failed_doc_when_downloaded_marker_exists(
        self,
        mock_fetch: Mock,
    ) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "listup.csv").write_text(
                "date,docID,edinetCode,secCode,filerName,docTypeCode,submitDateTime\n"
                "2026-08-14,S100PARTIAL,E1,10000,A,120,2026-08-14 09:00:00\n",
                encoding="utf-8",
            )
            marker = root / "doc" / ".complete" / "S100PARTIAL.complete"
            marker.parent.mkdir(parents=True)
            marker.write_text("2026-08-16T00:00:00+00:00\n", encoding="utf-8")
            failed_path = root / "state" / "failed_fetch_docs.csv"
            fetch.record_failed_fetch_doc("S100PARTIAL", "HTTP 503", failed_path)
            original_cwd = Path.cwd()
            try:
                os.chdir(root)
                fetch.main()
            finally:
                os.chdir(original_cwd)

            mock_fetch.assert_not_called()


if __name__ == "__main__":
    unittest.main()
