from __future__ import annotations

import os

API_KEY_ENV_VAR = "EDINET_API_KEY"

LISTUP_COLUMNS = [
    "date",
    "docID",
    "edinetCode",
    "secCode",
    "filerName",
    "docTypeCode",
    "submitDateTime",
    "listQueryDate",
    "opeDateTime",
    "docInfoEditStatus",
    "withdrawalStatus",
    "disclosureStatus",
    "csvFlag",
    "xbrlFlag",
    "pdfFlag",
    "englishDocFlag",
]


def get_api_key() -> str:
    try:
        return os.environ[API_KEY_ENV_VAR]
    except KeyError as exc:
        raise RuntimeError(
            f"Set the {API_KEY_ENV_VAR} environment variable before using the EDINET API."
        ) from exc
