from __future__ import annotations

import os

API_KEY_ENV_VAR = "EDINET_API_KEY"
DEFAULT_API_KEY = ""

LISTUP_COLUMNS = [
    "date",
    "docID",
    "edinetCode",
    "secCode",
    "filerName",
    "docTypeCode",
    "submitDateTime",
]


def get_api_key() -> str:
    return os.environ.get(API_KEY_ENV_VAR, DEFAULT_API_KEY)
