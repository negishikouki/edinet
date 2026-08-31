from __future__ import annotations

import math
import re
from numbers import Real
from typing import Any

import pandas as pd
import polars as pl

SEC_CODE_FLOAT_REGEX = r"^([0-9]+)\.0+$"
SEC_CODE_FLOAT_PATTERN = re.compile(SEC_CODE_FLOAT_REGEX)


def is_missing_scalar(value: object) -> bool:
    if value is None or value is pd.NA or value is pd.NaT:
        return True
    return isinstance(value, Real) and math.isnan(float(value))


def normalize_sec_code_text(value: Any) -> str | None:
    if pd.isna(value):
        return None

    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return None

    return SEC_CODE_FLOAT_PATTERN.sub(r"\1", text)


def normalize_sec_code_expr(expr: pl.Expr) -> pl.Expr:
    sec_code = expr.cast(pl.Utf8).str.strip_chars()
    return (
        pl.when(
            sec_code.is_null()
            | (sec_code == "")
            | sec_code.fill_null("").str.to_lowercase().is_in(["nan", "none"])
        )
        .then(None)
        .otherwise(sec_code.str.replace(SEC_CODE_FLOAT_REGEX, "$1"))
    )
