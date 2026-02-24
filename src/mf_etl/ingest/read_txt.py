"""Read CSV-like stock TXT files into a raw Polars DataFrame."""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path

import polars as pl

LOGGER = logging.getLogger(__name__)

RAW_COLUMNS: tuple[str, ...] = (
    "raw_ticker",
    "raw_per",
    "raw_date",
    "raw_time",
    "raw_open",
    "raw_high",
    "raw_low",
    "raw_close",
    "raw_volume",
    "raw_openint",
)
EXPECTED_COLUMN_COUNT = len(RAW_COLUMNS)
HEADER_TOKENS: tuple[str, ...] = (
    "TICKER",
    "PER",
    "DATE",
    "TIME",
    "OPEN",
    "HIGH",
    "LOW",
    "CLOSE",
    "VOL",
    "OPENINT",
)
DELIMITER_CANDIDATES: tuple[str, ...] = (",", "\t", "|", ";")


@dataclass(frozen=True, slots=True)
class TxtReadResult:
    """Container for parsed raw rows and malformed-row metadata."""

    data: pl.DataFrame
    rejects: pl.DataFrame
    skipped_header: bool
    delimiter: str


def _empty_raw_df() -> pl.DataFrame:
    """Return empty raw frame with stable schema."""

    schema: dict[str, pl.DataType] = {column: pl.String for column in RAW_COLUMNS}
    schema["source_line_no"] = pl.Int64
    return pl.DataFrame(schema=schema)


def _empty_reject_df() -> pl.DataFrame:
    """Return empty reject frame with stable schema."""

    return pl.DataFrame(
        schema={
            "source_line_no": pl.Int64,
            "raw_line": pl.String,
            "parsed_column_count": pl.Int64,
            "reason": pl.String,
        }
    )


def _detect_delimiter(first_line: str) -> str:
    """Infer delimiter by highest occurrence in the first non-empty line."""

    counts = {delimiter: first_line.count(delimiter) for delimiter in DELIMITER_CANDIDATES}
    winner = max(counts, key=counts.get)
    return winner if counts[winner] > 0 else ","


def _normalize_field(value: str | None) -> str | None:
    """Trim whitespace and map empty strings to null."""

    if value is None:
        return None
    cleaned = value.strip()
    return cleaned if cleaned != "" else None


def _is_header_row(fields: list[str]) -> bool:
    """Check whether a row matches the optional source header."""

    normalized = [field.replace("\ufeff", "").strip().strip("<>").upper() for field in fields]
    return tuple(normalized[:EXPECTED_COLUMN_COUNT]) == HEADER_TOKENS


def read_stock_txt_with_rejects(
    path: Path,
    delimiter: str | None = None,
    logger: logging.Logger | None = None,
) -> TxtReadResult:
    """Read one TXT file into raw columns.

    `source_line_no` is 1-based and references the original file line number.
    If a header row is present and skipped, subsequent data rows keep their original
    line numbers from the source file.
    """

    effective_logger = logger or LOGGER
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        raw_lines = handle.readlines()

    first_non_empty = next((line for line in raw_lines if line.strip()), "")
    active_delimiter = delimiter or _detect_delimiter(first_non_empty)

    rows: list[dict[str, object]] = []
    reject_rows: list[dict[str, object]] = []
    header_checked = False
    header_skipped = False

    for line_no, raw_line in enumerate(raw_lines, start=1):
        stripped_line = raw_line.strip()
        if not stripped_line:
            continue

        parsed_fields = next(csv.reader([stripped_line], delimiter=active_delimiter))
        if not header_checked:
            header_checked = True
            if _is_header_row(parsed_fields):
                header_skipped = True
                continue

        original_count = len(parsed_fields)
        reason: str | None = None
        if original_count < EXPECTED_COLUMN_COUNT:
            reason = f"too_few_columns:{original_count}"
            parsed_fields = [*parsed_fields, *([None] * (EXPECTED_COLUMN_COUNT - len(parsed_fields)))]
        elif original_count > EXPECTED_COLUMN_COUNT:
            reason = f"too_many_columns:{original_count}"
            parsed_fields = parsed_fields[:EXPECTED_COLUMN_COUNT]

        if reason is not None:
            effective_logger.warning(
                "read_stock_txt.malformed_row path=%s line=%s reason=%s",
                path,
                line_no,
                reason,
            )
            reject_rows.append(
                {
                    "source_line_no": line_no,
                    "raw_line": stripped_line,
                    "parsed_column_count": original_count,
                    "reason": reason,
                }
            )

        row = {column: _normalize_field(value) for column, value in zip(RAW_COLUMNS, parsed_fields)}
        row["source_line_no"] = line_no
        rows.append(row)

    if rows:
        data = pl.DataFrame(
            rows,
            schema_overrides={
                **{column: pl.String for column in RAW_COLUMNS},
                "source_line_no": pl.Int64,
            },
        )
    else:
        data = _empty_raw_df()

    rejects = (
        pl.DataFrame(
            reject_rows,
            schema_overrides={
                "source_line_no": pl.Int64,
                "raw_line": pl.String,
                "parsed_column_count": pl.Int64,
                "reason": pl.String,
            },
        )
        if reject_rows
        else _empty_reject_df()
    )
    return TxtReadResult(data=data, rejects=rejects, skipped_header=header_skipped, delimiter=active_delimiter)


def read_stock_txt(path: Path, separator: str | None = None) -> pl.DataFrame:
    """Backward-compatible reader that returns only the raw data frame."""

    return read_stock_txt_with_rejects(path=path, delimiter=separator).data
