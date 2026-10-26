import csv
from io import StringIO
from typing import BinaryIO

import pandas as pd

from shared.eda_schemas import CSVReadLimits


def load_csv_dataset(
    csv_file: BinaryIO,
    *,
    encoding: str = "utf-8-sig",
    sep: str = ",",
    limits: CSVReadLimits | None = None,
) -> pd.DataFrame:
    limits = limits or CSVReadLimits()

    if (
        not isinstance(sep, str)
        or len(sep) != 1
        or sep in {'"', "\r", "\n", "\x00"}
    ):
        raise ValueError(
            "Use a single separator character, "
            "such as a comma, semicolon or tab."
        )

    csv_file.seek(0)

    raw = csv_file.read(
        limits.max_bytes + 1
    )

    if not isinstance(raw, bytes):
        raise ValueError(
            "Open the CSV in binary mode: 'rb'."
        )

    if len(raw) > limits.max_bytes:
        raise ValueError(
            f"CSV exceeds the "
            f"{limits.max_bytes:,}-byte limit."
        )

    if not raw:
        raise ValueError(
            "The CSV file is empty."
        )

    try:
        text = raw.decode(
            encoding
        ).removeprefix("\ufeff")

    except (
        UnicodeDecodeError,
        LookupError,
    ) as error:
        raise ValueError(
            "Cannot read the CSV using "
            f"encoding {encoding!r}."
        ) from error

    if "\x00" in text:
        raise ValueError(
            "The CSV contains unsupported "
            "null characters."
        )

    reader = csv.reader(
        StringIO(text, newline=""),
        delimiter=sep,
        strict=True,
    )

    checked_csv = StringIO(
        newline=""
    )

    writer = csv.writer(
        checked_csv,
        delimiter=sep,
    )

    try:
        header = next(
            (
                row
                for row in reader
                if row
            ),
            None,
        )

        if header is None or any(
            not name.strip()
            for name in header
        ):
            raise ValueError(
                "CSV must have a header with "
                "non-empty column names."
            )

        if len(header) != len(set(header)):
            raise ValueError(
                "Duplicate CSV column names "
                "were found."
            )

        column_count = len(header)

        if column_count > limits.max_columns:
            raise ValueError(
                "CSV exceeds the "
                f"{limits.max_columns}-column limit."
            )

        writer.writerow(header)

        row_count = 0

        for row in reader:
            if not row:
                continue

            if len(row) != column_count:
                raise ValueError(
                    "CSV record ending at line "
                    f"{reader.line_num} "
                    f"has {len(row)} values; "
                    f"expected {column_count}."
                )

            row_count += 1

            if row_count > limits.max_rows:
                raise ValueError(
                    "CSV exceeds the "
                    f"{limits.max_rows:,}-row limit."
                )

            if (
                row_count * column_count
                > limits.max_cells
            ):
                raise ValueError(
                    "CSV exceeds the "
                    f"{limits.max_cells:,}-cell limit."
                )

            writer.writerow(row)

    except csv.Error as error:
        raise ValueError(
            f"CSV formatting error: {error}"
        ) from error

    if row_count == 0:
        raise ValueError(
            "CSV must contain at least "
            "one data row."
        )

    frame = pd.read_csv(
        StringIO(checked_csv.getvalue()),
        sep=sep,
        skip_blank_lines=False,
        low_memory=False,
        on_bad_lines="error",
    )

    if frame.shape != (
        row_count,
        column_count,
    ):
        raise ValueError(
            "Parsed CSV dimensions differ "
            "from the validated records."
        )

    return frame