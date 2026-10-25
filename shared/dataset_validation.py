import csv
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.utils.multiclass import type_of_target

from shared.modeling import TrainingInput


def load_dataset(dataset_path: str) -> pd.DataFrame:
    path = Path(dataset_path).expanduser()

    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset file was not found: {path}"
        )

    if path.suffix.lower() != ".csv":
        raise ValueError("Only CSV files are supported.")

    # Inspect the original CSV header and records.
    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.reader(file, strict=True)
        columns = next(reader, [])

        if not columns or any(
            not column.strip() for column in columns
        ):
            raise ValueError(
                "CSV must contain non-empty column names."
            )

        if len(columns) != len(set(columns)):
            raise ValueError(
                "CSV contains duplicate column names."
            )

        for row in reader:
            if not row:
                continue

            if len(row) != len(columns):
                raise ValueError(
                    f"Incorrect number of values near "
                    f"CSV line {reader.line_num}."
                )

    return pd.read_csv(path, encoding="utf-8-sig")


def validate_dataset(
    df: pd.DataFrame,
    training_input: TrainingInput,
) -> None:
    if df.empty:
        raise ValueError(
            "The dataset contains no data rows."
        )

    required_columns = [
        training_input.target_column,
        *training_input.feature_columns,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Required columns are missing: {missing_columns}"
        )

    target = df[training_input.target_column]

    if target.isna().any():
        raise ValueError(
            "The target contains missing values."
        )

    if target.astype(str).str.strip().eq("").any():
        raise ValueError(
            "The target contains blank values."
        )

    if pd.api.types.is_numeric_dtype(target):
        if not np.isfinite(target.to_numpy()).all():
            raise ValueError(
                "The target contains infinite values."
            )

    if training_input.task_type == "classification":
        target_type = type_of_target(target)

        if target_type not in {"binary", "multiclass"}:
            raise ValueError(
                "Classification requires discrete class labels."
            )

        if target.nunique() < 2:
            raise ValueError(
                "Classification requires at least two classes."
            )

    elif training_input.task_type == "regression":
        if (
            not pd.api.types.is_numeric_dtype(target)
            or pd.api.types.is_bool_dtype(target)
        ):
            raise ValueError(
                "Regression requires a numerical target."
            )

        if target.nunique() < 2:
            raise ValueError(
                "For this workflow, the regression target "
                "must contain at least two different values."
            )