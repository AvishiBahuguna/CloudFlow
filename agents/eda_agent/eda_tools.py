from typing import BinaryIO

import numpy as np
import pandas as pd

from shared.eda_schemas import (
    CategoryFrequency,
    CategoricalSummary,
    EDAResult,
    NumericSummary,
)

def _finite_or_none(value) -> float | None:
    if pd.isna(value):
        return None

    number = float(value)

    if not np.isfinite(number):
        return None

    return number


def summarize_numeric_column( series: pd.Series, ) -> NumericSummary:
    observed_values = series.dropna()

    finite_values = observed_values[
        np.isfinite(observed_values)
    ]

    return NumericSummary(
        finite_count=len(finite_values),
        infinite_count=(
            len(observed_values) - len(finite_values)
        ),
        mean=_finite_or_none(finite_values.mean()),
        median=_finite_or_none(finite_values.median()),
        std=_finite_or_none(
            finite_values.std(ddof=1)
        ),
        minimum=_finite_or_none(finite_values.min()),
        maximum=_finite_or_none(finite_values.max()),
    )

def summarize_categorical_column(
    series: pd.Series,
    top_n: int = 10,
) -> CategoricalSummary:
    if top_n < 1:
        raise ValueError("top_n must be at least 1.")

    observed_values = series.dropna()
    non_missing_count = len(observed_values)

    if non_missing_count == 0:
        return CategoricalSummary(
            non_missing_count=0,
            unique_count=0,
        )

    counts = observed_values.value_counts(sort=False)

    counts = counts[counts > 0]

    top_counts = counts.nlargest(top_n)

    top_values = []
    shown_count = 0

    for value, count in top_counts.items():
        count = int(count)

        top_values.append(
            CategoryFrequency(
                value=str(value),
                value_type=type(value).__name__,
                count=count,
                percentage=(
                    count / non_missing_count
                ) * 100,
            )
        )

        shown_count += count

    return CategoricalSummary(
        non_missing_count=non_missing_count,
        unique_count=len(counts),
        top_values=top_values,
        other_count=non_missing_count - shown_count,
    )

def summarize_dataset(df: pd.DataFrame) -> EDAResult:
    if df.empty:
        raise ValueError(
            "The dataset must contain at least one row and one column."
        )

    row_count, column_count = df.shape

    numeric_columns = (
        df.select_dtypes(
            include=["number"],
            exclude=["complex", "timedelta"],
        )
        .columns
        .tolist()
    )

    categorical_columns = (
        df.select_dtypes(
            include=["object", "string", "category", "bool"]
        )
        .columns
        .tolist()
    )

    other_columns = []

    for column in df.columns:
        if (
            column not in numeric_columns
            and column not in categorical_columns
        ):
            other_columns.append(column)

    missing_counts = {}
    missing_percentages = {}

    for column in df.columns:
        missing_count = int(df[column].isna().sum())

        missing_counts[column] = missing_count

        missing_percentages[column] = (
            missing_count / row_count
        ) * 100

    numeric_summaries = {}

    for column in numeric_columns:
        numeric_summaries[column] = (
            summarize_numeric_column(df[column])
        )


    categorical_summaries = {}

    for column in categorical_columns:
        categorical_summaries[column] = (
            summarize_categorical_column(df[column])
        )

    return EDAResult(
        row_count=row_count,
        column_count=column_count,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        other_columns=other_columns,
        missing_counts=missing_counts,
        missing_percentages=missing_percentages,
        numeric_summaries=numeric_summaries,
        categorical_summaries=categorical_summaries,
    )
def summarize_csv(
    csv_file: BinaryIO,
    *,
    encoding: str = "utf-8",
    sep: str = ",",
) -> EDAResult:
    csv_file.seek(0)

    df = pd.read_csv(
        csv_file,
        encoding=encoding,
        sep=sep,
    )

    return summarize_dataset(df)