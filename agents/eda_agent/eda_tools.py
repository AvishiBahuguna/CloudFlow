from typing import BinaryIO
from shared.eda_schemas import CSVReadLimits
import numpy as np
import pandas as pd

from shared.eda_schemas import (
    CategoryFrequency,
    CategoricalSummary,
    ColumnRole,
    DatasetOverview,
    EDAResult,
    NumericSummary,
    CSVReadLimits,
)
from agents.eda_agent.column_tools import (
    interpret_columns,
)
from agents.eda_agent.csv_tools import (
    load_csv_dataset,
)
from agents.eda_agent.relationship_tools import (
    summarize_correlations,
    summarize_group_comparisons,
)
from agents.eda_agent.distribution_tools import (
    summarize_numeric_distribution,
)

from agents.eda_agent.categorical_relationship_tools import (
    summarize_categorical_comparisons,
)

from agents.eda_agent.target_tools import (
    build_target_analysis,
    prepare_target_overrides,
)
from shared.eda_schemas import TargetConfig

from shared.eda_schemas import TimeConfig

from agents.eda_agent.time_tools import (
    build_time_analysis,
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

def summarize_dataset_overview(
    df: pd.DataFrame,
) -> DatasetOverview:
    if df.empty:
        raise ValueError(
            "The dataset must contain at least "
            "one row and one column."
        )

    missing_mask = df.isna()

    total_cells = int(df.size)

    missing_cells = int(
        missing_mask.sum().sum()
    )

    rows_with_missing = int(
        missing_mask.any(axis=1).sum()
    )

    duplicate_rows = int(
        df.duplicated().sum()
    )

    unique_counts = {
        column: int(count)
        for column, count in (
            df.nunique(dropna=True).items()
        )
    }

    dtype_counts = {
        dtype: int(count)
        for dtype, count in (
            df.dtypes
            .astype(str)
            .value_counts()
            .items()
        )
    }

    return DatasetOverview(
        total_cells=total_cells,
        missing_cells=missing_cells,
        missing_cell_percentage=(
            missing_cells / total_cells
        ) * 100,
        rows_with_missing=rows_with_missing,
        complete_rows=(
            len(df) - rows_with_missing
        ),
        duplicate_rows=duplicate_rows,
        duplicate_row_percentage=(
            duplicate_rows / len(df)
        ) * 100,
        empty_columns=[
            column
            for column, count in unique_counts.items()
            if count == 0
        ],
        constant_columns=[
            column
            for column, count in unique_counts.items()
            if count == 1
        ],
        unique_counts=unique_counts,
        dtype_counts=dtype_counts,
    )

def summarize_dataset(df: pd.DataFrame,*,role_overrides: dict[str,ColumnRole,] | None = None,target: TargetConfig | None = None,time_config: TimeConfig | None = None,) -> EDAResult:
    if df.empty:
        raise ValueError(
            "The dataset must contain at least one row and one column."
        )
    if not df.columns.is_unique:
        raise ValueError(
            "Dataset column names must be unique."
        )

    if any(
        not isinstance(column, str)
        or not column.strip()
        for column in df.columns
    ):
        raise ValueError(
            "Dataset column names must be "
            "non-empty strings."
        )
    
    row_count, column_count = df.shape

    effective_overrides = prepare_target_overrides(
        df,
        target,
        role_overrides,
    )

    column_interpretations = interpret_columns(
        df,
        role_overrides=effective_overrides,
    )

    numeric_columns = [
        column
        for column, info in (
            column_interpretations.items()
        )
        if info.role == "numeric"
    ]

    categorical_columns = [
        column
        for column, info in (
            column_interpretations.items()
        )
        if info.role in {
            "categorical",
            "boolean",
        }
    ]

    other_columns = [
        column
        for column in df.columns
        if (
            column not in numeric_columns
            and column not in categorical_columns
        )
    ]

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

    correlation_columns = list(numeric_columns)

    if (
        target is not None
        and target.task_type == "regression"
    ):
        correlation_columns.remove(
            target.target_column
        )

        correlation_columns.insert(
            0,
            target.target_column,
        )    
    correlation_summary = summarize_correlations(
        df,
        correlation_columns,
    )    

    group_comparisons, group_notes = summarize_group_comparisons(
        df,
        categorical_columns,
        numeric_columns,
    )
    numeric_distributions = {
        column: summarize_numeric_distribution(
            df[column]
        )
        for column in numeric_columns
    }
    categorical_comparisons, category_notes = (
        summarize_categorical_comparisons(
            df,
            categorical_columns,
        )
    )
    result = EDAResult(
        dataset_overview=summarize_dataset_overview(df),
        row_count=row_count,
        column_count=column_count,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        other_columns=other_columns,
        missing_counts=missing_counts,
        missing_percentages=missing_percentages,
        numeric_summaries=numeric_summaries,
        categorical_summaries=categorical_summaries,
        column_interpretations=column_interpretations,
        correlation_summary=correlation_summary,
        group_comparisons=group_comparisons,
        group_comparison_notes=group_notes,
        numeric_distributions=numeric_distributions,
        categorical_comparisons=categorical_comparisons,
        categorical_comparison_notes=category_notes,
    )
    if target is not None:
        result.target_analysis = build_target_analysis(
            df,
            result,
            target,
        )

    if time_config is not None:
        result.time_analysis = build_time_analysis(
            df,
            result,
            time_config,
        )    

    return result
def summarize_csv(
    csv_file: BinaryIO,
    *,
    encoding: str = "utf-8-sig",
    sep: str = ",",
    role_overrides: dict[str, ColumnRole] | None = None,
    csv_limits: CSVReadLimits | None = None,
    target: TargetConfig | None = None,
    time_config: TimeConfig | None = None,
) -> EDAResult:
    df = load_csv_dataset(
        csv_file,
        encoding=encoding,
        sep=sep,
        limits=csv_limits,
    )

    return summarize_dataset(
        df,
        role_overrides=role_overrides,
        target=target,
        time_config=time_config,
    )