import numpy as np
import pandas as pd

from pandas.api.types import (
    is_bool_dtype,
    is_complex_dtype,
    is_numeric_dtype,
)

from agents.eda_agent.categorical_relationship_tools import (
    summarize_categorical_comparisons,
)

from agents.eda_agent.relationship_tools import (
    summarize_group_comparisons,
)

from shared.eda_schemas import (
    ColumnRole,
    EDAResult,
    TargetAnalysis,
    TargetConfig,
    TargetCorrelation,
)


def prepare_target_overrides(
    df: pd.DataFrame,
    target: TargetConfig | None,
    role_overrides: dict[str, ColumnRole] | None,
) -> dict[str, ColumnRole]:
    overrides = dict(role_overrides or {})

    if target is None:
        return overrides

    column = target.target_column

    if not column.strip() or column not in df.columns:
        raise ValueError(
            f"Target column {column!r} was not found. "
            "Use the exact CSV column name."
        )

    series = df[column]

    numeric = (
        is_numeric_dtype(series.dtype)
        and not is_bool_dtype(series.dtype)
        and not is_complex_dtype(series.dtype)
    )

    if is_complex_dtype(series.dtype):
        raise ValueError(
            "Complex numbers are not supported as targets."
        )

    if target.task_type == "regression" and not numeric:
        raise ValueError(
            "A regression target must already have "
            "a real numerical dtype."
        )

    if target.task_type == "classification" and numeric:
        values = (
            series.dropna()
            .to_numpy(dtype=float)
        )

        if not np.isfinite(values).all():
            raise ValueError(
                "Numerical class labels contain infinity. "
                "Review these target values during preparation."
            )

    allowed = (
        {"categorical", "boolean"}
        if target.task_type == "classification"
        else {"numeric"}
    )

    if (
        column in overrides
        and overrides[column] not in allowed
    ):
        raise ValueError(
            "The target role override conflicts "
            "with the selected task type."
        )

    overrides.setdefault(
        column,
        (
            "categorical"
            if target.task_type == "classification"
            else "numeric"
        ),
    )

    return overrides


def build_target_analysis(
    df: pd.DataFrame,
    result: EDAResult,
    target: TargetConfig,
) -> TargetAnalysis:
    column = target.target_column
    series = df[column]

    usable = series.dropna()

    if target.task_type == "regression":
        usable = usable[
            np.isfinite(
                usable.to_numpy(dtype=float)
            )
        ]

    analysis = TargetAnalysis(
        config=target,
        missing_count=int(series.isna().sum()),
        usable_rows=len(usable),
        excluded_rows=len(df) - len(usable),
        unique_count=int(usable.nunique()),
        notes=[
            "The target and task type were explicitly "
            "selected by the caller.",
            "Target relationships are descriptive, "
            "not model feature importance.",
        ],
    )

    if len(usable) == 0:
        analysis.notes.append(
            "No usable target values exist. "
            "Target comparisons are unavailable."
        )

    elif analysis.unique_count == 1:
        analysis.notes.append(
            "The target has only one usable value. "
            "Review the selected target and dataset "
            "before modelling."
        )

    if target.task_type == "classification":
        analysis.class_summary = (
            result.categorical_summaries[column]
        )

        if analysis.unique_count > 1:
            largest = max(
                item.percentage
                for item in analysis.class_summary.top_values
            )

            if largest >= 90:
                analysis.notes.append(
                    "One class contains at least 90% "
                    "of observed target values. "
                    "Review the class balance."
                )

            if analysis.unique_count == len(usable):
                analysis.notes.append(
                    "Every observed target value is different. "
                    "Review whether classification is intended."
                )

        analysis.group_comparisons, group_notes = (
            summarize_group_comparisons(
                df,
                [column],
                [
                    name
                    for name in result.numeric_columns
                    if name != column
                ],
            )
        )

        (
            analysis.categorical_comparisons,
            category_notes,
        ) = summarize_categorical_comparisons(
            df,
            result.categorical_columns,
            focus_column=column,
        )

        analysis.notes.extend(
            group_notes + category_notes
        )

    else:
        analysis.numeric_summary = (
            result.numeric_summaries[column]
        )

        analysis.group_comparisons, group_notes = (
            summarize_group_comparisons(
                df,
                [
                    name
                    for name in result.categorical_columns
                    if name != column
                ],
                [column],
            )
        )

        analysis.notes.extend(group_notes)

        correlation = result.correlation_summary

        if (
            correlation is not None
            and column in correlation.columns
        ):
            analysis.numeric_correlations = [
                TargetCorrelation(
                    feature=name,
                    coefficient=(
                        correlation.matrix[column][name]
                    ),
                    pair_count=(
                        correlation.pair_counts[column][name]
                    ),
                )
                for name in correlation.columns
                if name != column
            ]

        else:
            reason = (
                correlation.excluded_columns.get(
                    column,
                    "Unavailable.",
                )
                if correlation is not None
                else "Unavailable."
            )

            analysis.notes.append(
                "Target correlations are unavailable: "
                f"{reason}"
            )

        analysis.notes.append(
            "Numerical target relationships use the "
            "correlation summary's column limit and "
            "minimum paired-row rules. The target is "
            "considered first, followed by other "
            "numerical columns in CSV order."
        )

    return analysis