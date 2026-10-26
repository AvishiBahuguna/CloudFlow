import numpy as np
import pandas as pd

from shared.eda_schemas import (
    CorrelationSummary,
    GroupComparison,
    GroupStatistics,
)


def summarize_correlations(
    df: pd.DataFrame,
    numeric_columns: list[str],
    *,
    min_pairs: int = 3,
    max_columns: int = 12,
) -> CorrelationSummary:
    if type(min_pairs) is not int or min_pairs < 3:
        raise ValueError(
            "min_pairs must be an integer "
            "of at least 3."
        )

    if type(max_columns) is not int or max_columns < 2:
        raise ValueError(
            "max_columns must be an integer "
            "of at least 2."
        )

    working = (
        df[numeric_columns]
        .astype(float)
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
    )

    selected = []
    excluded = {}

    for column in numeric_columns:
        values = working[column].dropna()

        if len(values) < min_pairs:
            excluded[column] = (
                "Too few finite observations."
            )

        elif values.nunique() < 2:
            excluded[column] = (
                "Only one distinct finite value."
            )

        elif len(selected) >= max_columns:
            excluded[column] = (
                "Column limit for this correlation view."
            )

        else:
            selected.append(column)

    summary = CorrelationSummary(
        min_pairs=min_pairs,
        columns=selected,
        excluded_columns=excluded,
        notes=[
            "Pearson correlation describes linear relationships.",
            (
                "Correlation does not establish causation. "
                "Small samples can give unstable results."
            ),
            (
                "Each comparison uses rows where both "
                "values are present and finite."
            ),
        ],
    )

    if len(selected) < 2:
        summary.notes.append(
            "At least two suitable numerical "
            "columns are required."
        )
        return summary

    working = working[selected]

    correlations = working.corr(
        method="pearson",
        min_periods=min_pairs,
    ).clip(-1, 1)

    available = working.notna().astype("int64")

    counts = available.T.dot(available)

    for left in selected:
        summary.matrix[left] = {}
        summary.pair_counts[left] = {}

        for right in selected:
            value = correlations.loc[left, right]

            summary.matrix[left][right] = (
                float(value)
                if np.isfinite(value)
                else None
            )

            summary.pair_counts[left][right] = int(
                counts.loc[left, right]
            )

    if any(
        summary.matrix[left][right] is None
        for left in selected
        for right in selected
        if left != right
    ):
        summary.notes.append(
            "Some comparisons are unavailable because "
            "of insufficient shared data, no shared "
            "variation, or numerical limitations."
        )

    if len(numeric_columns) > max_columns:
        summary.notes.append(
            f"This view includes up to {max_columns} "
            "eligible columns in dataset order. "
            "All columns retain their basic profiles."
        )

    return summary
def summarize_group_comparisons(
    df: pd.DataFrame,
    categorical_columns: list[str],
    numeric_columns: list[str],
    *,
    max_groups: int = 12,
    max_comparisons: int = 12,
) -> tuple[list[GroupComparison], list[str]]:
    if type(max_groups) is not int or max_groups < 2:
        raise ValueError(
            "max_groups must be an integer of at least 2."
        )

    if (
        type(max_comparisons) is not int
        or max_comparisons < 1
    ):
        raise ValueError(
            "max_comparisons must be a positive integer."
        )

    comparisons = []

    notes = [
        "Comparisons follow CSV column order; "
        "they are not ranked by importance.",
        "Group differences are descriptive and do not "
        "establish causation or statistical significance.",
    ]

    numeric_values = {}

    for column in numeric_columns:
        values = df[column].astype(float).replace(
            [np.inf, -np.inf],
            np.nan,
        )

        if values.nunique(dropna=True) < 2:
            notes.append(
                f"{column}: skipped because fewer than "
                "two distinct finite values exist."
            )
        else:
            numeric_values[column] = values

    if not categorical_columns or not numeric_values:
        notes.append(
            "At least one categorical and one varying "
            "numerical column are required."
        )
        return comparisons, notes

    for category in categorical_columns:
        unique_count = df[category].nunique(dropna=True)

        if unique_count < 2 or unique_count > max_groups:
            notes.append(
                f"{category}: skipped because it has "
                f"{unique_count} observed groups; "
                f"this view supports 2–{max_groups}."
            )
            continue

        
        keys = pd.Series(
            [
                None
                if pd.isna(value)
                else (type(value).__name__, str(value))
                for value in df[category]
            ],
            index=df.index,
            dtype="object",
        )

        for number, values in numeric_values.items():
            if len(comparisons) >= max_comparisons:
                notes.append(
                    f"Reached the {max_comparisons}-comparison "
                    "limit. Remaining pairs were not evaluated."
                )
                return comparisons, notes

            frame = pd.DataFrame({
                "group": keys,
                "number": values,
            }).dropna()

            if frame["group"].nunique() < 2:
                notes.append(
                    f"{category} / {number}: fewer than "
                    "two groups have usable numerical values."
                )
                continue

            groups = []

            for key, rows in frame.groupby(
                "group",
                sort=False,
                observed=True,
            ):
                value_type, label = key
                numbers = rows["number"]

                statistics = [
                    float(numbers.mean()),
                    float(numbers.median()),
                    float(numbers.min()),
                    float(numbers.max()),
                ]

                if not np.isfinite(statistics).all():
                    groups = []

                    notes.append(
                        f"{category} / {number}: skipped "
                        "because the numerical range could "
                        "not be summarized safely."
                    )
                    break

                groups.append(
                    GroupStatistics(
                        value=label,
                        value_type=value_type,
                        count=len(numbers),
                        mean=statistics[0],
                        median=statistics[1],
                        minimum=statistics[2],
                        maximum=statistics[3],
                    )
                )

            if not groups:
                continue

            comparison_notes = [
                "Rows with a missing category or "
                "missing/infinite numerical value are excluded."
            ]

            if any(group.count < 5 for group in groups):
                comparison_notes.append(
                    "Some groups have fewer than five usable "
                    "rows. Treat these as descriptive examples."
                )

            comparisons.append(
                GroupComparison(
                    category_column=category,
                    numeric_column=number,
                    usable_rows=len(frame),
                    excluded_rows=len(df) - len(frame),
                    groups=groups,
                    notes=comparison_notes,
                )
            )

    return comparisons, notes