from itertools import combinations

import pandas as pd

from shared.eda_schemas import (
    CategoryLabel,
    CategoricalComparison,
)


def summarize_categorical_comparisons(
    df: pd.DataFrame,
    categorical_columns: list[str],
    *,
    max_categories: int = 8,
    max_comparisons: int = 12,
    focus_column: str | None = None,
) -> tuple[list[CategoricalComparison], list[str]]:
    if (
        type(max_categories) is not int
        or max_categories < 2
    ):
        raise ValueError(
            "max_categories must be an integer "
            "of at least 2."
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
        "Pairs follow CSV column order; "
        "they are not ranked by importance.",
        "Percentages are calculated within "
        "each usable group.",
        "These comparisons do not establish "
        "causation or statistical significance.",
    ]

    encoded = {}

    for column in categorical_columns:
        # Preserve distinctions such as number 1
        # versus text "1".
        keys = pd.Series(
            [
                None
                if pd.isna(value)
                else (
                    type(value).__name__,
                    str(value),
                )
                for value in df[column]
            ],
            dtype="object",
        )

        codes, levels = pd.factorize(
            keys,
            sort=False,
        )

        if not 2 <= len(levels) <= max_categories:
            notes.append(
                f"{column}: skipped because it has "
                f"{len(levels)} observed categories; "
                f"this view supports 2–{max_categories}."
            )
            continue

        encoded[column] = (codes, levels)
    if (
        focus_column is not None
        and focus_column not in encoded
    ):
        notes.append(
            f"{focus_column}: focused category comparisons "
            "are unavailable. The target must have "
            f"2–{max_categories} observed categories."
        )

        return comparisons, notes    
    if len(encoded) < 2:
        notes.append(
            "At least two suitable categorical "
            "columns are required."
        )
        return comparisons, notes
    pairs = (
        combinations(encoded, 2)
        if focus_column is None
        else (
            (name, focus_column)
            for name in encoded
            if name != focus_column
        )
    )

    for first, second in pairs:

        if len(comparisons) >= max_comparisons:
            notes.append(
                f"Reached the {max_comparisons}-comparison "
                "limit. Remaining pairs were not evaluated."
            )
            break

        first_codes, first_levels = encoded[first]
        second_codes, second_levels = encoded[second]

        usable = (
            (first_codes >= 0)
            & (second_codes >= 0)
        )

        if not usable.any():
            notes.append(
                f"{first} / {second}: "
                "no rows have both values."
            )
            continue

        table = pd.crosstab(
            pd.Series(
                first_codes[usable],
                name="group",
            ),
            pd.Series(
                second_codes[usable],
                name="comparison",
            ),
        ).sort_index().sort_index(axis=1)

        if min(table.shape) < 2:
            notes.append(
                f"{first} / {second}: fewer than two "
                "categories remain on one side after "
                "excluding missing values."
            )
            continue

        totals = table.sum(axis=1)

        percentages = (
            table.div(totals, axis=0) * 100
        )

        pair_notes = [
            "Rows missing either value are excluded "
            "from this comparison.",
            "Every percentage row sums to 100%, "
            "allowing for rounding.",
        ]

        if bool((totals < 5).any()):
            pair_notes.append(
                "Some groups have fewer than five "
                "usable rows. Treat the percentages "
                "as descriptive examples."
            )

        comparisons.append(
            CategoricalComparison(
                group_column=first,
                comparison_column=second,
                usable_rows=int(usable.sum()),
                excluded_rows=int((~usable).sum()),

                group_values=[
                    CategoryLabel(
                        value=first_levels[int(code)][1],
                        value_type=first_levels[int(code)][0],
                    )
                    for code in table.index
                ],

                comparison_values=[
                    CategoryLabel(
                        value=second_levels[int(code)][1],
                        value_type=second_levels[int(code)][0],
                    )
                    for code in table.columns
                ],

                group_counts=(
                    totals.astype(int).tolist()
                ),

                counts=(
                    table.to_numpy(dtype=int).tolist()
                ),

                percentages=(
                    percentages
                    .to_numpy(dtype=float)
                    .tolist()
                ),

                notes=pair_notes,
            )
        )

    return comparisons, notes