import numpy as np
import pandas as pd

from shared.eda_schemas import (
    EDAResult,
    TimeAnalysis,
    TimeConfig,
    TimeNumericTrend,
)


def build_time_analysis(
    df: pd.DataFrame,
    result: EDAResult,
    config: TimeConfig,
) -> TimeAnalysis:
    column = config.date_column

    if not column.strip() or column not in df.columns:
        raise ValueError(
            f"Date column {column!r} was not found. "
            "Use the exact CSV column name."
        )

    source = df[column].reset_index(drop=True)

    strings = (
        source.astype("string")
        .str.strip()
    )

    iso_shape = strings.str.fullmatch(
        r"\d{4}-\d{2}-\d{2}(?:[T ].+)?",
        na=False,
    )

    dates = pd.to_datetime(
        strings.where(iso_shape),
        format="ISO8601",
        errors="coerce",
        utc=True,
    )

    analysis = TimeAnalysis(
        config=config,
        missing_count=int(source.isna().sum()),
        invalid_count=int(
            (source.notna() & dates.isna()).sum()
        ),
        usable_rows=int(dates.notna().sum()),
        excluded_rows=int(dates.isna().sum()),
        notes=[
            "Dates use ISO-style YYYY-MM-DD values "
            "with optional timestamps.",
            "Times are grouped in UTC; dates without "
            "a timezone are treated as UTC.",
            "Weeks start on Monday. Monthly and yearly "
            "periods start on the first day.",
            "Counts describe rows in the uploaded dataset.",
            "Empty periods have zero uploaded rows "
            "and an unavailable numerical mean.",
            "These summaries describe observed data; "
            "they are not forecasts.",
        ],
    )

    if analysis.usable_rows == 0:
        analysis.notes.append(
            "No valid ISO-style dates were found. "
            "Review the selected column or prepare "
            "dates as YYYY-MM-DD."
        )
        return analysis

    analysis.first_date = dates.min().isoformat()
    analysis.last_date = dates.max().isoformat()

    frequency = {
        "day": "D",
        "week": "W-SUN",
        "month": "M",
        "year": "Y-DEC",
    }[config.frequency]

    # Dates are already in UTC.
    # Remove the timezone for calendar grouping.
    buckets = (
        dates.dt.tz_localize(None)
        .dt.to_period(frequency)
    )

    first = buckets.min()
    last = buckets.max()

    analysis.period_count = (
        last.ordinal - first.ordinal + 1
    )

    analysis.observed_period_count = int(
        buckets.nunique()
    )

    if analysis.period_count > config.max_periods:
        analysis.notes.append(
            f"This range needs {analysis.period_count} "
            f"periods, exceeding the "
            f"{config.max_periods}-period limit. "
            "Time aggregation and charts were skipped. "
            "Choose a coarser frequency or a shorter range."
        )
        return analysis

    calendar = pd.period_range(
        first,
        last,
        freq=frequency,
    )

    analysis.period_starts = [
        period.start_time.date().isoformat()
        for period in calendar
    ]

    analysis.row_counts = (
        buckets.value_counts()
        .reindex(calendar, fill_value=0)
        .astype(int)
        .tolist()
    )

    columns = list(result.numeric_columns)
    target = result.target_analysis

    if (
        target is not None
        and target.config.task_type == "regression"
    ):
        name = target.config.target_column

        columns = [name] + [
            column
            for column in columns
            if column != name
        ]

    for name in columns:
        values = (
            df[name].reset_index(drop=True)
            .astype(float)
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
        )

        frame = pd.DataFrame({
            "period": buckets,
            "value": values,
        }).dropna()

        grouped = frame.groupby(
            "period",
            sort=True,
            observed=True,
        )["value"]

        counts = (
            grouped.count()
            .reindex(calendar, fill_value=0)
        )

        with np.errstate(
            over="ignore",
            invalid="ignore",
        ):
            means = (
                grouped.mean()
                .reindex(calendar)
            )

        safe_means = [
            float(value)
            if pd.notna(value) and np.isfinite(value)
            else None
            for value in means
        ]

        trend = TimeNumericTrend(
            usable_rows=len(frame),
            excluded_rows=len(df) - len(frame),
            counts=counts.astype(int).tolist(),
            means=safe_means,
        )

        if any(
            count > 0 and mean is None
            for count, mean in zip(
                trend.counts,
                trend.means,
            )
        ):
            trend.notes.append(
                "Some means were unavailable because "
                "the numerical range overflowed."
            )

        if sum(
            mean is not None
            for mean in safe_means
        ) < 2:
            trend.notes.append(
                "A numerical time chart requires usable "
                "means in at least two periods."
            )

        if bool(
            ((counts > 0) & (counts < 5)).any()
        ):
            trend.notes.append(
                "Some period averages use fewer than "
                "five values. Treat these as "
                "descriptive examples."
            )

        analysis.numeric_trends[name] = trend

    if analysis.observed_period_count < 2:
        analysis.notes.append(
            "Only one period contains records; "
            "between-period comparisons are limited."
        )

    return analysis