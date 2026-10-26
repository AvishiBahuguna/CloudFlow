import numpy as np
import pandas as pd

from shared.eda_schemas import NumericDistribution


def summarize_numeric_distribution(
    series: pd.Series,
) -> NumericDistribution:
    observed = (
        series.dropna()
        .astype(float)
        .to_numpy()
    )

    values = observed[np.isfinite(observed)]

    summary = NumericDistribution(
        finite_count=len(values),
        missing_count=int(series.isna().sum()),
        infinite_count=int(np.isinf(observed).sum()),
    )

    if len(values) == 0:
        summary.notes.append(
            "No finite numerical values to analyse."
        )
        return summary

    # Calculate the edges and centre of the middle half.
    with np.errstate(over="ignore", invalid="ignore"):
        q1, median, q3 = np.quantile(
            values,
            [0.25, 0.5, 0.75],
            method="linear",
        )

        iqr = q3 - q1

    if not np.isfinite([q1, median, q3, iqr]).all():
        summary.notes.append(
            "Numerical range is too extreme to "
            "calculate safe quartiles."
        )
        return summary

    summary.q1 = float(q1)
    summary.median = float(median)
    summary.q3 = float(q3)
    summary.iqr = float(iqr)

    summary.lower_whisker = float(values.min())
    summary.upper_whisker = float(values.max())

    # Calculate the limits for potential outliers.
    with np.errstate(over="ignore", invalid="ignore"):
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

    if len(values) < 4:
        summary.notes.append(
            "Fewer than four finite values: outlier "
            "checking was skipped. Box-plot whiskers "
            "show the full observed range."
        )

    elif not np.isfinite([lower, upper]).all():
        summary.notes.append(
            "Outlier checking was skipped because "
            "its limits overflowed. Box-plot whiskers "
            "show the full observed range."
        )

    else:
        outside = (
            (values < lower)
            | (values > upper)
        )

        inside = values[~outside]

        summary.lower_fence = float(lower)
        summary.upper_fence = float(upper)

        summary.lower_whisker = min(
            float(inside.min()),
            float(q1),
        )

        summary.upper_whisker = max(
            float(inside.max()),
            float(q3),
        )

        count = int(outside.sum())

        summary.potential_outlier_count = count

        summary.potential_outlier_percentage = (
            count / len(values) * 100
        )

        summary.outlier_sample = (
            values[outside][:20].tolist()
        )

        if iqr == 0:
            summary.notes.append(
                "The middle 50% has no spread. "
                "Review any outlier flags carefully."
            )

    summary.notes.append(
        "Quartiles use linear interpolation. "
        "Potential outliers use the 1.5 x IQR rule "
        "and are not automatically errors."
    )

    if summary.potential_outlier_count is not None:
        if summary.potential_outlier_count > 20:
            summary.notes.append(
                "Only the first 20 flagged values are "
                "stored for display; the count includes "
                "all flagged values."
            )

    # Choose a manageable number of histogram ranges.
    bins = (
        1
        if values.min() == values.max()
        else min(
            30,
            int(np.ceil(np.sqrt(len(values)))),
        )
    )

    try:
        with np.errstate(
            over="ignore",
            invalid="ignore",
            divide="ignore",
        ):
            counts, edges = np.histogram(
                values,
                bins=bins,
            )

            widths = np.diff(edges)
            span = edges[-1] - edges[0]

        if (
            not np.isfinite(edges).all()
            or not np.isfinite(widths).all()
            or not np.isfinite(span)
            or not (widths > 0).all()
            or int(counts.sum()) != len(values)
        ):
            raise ValueError(
                "Histogram boundaries are not usable."
            )

        summary.histogram_edges = edges.tolist()
        summary.histogram_counts = counts.tolist()

    except (
        ValueError,
        OverflowError,
        FloatingPointError,
    ):
        summary.notes.append(
            "The distribution chart was skipped because "
            "histogram boundaries could not be "
            "represented safely."
        )

    return summary