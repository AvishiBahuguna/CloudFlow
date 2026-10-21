from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

from shared.eda_schemas import EDAResult, PlotArtifact


def save_categorical_bar_chart(
    result: EDAResult,
    column_name: str,
    run_dir: str | Path,
    *,
    plot_index: int = 1,
) -> PlotArtifact:
    if plot_index < 1:
        raise ValueError("plot_index must be positive.")

    if column_name not in result.categorical_summaries:
        raise ValueError(
            f"No categorical summary found for {column_name!r}."
        )

    summary = result.categorical_summaries[column_name]

    if not summary.top_values:
        raise ValueError(
            f"No observed categories to plot for {column_name!r}."
        )

    labels = [
        item.value
        for item in summary.top_values
    ]

    counts = [
        item.count
        for item in summary.top_values
    ]

    if len(set(labels)) != len(labels):
        labels = [
            f"{item.value} ({item.value_type})"
            for item in summary.top_values
        ]

    if summary.other_count > 0:
        labels.append("Other categories (combined)")
        counts.append(summary.other_count)

    relative_path = (
        Path("plots")
        / f"categorical_{plot_index:03d}.png"
    )

    output_path = Path(run_dir) / relative_path

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    title = f"{column_name}: category counts"

    figure = Figure(
        figsize=(9, max(3, 0.5 * len(labels) + 1.5)),
        layout="constrained",
    )

    FigureCanvasAgg(figure)

    axes = figure.subplots()

    positions = list(range(len(labels)))

    bars = axes.barh(
        positions,
        counts,
        color="#2563eb",
    )

    axes.set_yticks(positions, labels=labels)
    axes.invert_yaxis()

    axes.set_xlabel("Count (missing values excluded)")
    axes.set_title(title, parse_math=False)

    for label in axes.get_yticklabels():
        label.set_parse_math(False)

    axes.xaxis.set_major_locator(
        MaxNLocator(integer=True)
    )

    axes.set_xlim(0, max(counts) * 1.2)

    axes.bar_label(bars, padding=3)

    try:
        with output_path.open("xb") as output_file:
            figure.savefig(
                output_file,
                format="png",
                dpi=150,
            )
    finally:
        figure.clear()

    return PlotArtifact(
        column=column_name,
        path=relative_path.as_posix(),
        title=title,
        description=(
            "Counts of stored top categories; remaining known "
            "values are combined when present. "
            "Missing values are excluded."
        ),
    )