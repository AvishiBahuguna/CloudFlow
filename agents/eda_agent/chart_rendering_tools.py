from collections import Counter
from pathlib import Path

from agents.eda_agent.plotting_tools import (
    save_categorical_bar_chart,
    save_categorical_comparison_chart,
    save_correlation_heatmap,
    save_group_comparison_chart,
    save_numeric_distribution_chart,
    save_time_charts,
)

from shared.eda_schemas import EDAResult


def render_chart_plan(
    result: EDAResult,
    run_dir: str | Path,
) -> None:
    plan = result.chart_plan

    if plan is None:
        raise ValueError(
            "Build a chart plan before rendering."
        )

    if result.plots:
        raise ValueError(
            "Charts already exist in this result. "
            "Replace the old chart loops with the planner."
        )

    keys = [
        (item.kind, tuple(item.columns))
        for item in plan.selected
    ]

    counts = Counter(
        item.kind for item in plan.selected
    )

    if (
        len(keys) != len(set(keys))
        or len(keys) > plan.config.max_charts
        or any(
            count > plan.config.max_per_kind
            for count in counts.values()
        )
    ):
        raise ValueError(
            "The chart plan violates its limits "
            "or has duplicates."
        )

    for index, request in enumerate(
        plan.selected,
        start=1,
    ):
        kind = request.kind

        if kind == "categorical_bar":
            plot = save_categorical_bar_chart(
                result=result,
                column_name=request.columns[0],
                run_dir=run_dir,
                plot_index=index,
            )

        elif kind == "numeric_distribution":
            plot = save_numeric_distribution_chart(
                result=result,
                column_name=request.columns[0],
                run_dir=run_dir,
                plot_index=index,
            )

        elif kind == "correlation_heatmap":
            plot = save_correlation_heatmap(
                result=result,
                run_dir=run_dir,
            )

        elif kind in {
            "group_comparison",
            "categorical_comparison",
        }:
            owner = (
                result
                if request.source == "dataset"
                else result.target_analysis
            )

            if (
                owner is None
                or request.source_index is None
            ):
                raise ValueError(
                    "The comparison referenced "
                    "by the plan is missing."
                )

            attribute = (
                "group_comparisons"
                if kind == "group_comparison"
                else "categorical_comparisons"
            )

            comparison = getattr(
                owner,
                attribute,
            )[request.source_index]

            columns = (
                [
                    comparison.category_column,
                    comparison.numeric_column,
                ]
                if kind == "group_comparison"
                else [
                    comparison.group_column,
                    comparison.comparison_column,
                ]
            )

            if columns != request.columns:
                raise ValueError(
                    "The analysis changed. "
                    "Rebuild its chart plan."
                )

            function = (
                save_group_comparison_chart
                if kind == "group_comparison"
                else save_categorical_comparison_chart
            )

            plot = function(
                comparison=comparison,
                run_dir=run_dir,
                plot_index=index,
            )

        else:
            plots = save_time_charts(
                result=result,
                run_dir=run_dir,
                include_counts=(
                    kind == "time_row_counts"
                ),
                numeric_columns=(
                    []
                    if kind == "time_row_counts"
                    else [request.columns[1]]
                ),
                plot_index_start=index,
            )

            if len(plots) != 1:
                raise ValueError(
                    "A selected time chart "
                    "could not be created."
                )

            plot = plots[0]

        if plot is None:
            raise ValueError(
                "A selected chart could not be created."
            )

        request.path = plot.path
        result.plots.append(plot)

    if result.target_analysis is not None:
        name = (
            result.target_analysis
            .config.target_column
        )

        result.target_analysis.plot_paths = [
            plot.path
            for plot in result.plots
            if (
                plot.column == name
                or name in plot.columns
            )
        ]

    if result.time_analysis is not None:
        name = (
            result.time_analysis
            .config.date_column
        )

        result.time_analysis.plot_paths = [
            plot.path
            for plot in result.plots
            if (
                plot.kind in {
                    "time_row_counts",
                    "time_numeric_mean",
                }
                and plot.columns
                and plot.columns[0] == name
            )
        ]