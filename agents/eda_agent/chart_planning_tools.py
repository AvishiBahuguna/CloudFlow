from collections import Counter

from shared.eda_schemas import (
    ChartConfig,
    ChartPlan,
    ChartRequest,
    EDAResult,
    SkippedChart,
)


def build_chart_plan(
    result: EDAResult,
    config: ChartConfig | None = None,
    *,
    requested_column: str | None = None,
) -> ChartPlan:
    config = (
        config or ChartConfig()
    ).model_copy(deep=True)

    candidates = {}
    skipped = {}
    duplicates = 0

    def offer(
        kind,
        columns,
        source,
        priority,
        usable_rows,
        reason,
        index=None,
        problem=None,
    ):
        nonlocal duplicates

        key = (kind, tuple(columns))

        if problem is not None:
            if key not in candidates:
                skipped[key] = SkippedChart(
                    kind=kind,
                    columns=columns,
                    reason=problem,
                )
            return False

        request = ChartRequest(
            kind=kind,
            columns=columns,
            source=source,
            source_index=index,
            priority=priority,
            usable_rows=usable_rows,
            reason=reason,
        )

        skipped.pop(key, None)
        previous = candidates.get(key)

        if previous is not None:
            duplicates += 1

            if (priority, -usable_rows) >= (
                previous.priority,
                -previous.usable_rows,
            ):
                return True

        candidates[key] = request
        return True

    def category(
        name,
        priority,
        source,
        explicit=False,
    ):
        summary = result.categorical_summaries.get(name)
        problem = None

        if summary is None or not summary.top_values:
            problem = "No observed categories are available."

        elif not explicit and summary.unique_count < 2:
            problem = (
                "The column has only one observed category."
            )

        elif not explicit and summary.unique_count > 20:
            problem = (
                "More than 20 categories would make a "
                "generic category view less useful."
            )

        return offer(
            "categorical_bar",
            [name],
            source,
            priority,
            summary.non_missing_count if summary else 0,
            "Category counts; remaining values are "
            "combined when necessary.",
            problem=problem,
        )

    def distribution(
        name,
        priority,
        source,
        explicit=False,
    ):
        summary = result.numeric_distributions.get(name)
        basic = result.numeric_summaries.get(name)
        problem = None

        if summary is None or not summary.histogram_counts:
            problem = (
                "No usable histogram summary is available."
            )

        elif not explicit and summary.finite_count < 2:
            problem = (
                "Fewer than two finite values are available."
            )

        elif (
            not explicit
            and basic is not None
            and basic.minimum == basic.maximum
        ):
            problem = (
                "The column is constant among its finite values."
            )

        return offer(
            "numeric_distribution",
            [name],
            source,
            priority,
            summary.finite_count if summary else 0,
            "Numerical distribution and potential outliers.",
            problem=problem,
        )

    focus = result.target_analysis

    target_name = (
        focus.config.target_column
        if focus is not None
        else None
    )

    if requested_column is not None:
        if requested_column not in result.categorical_summaries:
            raise ValueError(
                "--plot-column must name an available "
                "categorical column."
            )

        category(
            requested_column,
            0,
            "dataset",
            explicit=True,
        )

    if focus is not None:
        if focus.config.task_type == "classification":
            category(
                target_name, 1, "target", explicit=True
            )
        else:
            distribution(
                target_name, 1, "target", explicit=True
            )

    for source, owner in [
        ("target", focus),
        ("dataset", result),
    ]:
        if owner is None:
            continue

        for attribute, kind in [
            ("group_comparisons", "group_comparison"),
            (
                "categorical_comparisons",
                "categorical_comparison",
            ),
        ]:
            indexed = list(
                enumerate(getattr(owner, attribute))
            )

            indexed.sort(
                key=lambda pair: -pair[1].usable_rows
            )

            for rank, (index, comparison) in enumerate(indexed):
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

                offer(
                    kind,
                    columns,
                    source,
                    (
                        2 if rank == 0 else 4
                    ) if source == "target" else 6,
                    comparison.usable_rows,
                    (
                        "Target-related comparison."
                        if source == "target"
                        else "Comparison between dataset columns."
                    ),
                    index=index,
                )

    correlation = result.correlation_summary
    columns = correlation.columns if correlation else []
    pair_counts = []

    if correlation is not None:
        for index, first in enumerate(columns):
            for second in columns[index + 1:]:
                if (
                    correlation.matrix
                    .get(first, {})
                    .get(second)
                ) is not None:
                    pair_counts.append(
                        correlation.pair_counts[first][second]
                    )

    offer(
        "correlation_heatmap",
        columns,
        "dataset",
        3,
        min(pair_counts, default=0),
        "Overview of available numerical relationships.",
        problem=(
            None if pair_counts
            else "No numerical pair has an available correlation."
        ),
    )

    time = result.time_analysis

    if time is not None:
        offer(
            "time_row_counts",
            [time.config.date_column],
            "time",
            3,
            time.usable_rows,
            "Coverage of uploaded rows over time.",
            problem=(
                None if time.period_starts
                else "No calendar was produced; "
                "see time-analysis notes."
            ),
        )

        ranked = sorted(
            time.numeric_trends.items(),
            key=lambda item: -item[1].usable_rows,
        )

        context_rank = 0

        for name, trend in ranked:
            valid = sum(
                mean is not None
                for mean in trend.means
            ) >= 2

            if name == target_name:
                priority = 2
            else:
                priority = (
                    3 if context_rank == 0 else 5
                )

                if valid:
                    context_rank += 1

            offer(
                "time_numeric_mean",
                [time.config.date_column, name],
                "time",
                priority,
                trend.usable_rows,
                "Numerical averages over calendar periods.",
                problem=(
                    None if valid
                    else "Fewer than two periods have "
                    "usable numerical averages."
                ),
            )

    rank = 0

    for name, summary in sorted(
        result.numeric_distributions.items(),
        key=lambda item: -item[1].finite_count,
    ):
        if name != target_name:
            if distribution(
                name,
                3 if rank == 0 else 5,
                "dataset",
            ):
                rank += 1

    rank = 0

    for name, summary in sorted(
        result.categorical_summaries.items(),
        key=lambda item: -item[1].non_missing_count,
    ):
        if name != target_name:
            if category(
                name,
                3 if rank == 0 else 5,
                "dataset",
            ):
                rank += 1

    selected = []
    counts = Counter()

    for request in sorted(
        candidates.values(),
        key=lambda item: (
            item.priority,
            -item.usable_rows,
        ),
    ):
        reason = None

        if len(selected) >= config.max_charts:
            reason = (
                f"Overall limit of "
                f"{config.max_charts} charts reached."
            )

        elif counts[request.kind] >= config.max_per_kind:
            reason = (
                f"Limit of {config.max_per_kind} charts "
                "for this chart type reached."
            )

        if reason is not None:
            skipped[
                (request.kind, tuple(request.columns))
            ] = SkippedChart(
                kind=request.kind,
                columns=request.columns,
                reason=reason,
            )

        else:
            selected.append(request)
            counts[request.kind] += 1

    return ChartPlan(
        config=config,
        eligible_count=len(candidates),
        duplicates_removed=duplicates,
        selected=selected,
        skipped=list(skipped.values()),
        notes=[
            "Explicit requests and target views have "
            "priority. Overview charts provide a mix "
            "of chart types.",
            "Similar candidates are ordered by usable "
            "row counts; ties preserve their existing order.",
            "These are display priorities, "
            "not model feature importance.",
            "Relationship candidates come from already-"
            "calculated summaries. Existing analysis "
            "limits still apply.",
            "Calculated statistics remain available "
            "when their charts are skipped.",
        ],
    )