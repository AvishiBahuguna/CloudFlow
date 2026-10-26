import tempfile
from collections import Counter
from pathlib import Path

import pandas as pd

from agents.eda_agent.chart_planning_tools import (
    build_chart_plan,
)

from agents.eda_agent.chart_rendering_tools import (
    render_chart_plan,
)

from agents.eda_agent.eda_tools import (
    summarize_dataset,
)

from shared.eda_schemas import (
    ChartConfig,
    EDAResult,
    TargetConfig,
    TimeConfig,
)


def main():
    df = pd.DataFrame({
        "CustomerID": [10, 11, 12, 13, 14, 15],
        "Age": [20, 30, 40, 50, 60, 70],
        "Charges": [100, 200, 150, 300, 250, 400],
        "Constant": [7] * 6,
        "Contract": ["M", "M", "Y", "Y", "M", "Y"],
        "Region": ["N", "S", "N", "S", "S", "N"],
        "Churn": ["Yes", "No", "Yes", "No", "Yes", "No"],
    })

    result = summarize_dataset(
        df,
        target=TargetConfig(
            target_column="Churn",
            task_type="classification",
        ),
    )

    before = result.model_dump_json()

    config = ChartConfig(
        max_charts=5,
        max_per_kind=1,
    )

    plan = build_chart_plan(result, config)

    # Planning itself should not change the report.
    assert result.model_dump_json() == before
    assert plan == build_chart_plan(result, config)

    assert len(plan.selected) <= 5
    assert plan.selected[0].columns == ["Churn"]
    assert plan.duplicates_removed > 0

    counts = Counter(
        item.kind for item in plan.selected
    )
    assert all(count <= 1 for count in counts.values())

    keys = [
        (item.kind, tuple(item.columns))
        for item in plan.selected
    ]
    assert len(keys) == len(set(keys))

    assert all(
        "CustomerID" not in item.columns
        for item in plan.selected
    )

    assert any(
        item.columns == ["Constant"]
        and "constant" in item.reason
        for item in plan.skipped
    )

    manual = build_chart_plan(
        result,
        ChartConfig(max_charts=1),
        requested_column="Contract",
    )
    assert manual.selected[0].columns == ["Contract"]

    zero = result.model_copy(deep=True)
    zero.chart_plan = build_chart_plan(
        zero,
        ChartConfig(max_charts=0),
    )

    with tempfile.TemporaryDirectory() as folder:
        render_chart_plan(zero, folder)
        assert zero.plots == []
        assert not list(Path(folder).rglob("*.png"))

    result.chart_plan = plan

    with tempfile.TemporaryDirectory() as folder:
        render_chart_plan(result, folder)

        files = list(Path(folder).rglob("*.png"))

        assert len(files) == len(plan.selected)
        assert len(result.plots) == len(plan.selected)

        for plot in result.plots:
            assert (
                Path(folder) / plot.path
            ).read_bytes().startswith(
                b"\x89PNG\r\n\x1a\n"
            )

    loaded = EDAResult.model_validate_json(
        result.model_dump_json()
    )
    assert loaded == result

    # A time-mean request must create only one image.
    timed = summarize_dataset(
        pd.DataFrame({
            "Date": [
                "2026-01-05", "2026-01-20",
                "2026-03-03", "2026-03-04",
            ],
            "Revenue": [10, 20, 30, 40],
        }),
        target=TargetConfig(
            target_column="Revenue",
            task_type="regression",
        ),
        time_config=TimeConfig(
            date_column="Date",
        ),
    )

    timed.chart_plan = build_chart_plan(
        timed,
        ChartConfig(max_charts=2),
    )

    with tempfile.TemporaryDirectory() as folder:
        render_chart_plan(timed, folder)

        assert len(timed.plots) == 2
        assert len(list(Path(folder).rglob("*.png"))) == 2

        assert any(
            plot.kind == "time_numeric_mean"
            for plot in timed.plots
        )

        assert all(
            plot.kind != "time_row_counts"
            for plot in timed.plots
        )

    print("Chart planning checks passed.")


if __name__ == "__main__":
    main()