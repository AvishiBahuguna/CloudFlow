import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from agents.eda_agent.eda_tools import (
    summarize_dataset,
)

from agents.eda_agent.plotting_tools import (
    save_target_charts,
)

from shared.eda_schemas import (
    EDAResult,
    TargetConfig,
)


def main():
    df = pd.DataFrame({
        "Age": [20.0, np.nan, 30.0, 40.0],
        "MonthlyCharges": [100, 150, 200, 250],
        "Contract": [
            "Monthly", "Monthly", "Yearly", "Yearly",
        ],
        "Churn": ["Yes", "No", "No", "Yes"],
    })

    original = df.copy(deep=True)

    result = summarize_dataset(
        df,
        target=TargetConfig(
            target_column="Churn",
            task_type="classification",
        ),
    )

    focus = result.target_analysis

    assert focus is not None
    assert focus.usable_rows == 4
    assert focus.unique_count == 2
    assert len(focus.group_comparisons) == 2
    assert len(focus.categorical_comparisons) == 1

    assert all(
        item.comparison_column == "Churn"
        for item in focus.categorical_comparisons
    )

    pd.testing.assert_frame_equal(df, original)

    with tempfile.TemporaryDirectory() as folder:
        result.plots.extend(
            save_target_charts(result, folder)
        )

        assert focus.plot_paths

        # A second call should reuse existing charts.
        assert save_target_charts(result, folder) == []

        for path in focus.plot_paths:
            assert (
                Path(folder) / path
            ).read_bytes().startswith(
                b"\x89PNG\r\n\x1a\n"
            )

    loaded = EDAResult.model_validate_json(
        result.model_dump_json()
    )

    assert loaded == result

    regression = summarize_dataset(
        pd.DataFrame({
            "Hours": [1, 2, 3, 4],
            "Group": ["A", "A", "B", "B"],
            "Revenue": [10, 20, 30, 40],
        }),
        target=TargetConfig(
            target_column="Revenue",
            task_type="regression",
        ),
    )

    assert (
        regression.correlation_summary.columns[0]
        == "Revenue"
    )

    relation = (
        regression.target_analysis
        .numeric_correlations[0]
    )

    assert relation.feature == "Hours"
    assert np.isclose(relation.coefficient, 1.0)
    assert relation.pair_count == 4

    codes = pd.DataFrame({
        "Feature": [10, 20, 30, 40],
        "Outcome": [0, 1, 0, 1],
    })

    coded_result = summarize_dataset(
        codes,
        target=TargetConfig(
            target_column="Outcome",
            task_type="classification",
        ),
    )

    assert "Outcome" in coded_result.categorical_columns
    assert "Outcome" not in coded_result.numeric_columns
    assert codes["Outcome"].tolist() == [0, 1, 0, 1]

    for selection in [
        TargetConfig(
            target_column="Missing",
            task_type="classification",
        ),
        TargetConfig(
            target_column="Churn",
            task_type="regression",
        ),
    ]:
        try:
            summarize_dataset(
                df,
                target=selection,
            )
        except ValueError:
            pass
        else:
            raise AssertionError(
                "An invalid target was accepted."
            )

    plain = summarize_dataset(df)

    assert plain.target_analysis is None

    assert (
        plain.categorical_comparisons
        == result.categorical_comparisons
    )

    print("Target analysis checks passed.")


if __name__ == "__main__":
    main()