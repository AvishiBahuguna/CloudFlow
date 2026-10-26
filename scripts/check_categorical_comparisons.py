import tempfile
from pathlib import Path

import pandas as pd

from agents.eda_agent.eda_tools import (
    summarize_dataset,
)

from agents.eda_agent.plotting_tools import (
    save_categorical_comparison_chart,
)

from shared.eda_schemas import EDAResult


def main():
    df = pd.DataFrame({
        "Contract": [
            "Monthly",
            "Monthly",
            "Monthly",
            "Monthly",
            "Yearly",
            "Yearly",
            None,
            "Monthly",
        ],
        "Churn": [
            "Yes",
            "Yes",
            "Yes",
            "No",
            "Yes",
            "No",
            "Yes",
            None,
        ],
    })

    original = df.copy(deep=True)
    result = summarize_dataset(df)

    assert len(result.categorical_comparisons) == 1

    comparison = result.categorical_comparisons[0]

    assert comparison.group_column == "Contract"
    assert comparison.comparison_column == "Churn"

    assert comparison.usable_rows == 6
    assert comparison.excluded_rows == 2

    assert comparison.group_counts == [4, 2]

    assert comparison.counts == [
        [3, 1],
        [1, 1],
    ]

    assert comparison.percentages == [
        [75.0, 25.0],
        [50.0, 50.0],
    ]

    pd.testing.assert_frame_equal(
        df,
        original,
    )

    loaded = EDAResult.model_validate_json(
        result.model_dump_json()
    )

    assert loaded == result

    with tempfile.TemporaryDirectory() as folder:
        plot = save_categorical_comparison_chart(
            comparison=loaded.categorical_comparisons[0],
            run_dir=folder,
        )

        path = Path(folder) / plot.path

        assert path.read_bytes().startswith(
            b"\x89PNG\r\n\x1a\n"
        )

    print("Categorical comparison checks passed.")
    print(comparison.model_dump_json(indent=2))


if __name__ == "__main__":
    main()