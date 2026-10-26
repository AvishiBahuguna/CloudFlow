import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from agents.eda_agent.distribution_tools import (
    summarize_numeric_distribution,
)

from agents.eda_agent.eda_tools import summarize_dataset

from agents.eda_agent.plotting_tools import (
    save_numeric_distribution_chart,
)

from shared.eda_schemas import EDAResult


def main():
    df = pd.DataFrame({
        "Value": [
            10, 11, 12, 13, 14,
            15, 16, 100, None, np.inf,
        ]
    })

    original = df.copy(deep=True)

    result = summarize_dataset(df)
    info = result.numeric_distributions["Value"]

    assert info.finite_count == 8
    assert info.missing_count == 1
    assert info.infinite_count == 1

    assert info.q1 == 11.75
    assert info.median == 13.5
    assert info.q3 == 15.25
    assert info.iqr == 3.5

    assert info.lower_fence == 6.5
    assert info.upper_fence == 20.5

    assert info.potential_outlier_count == 1
    assert info.potential_outlier_percentage == 12.5
    assert info.outlier_sample == [100.0]

    assert info.upper_whisker == 16.0
    assert sum(info.histogram_counts) == 8

    pd.testing.assert_frame_equal(df, original)

    constant = summarize_numeric_distribution(
        pd.Series([7] * 8)
    )

    assert constant.potential_outlier_count == 0
    assert constant.histogram_counts == [8]

    small = summarize_numeric_distribution(
        pd.Series([1, 2, 3])
    )

    assert small.potential_outlier_count is None

    empty = summarize_numeric_distribution(
        pd.Series([None, np.inf, -np.inf])
    )

    assert empty.finite_count == 0
    assert empty.histogram_counts == []

    loaded = EDAResult.model_validate_json(
        result.model_dump_json()
    )

    assert loaded == result

    with tempfile.TemporaryDirectory() as folder:
        plot = save_numeric_distribution_chart(
            result=loaded,
            column_name="Value",
            run_dir=folder,
        )

        assert plot is not None

        path = Path(folder) / plot.path

        assert path.read_bytes().startswith(
            b"\x89PNG\r\n\x1a\n"
        )

    print("Numeric distribution checks passed.")
    print(info.model_dump_json(indent=2))


if __name__ == "__main__":
    main()