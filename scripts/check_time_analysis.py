import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from agents.eda_agent.eda_tools import (
    summarize_dataset,
)

from agents.eda_agent.plotting_tools import (
    save_time_charts,
)

from shared.eda_schemas import (
    EDAResult,
    TimeConfig,
)


def main():
    df = pd.DataFrame({
        "OrderDate": [
            "2026-01-05",
            "2026-01-20",
            "2026-03-03",
            "2026-03-04",
            "2026-03-05",
            None,
        ],
        "Revenue": [
            100, 200, 300,
            None, np.inf, 900,
        ],
    })

    original = df.copy(deep=True)

    result = summarize_dataset(
        df,
        time_config=TimeConfig(
            date_column="OrderDate",
            frequency="month",
        ),
    )

    analysis = result.time_analysis

    assert analysis is not None
    assert analysis.usable_rows == 5
    assert analysis.missing_count == 1
    assert analysis.invalid_count == 0

    assert analysis.period_starts == [
        "2026-01-01",
        "2026-02-01",
        "2026-03-01",
    ]

    assert analysis.row_counts == [2, 0, 3]

    trend = analysis.numeric_trends["Revenue"]

    assert trend.counts == [2, 0, 1]
    assert trend.means == [150.0, None, 300.0]
    assert trend.usable_rows == 3
    assert trend.excluded_rows == 3

    pd.testing.assert_frame_equal(df, original)

    with tempfile.TemporaryDirectory() as folder:
        result.plots.extend(
            save_time_charts(result, folder)
        )

        assert len(analysis.plot_paths) == 2

        # Existing chart references prevent duplicates.
        assert save_time_charts(result, folder) == []

        for relative_path in analysis.plot_paths:
            assert (
                Path(folder) / relative_path
            ).read_bytes().startswith(
                b"\x89PNG\r\n\x1a\n"
            )

    loaded = EDAResult.model_validate_json(
        result.model_dump_json()
    )

    assert loaded == result

    bad = df.copy()
    bad.loc[0, "OrderDate"] = "01/02/2026"

    checked = summarize_dataset(
        bad,
        time_config=TimeConfig(
            date_column="OrderDate",
        ),
    )

    assert checked.time_analysis.invalid_count == 1

    assert summarize_dataset(df).time_analysis is None

    print("Time analysis checks passed.")


if __name__ == "__main__":
    main()