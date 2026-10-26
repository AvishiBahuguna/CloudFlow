import pandas as pd

from agents.eda_agent.eda_tools import (
    summarize_dataset,
)
from shared.eda_schemas import EDAResult


def main() -> None:
    df = pd.DataFrame({
        "CustomerID": [
            "C001", "C002", "C003", "C004", "C005"
        ],
        "Hours": [
            1, 2, 3, 4, 5
        ],
        "Score": [
            10, 20, 30, 40, 50
        ],
        "Errors": [
            5, 4, 3, 2, 1
        ],
        "PartialScore": [
            10, 20, None, 40, 50
        ],
        "Constant": [
            7, 7, 7, 7, 7
        ],
        "Sparse": [
            None, None, None, 10, 20
        ],
    })

    original = df.copy(deep=True)

    result = summarize_dataset(df)

    summary = result.correlation_summary

    assert summary is not None

    assert abs(
        summary.matrix["Hours"]["Score"] - 1
    ) < 1e-10

    assert abs(
        summary.matrix["Hours"]["Errors"] + 1
    ) < 1e-10

    assert (
        summary.pair_counts["Hours"]["PartialScore"]
        == 4
    )

    assert "Constant" in summary.excluded_columns
    assert "Sparse" in summary.excluded_columns
    assert "CustomerID" not in summary.columns

    loaded = EDAResult.model_validate_json(
        result.model_dump_json()
    )

    assert loaded == result

    pd.testing.assert_frame_equal(
        df,
        original,
    )

    print("Correlation checks passed.")
    print(summary.model_dump_json(indent=2))


if __name__ == "__main__":
    main()