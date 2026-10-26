import pandas as pd

from agents.eda_agent.eda_tools import (
    summarize_dataset,
)
from shared.eda_schemas import EDAResult


def main() -> None:
    df = pd.DataFrame({
        "Age": [
            20, None, 40, 20, None
        ],
        "Contract": [
            "Monthly",
            "Yearly",
            "Monthly",
            "Monthly",
            None,
        ],
        "Region": [
            "North",
            "North",
            "North",
            "North",
            "North",
        ],
        "Unused": [
            None, None, None, None, None
        ],
    })

    original = df.copy(deep=True)

    result = summarize_dataset(df)
    overview = result.dataset_overview

    assert overview is not None

    assert overview.total_cells == 20
    assert overview.missing_cells == 8
    assert overview.missing_cell_percentage == 40.0

    assert overview.rows_with_missing == 5
    assert overview.complete_rows == 0

    assert overview.duplicate_rows == 1
    assert overview.duplicate_row_percentage == 20.0

    assert overview.empty_columns == ["Unused"]
    assert overview.constant_columns == ["Region"]

    assert overview.unique_counts == {
        "Age": 2,
        "Contract": 2,
        "Region": 1,
        "Unused": 0,
    }

    pd.testing.assert_frame_equal(
        df,
        original,
    )

    loaded = EDAResult.model_validate_json(
        result.model_dump_json()
    )

    assert loaded == result

    complete_result = summarize_dataset(
        pd.DataFrame({
            "value": [1, 2],
            "group": ["A", "B"],
        })
    )

    complete = complete_result.dataset_overview

    assert complete is not None
    assert complete.complete_rows == 2
    assert complete.rows_with_missing == 0
    assert complete.missing_cells == 0

    print("Dataset overview checks passed.")
    print(overview.model_dump_json(indent=2))


if __name__ == "__main__":
    main()