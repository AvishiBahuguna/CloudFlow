import numpy as np
import pandas as pd

from agents.eda_agent.eda_tools import summarize_dataset
from shared.eda_schemas import EDAResult


def main():
    df = pd.DataFrame({
        "Contract": [
            "Monthly",
            "Monthly",
            "Yearly",
            "Yearly",
            None,
            "Monthly",
            "Yearly",
        ],
        "Charges": [
            100,
            200,
            300,
            500,
            900,
            np.inf,
            None,
        ],
    })

    original = df.copy(deep=True)
    result = summarize_dataset(df)

    assert len(result.group_comparisons) == 1

    comparison = result.group_comparisons[0]

    groups = {
        group.value: group
        for group in comparison.groups
    }

    assert groups["Monthly"].mean == 150.0
    assert groups["Monthly"].median == 150.0
    assert groups["Monthly"].count == 2

    assert groups["Yearly"].mean == 400.0
    assert groups["Yearly"].median == 400.0
    assert groups["Yearly"].count == 2

    assert groups["Monthly"].minimum == 100.0
    assert groups["Yearly"].maximum == 500.0

    assert comparison.usable_rows == 4
    assert comparison.excluded_rows == 3

    pd.testing.assert_frame_equal(df, original)

    reloaded = EDAResult.model_validate_json(
        result.model_dump_json()
    )

    assert reloaded == result

    print("Group comparison checks passed.")
    print(comparison.model_dump_json(indent=2))


if __name__ == "__main__":
    main()