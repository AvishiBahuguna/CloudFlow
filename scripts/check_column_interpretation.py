import pandas as pd

from agents.eda_agent.eda_tools import (
    summarize_dataset,
)
from shared.eda_schemas import EDAResult


def main() -> None:
    df = pd.DataFrame({
        "CustomerID": [
            "C001", "C002", "C003", "C004"
        ],
        "Age": [
            20, 30, None, 40
        ],
        "Contract": [
            "Monthly", "Yearly", "Monthly", None
        ],
        "JoinedOn": [
            "2026-01-01",
            "2026-01-02",
            "2026-01-03",
            "2026-01-04",
        ],
        "Active": pd.Series(
            [True, False, None, True],
            dtype="boolean",
        ),
        "Feedback": [
            (
                "This customer left a detailed written "
                "description of their experience with "
                "the service and its support team."
            )
        ] * 4,
        "Unused": [
            None, None, None, None
        ],
        "RiskCode": [
            0, 1, 0, 1
        ],
    })

    original = df.copy(deep=True)

    result = summarize_dataset(df)

    actual_roles = {
        column: info.role
        for column, info in (
            result.column_interpretations.items()
        )
    }

    assert actual_roles == {
        "CustomerID": "identifier",
        "Age": "numeric",
        "Contract": "categorical",
        "JoinedOn": "datetime",
        "Active": "boolean",
        "Feedback": "text",
        "Unused": "unknown",
        "RiskCode": "numeric",
    }

    corrected = summarize_dataset(
        df,
        role_overrides={
            "RiskCode": "categorical",
        },
    )

    assert "RiskCode" not in corrected.numeric_columns

    assert (
        "RiskCode"
        in corrected.categorical_columns
    )

    assert (
        corrected.categorical_summaries[
            "RiskCode"
        ].unique_count
        == 2
    )

    assert (
        corrected.column_interpretations[
            "RiskCode"
        ].source
        == "override"
    )

    loaded = EDAResult.model_validate_json(
        corrected.model_dump_json()
    )

    assert loaded == corrected

    pd.testing.assert_frame_equal(
        df,
        original,
    )

    print("Column interpretation checks passed.")

    for column, info in (
        result.column_interpretations.items()
    ):
        print(f"{column}: {info.role}")


if __name__ == "__main__":
    main()