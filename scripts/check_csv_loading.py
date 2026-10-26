from io import BytesIO

import pandas as pd

from agents.eda_agent.csv_tools import (
    load_csv_dataset,
)
from shared.eda_schemas import CSVReadLimits


def expect_error(
    text: str,
    expected_message: str,
    **kwargs,
) -> None:
    try:
        load_csv_dataset(
            BytesIO(text.encode("utf-8")),
            **kwargs,
        )

    except ValueError as error:
        assert expected_message in str(error)
        print(f"Correctly rejected: {error}")

    else:
        raise AssertionError(
            "Expected the CSV to be rejected."
        )


def main() -> None:
    valid_csv = (
        "Age,Contract,Notes\n"
        '20,Monthly,"Fast, useful"\n'
        ',Yearly,"Works well"\n'
    )

    df = load_csv_dataset(
        BytesIO(valid_csv.encode("utf-8"))
    )

    assert df.shape == (2, 3)
    assert df.loc[0, "Notes"] == "Fast, useful"
    assert pd.isna(df.loc[1, "Age"])

    expect_error(
        "Age,Age\n20,30\n",
        "Duplicate",
    )

    expect_error(
        "Age,\n20,Monthly\n",
        "non-empty",
    )

    expect_error(
        "Age,Contract\n20,Monthly,Extra\n",
        "expected 2",
    )

    expect_error(
        "Age,Contract\n20\n",
        "expected 2",
    )

    expect_error(
        'Age,Contract\n20,"Monthly\n',
        "formatting",
    )

    expect_error(
        "",
        "empty",
    )

    expect_error(
        "Age,Contract\n",
        "at least one data row",
    )

    expect_error(
        "Age\n20\n30\n",
        "row limit",
        limits=CSVReadLimits(max_rows=1),
    )

    expect_error(
        "Age,Contract\n20,Monthly\n",
        "column limit",
        limits=CSVReadLimits(max_columns=1),
    )

    expect_error(
        "Age,Contract\n20,Monthly\n",
        "cell limit",
        limits=CSVReadLimits(max_cells=1),
    )

    expect_error(
        "Age\n20\n",
        "byte limit",
        limits=CSVReadLimits(max_bytes=5),
    )

    print("\nCSV loading checks passed.")


if __name__ == "__main__":
    main()