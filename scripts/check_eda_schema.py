from shared.eda_schemas import EDAResult


def main():
    result = EDAResult(
        row_count=4,
        column_count=4,
        numeric_columns=["Age", "MonthlyCharges"],
        categorical_columns=["Contract", "Churn"],
        missing_counts={
            "Age": 1,
            "MonthlyCharges": 0,
            "Contract": 0,
            "Churn": 0,
        },
    )

    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()