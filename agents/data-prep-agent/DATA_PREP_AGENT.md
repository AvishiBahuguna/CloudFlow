# DATA PREPARATION AGENT
## Overview

The Data Preparation Agent is responsible for inspecting, preparing and validating the dataset before it is passed to the subsequent agents in the CloudFlow pipeline.

It uses a combination of Python-based data profiling, predefined operations and LLM-driven decision-making to determine which preparation steps are appropriate for the dataset and the user's question.

The agent does not automatically apply every possible cleaning operation. Instead, it identifies potential data-quality issues and selects relevant operations based on the dataset's characteristics and the requirements of the analysis.

## Responsibilities

- **Data Inspection:** Identify missing values, duplicate rows, data types and values that violate specified validation rules.
- **Column Identification:** Suggest potential target columns and identifier columns based on dataset characteristics.
- **Data Transformation:** Handle missing values, remove duplicate rows, convert data types and handle invalid values when required.
- **Validation:** Verify the dataset after preparation and collect updated data-quality information.
- **Reporting:** Record the operations performed and their results for use by subsequent agents.

## How It Works

1. **Profile the dataset:** Generate an initial profile containing information such as the number of rows and columns, data types, missing values, unique values and duplicate rows.
2. **Determine preparation requirements:** Use the dataset profile and the user's question to identify relevant preparation steps.
3. **Select operations:** Choose appropriate operations from a predefined operation catalogue.
4. **Execute operations:** Apply the selected transformations to the dataset while recording their results.
5. **Validate the output:** Check the prepared dataset to ensure that the operations produced the expected changes.
6. **Save artifacts:** Store the prepared dataset, updated profile and preparation results for use by the next agent.

## Operation Categories

The agent organizes its operations into four categories:

| Category | Purpose |
|---|---|
| Inspection | Examine data quality without modifying the dataset. |
| Identification | Suggest potential target and identifier columns. |
| Transformation | Modify the dataset to address identified issues. |
| Validation | Check the dataset's structure and data quality after preparation. |

## Outputs

The Data Preparation Agent produces:

- An initial dataset profile.
- A prepared dataset.
- An updated dataset profile.
- A record of the operations performed and their results.

These outputs are passed to subsequent agents in the CloudFlow pipeline, providing them with a consistent dataset and information about the preparation process.

## Design Principles

- **Question-aware preparation:** Preparation decisions should reflect the user's analytical objective.
- **Controlled transformations:** Only selected and supported operations should be executed.
- **Validation:** Changes should be checked rather than assumed to be correct.
- **Traceability:** Operations and their results should be recorded.
- **Separation of responsibilities:** Python handles deterministic profiling, transformations and validation, while the LLM helps interpret the analytical requirements and select appropriate operations.