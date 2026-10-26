# Data Preparation Agent

## Overview

The Data Preparation Agent is responsible for inspecting, preparing and validating a dataset before it is passed to subsequent agents in the CloudFlow pipeline.

It combines Python-based data profiling, predefined operations and LLM-driven decision-making to determine which preparation steps are appropriate for the dataset and the user's analytical objective.

The agent does not automatically apply every possible cleaning operation. Instead, it identifies potential data-quality issues and selects relevant operations based on the dataset's characteristics and the requirements of the analysis.

## Responsibilities

- **Data Inspection:** Identify missing values, duplicate rows, data types and values that violate specified validation rules.
- **Column Identification:** Identify potential target, identifier and relevant columns based on the dataset profile and the user's question.
- **Data Transformation:** Handle missing values, remove duplicate rows, convert data types and handle invalid values when required.
- **Validation:** Verify the dataset after preparation and collect updated data-quality information.
- **Reporting:** Record the preparation plan, operations performed and their results for use by subsequent agents.

## How the Data Prep Agent Works

The Data Prep Agent uses an LLM to plan data preparation operations based on the dataset profile and the user's objective.

Its workflow is:

```text
Load dataset
     |
     v
Profile dataset
     |
     v
Generate and validate preparation plan
     |
     v
Generate Python code
     |
     v
Execute code in a Jupyter notebook
     |
     v
Save cleaned dataset and preparation artifacts
```

The agent separates its preparation process into distinct steps:

1. **Profile the dataset:** Generate an initial profile containing information such as the number of rows and columns, data types, missing values, unique values and duplicate rows.
2. **Determine preparation requirements:** Use the dataset profile and the user's question to determine which preparation steps are relevant.
3. **Generate and validate a plan:** Use the LLM to select appropriate operations from a predefined operation catalogue. Validate the generated plan before executing it.
4. **Generate Python code:** Generate code for the selected operations.
5. **Execute the preparation steps:** Run the generated code in a Jupyter notebook and collect the actual execution results.
6. **Validate and report:** Record the results of the preparation operations and the final dataset's characteristics.
7. **Save artifacts:** Store the cleaned dataset, preparation plan, initial profile, notebook artifacts and preparation report in the run's workspace.

The agent keeps its generated code, computed results and preparation report as separate artifacts so that the processing steps can be inspected.

## Operation Categories

The agent organizes its preparation operations into three categories:

| Category | Purpose |
|---|---|
| Inspection | Examine data quality without modifying the dataset. |
| Transformation | Modify the dataset to address identified issues. |
| Validation | Check the dataset's structure and data quality after preparation. |

Examples of preparation tasks include inspecting missing values, detecting duplicate rows, examining data types, handling missing values, converting column types and checking the dataset after transformations.

The agent selects operations according to the dataset's characteristics and the user's objective rather than applying every available operation.

## Outputs

The Data Preparation Agent produces artifacts that describe both the prepared dataset and the process used to prepare it.

Typical outputs include:

- **Initial dataset profile:** Information about the original dataset's structure and data quality.
- **Preparation plan:** The selected operations and the reasoning behind the preparation decisions.
- **Prepared dataset:** The dataset after the selected transformations have been executed.
- **Preparation report:** A record of the preparation results, including relevant before-and-after dataset information.
- **Notebook artifacts:** Generated code and execution outputs that help make the preparation process inspectable.
- **Container status:** A status record indicating whether the agent completed successfully or encountered an error.

Artifacts are saved under the corresponding run's workspace directory.

For example:

```text
workspace/
└── <run_id>/
    ├── input.csv
    ├── container_status.json
    └── data-prep/
        ├── initial_profile.json
        ├── operation_plan.json
        ├── preparation_report.json
        └── ... additional generated artifacts
```

The exact set of artifacts may vary depending on the operations executed.

These outputs are intended to provide subsequent agents with a prepared dataset and information about the decisions and transformations made during preparation.

## Design Principles

- **Question-aware preparation:** Preparation decisions should reflect the user's analytical objective.
- **Controlled transformations:** Only selected and supported operations should be executed.
- **Validation:** Changes should be checked rather than assumed to be correct.
- **Traceability:** Preparation decisions, operations and results should be recorded.
- **Separation of responsibilities:** Python handles deterministic profiling, transformations, and validation, while the LLM helps interpret analytical requirements and select appropriate operations.
- **Reproducibility:** Generated notebooks and structured artifacts help users inspect the preparation workflow and its results.

## Running the Agent

The Data Prep Agent is designed to run inside its own Docker container, launched by the CloudFlow FastAPI backend.

To build the Docker image, run the following command from the CloudFlow repository root:

```bash
docker build \
  -f agents/data_prep_agent/Dockerfile \
  -t cloudflow-data-prep:latest \
  .
```

The FastAPI backend launches the container when a pipeline run is submitted through the API. The backend passes the run ID, analytical objective, workspace configuration, and LLM environment variables to the container.

For the complete setup and instructions to run the backend and frontend, see the root [`README.md`](../../README.md).

## Sample Input

**Dataset:** [Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

Download the dataset from Kaggle and use the `WA_Fn-UseC_-Telco-Customer-Churn.csv` file, renaming it to `churn.csv` if needed.

**Objective:**

> Analyze customer churn by identifying factors associated with Churn and cleaning the dataset for exploratory data analysis and machine learning.