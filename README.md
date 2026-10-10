# CloudFlow

CloudFlow is a planned multi-container, question-driven data analysis
pipeline. A user supplies a dataset and a natural-language question, the
system analyzes the data, returns an evidence-backed answer and
generates a report describing the workflow, decisions, findings, and
limitations.

> **Project status:** Early prototype / architecture and Data Prep
> implementation in progress. The full four-stage pipeline is not yet
> implemented.

## Goals

Given an input such as:

-   **Dataset:** `churn.csv`
-   **Question:** "Which factors are associated with customer churn, and
    can we predict which customers are likely to leave?"

CloudFlow should:

1.  Inspect and prepare the dataset.
2.  Perform exploratory data analysis (EDA) relevant to the question.
3.  Engineer features and train/evaluate a model when modeling is
    appropriate.
4.  Validate the results, answer the question, and produce a report.
5.  Preserve notebooks, results, and artifacts so the analysis can be
    inspected and reproduced.

The pipeline should not force every question through machine learning. A
descriptive question may only need a verified calculation or EDA.

## Proposed architecture

``` text
User: dataset + question
          |
          v
   Orchestrator / Planner
   - creates a run ID
   - plans relevant stages
   - tracks state and errors
   - passes artifact references
          |
          v
   1. Data Prep Agent
          |
          v
   2. EDA Agent
          |
          v
   3. Feature Engineering & ML Agent
          |
          v
   4. Evaluation & Report Agent
          |
          v
   Final answer + report + supporting artifacts
```

Each agent is planned as a separate Docker container exposing a small
API (for example, with FastAPI). The orchestrator coordinates the
services; it should not do the agents' analysis itself.

### Agent responsibilities

  -----------------------------------------------------------------------
  Component               Responsibilities        Expected outputs
  ----------------------- ----------------------- -----------------------
  Orchestrator / Planner  Accept input, create a  Pipeline state and run
                          run, choose required    summary
                          stages, track status,   
                          handle                  
                          timeouts/retries,       
                          validate agent          
                          responses               

  1\. Data Prep           Profile columns, types, Cleaned dataset, data
                          missing values,         profile, cleaning
                          duplicates and invalid  summary, executed
                          values; apply justified notebook
                          cleaning; identify      
                          possible target columns 

  2\. EDA                 Run question-relevant   Findings JSON, charts,
                          statistics and          executed notebook
                          visualizations; record  
                          observations and        
                          caveats                 

  3\. Feature Engineering Decide whether modeling Model artifact when
  & ML                    is needed; engineer     applicable,
                          features; establish a   predictions, experiment
                          baseline; train and     results, executed
                          compare suitable models notebook

  4\. Evaluation & Report Verify claims against   Final answer, report,
                          computed evidence;      evidence/claim
                          assess evaluation       references
                          quality; answer the     
                          question; summarize     
                          decisions and           
                          limitations             
  -----------------------------------------------------------------------

The exact plan should depend on the question and dataset. The target
column, transformations, and metrics must be validated rather than
guessed silently.

## Proposed end-to-end workflow

1.  **Receive input:** Store the original dataset without modifying it,
    record the question, and create a unique `run_id`.
2.  **Plan:** Inspect available metadata and select the stages needed to
    answer the question.
3.  **Run Data Prep:** Profile and validate the data, make documented
    cleaning decisions, execute the work, and save outputs.
4.  **Run EDA:** Read the prepared dataset and question, execute
    relevant analyses, and save findings and charts.
5.  **Run Feature Engineering & ML when needed:** Use an appropriate
    split and preprocessing pipeline, check for leakage, train a
    baseline and evaluate on held-out data.
6.  **Evaluate and answer:** Check that important numerical claims match
    the artifacts. If evidence is insufficient, state the limitation
    rather than inventing an answer.
7.  **Write report:** Produce a human-readable report plus structured
    machine-readable results.
8.  **Return results:** Provide the answer, report location, and
    links/paths to supporting artifacts.

## LLM-generated code and notebooks

The planned workflow is for each analysis agent to use an LLM to choose
analyses and generate Python code, then execute that code in a Jupyter
notebook.

``` text
Agent receives question + input artifact references
                     |
                     v
              LLM plans analysis
                     |
                     v
          Generate notebook code cells
                     |
                     v
             Execute the notebook
                     |
                     v
         Capture outputs, errors, metrics
                     |
                     v
         LLM interprets actual results
                     |
                     v
        Save notebook + structured artifacts
```

Recommended components:

-   **Notebook format:** Jupyter `.ipynb`
-   **Notebook execution:** `nbclient` and `nbformat`
-   **Data analysis:** Pandas, NumPy, SciPy
-   **Visualization:** Matplotlib and/or Seaborn
-   **Machine learning:** scikit-learn
-   **Agent APIs:** FastAPI
-   **Orchestration:** Python service (proposed)
-   **Initial artifact exchange:** Shared Docker volume
-   **Report:** Markdown first; PDF can be added later

A notebook is an execution record, not proof that its conclusions are
correct. Numerical results should be stored in structured artifacts and
checked before they appear in the final answer.

### Execution security

LLM-generated Python is arbitrary code. Do not execute it with
unrestricted access to the host, credentials, Docker socket or network.
Use a restricted execution worker with resource limits and only the
files and permissions it needs. A Docker container provides isolation,
but it is not automatically a complete security sandbox.

## Artifact layout (proposed)

``` text
workspace/
└── <run_id>/
    ├── input/
    │   └── original_dataset.csv
    └── artifacts/
        ├── data-prep/
        │   ├── cleaned_data.parquet
        │   ├── data_profile.json
        │   ├── cleaning_summary.json
        │   └── data_prep.ipynb
        ├── eda/
        │   ├── findings.json
        │   ├── eda.ipynb
        │   └── charts/
        ├── ml/
        │   ├── experiment_results.json
        │   ├── predictions.csv
        │   ├── model.joblib
        │   └── ml.ipynb
        └── final/
            ├── answer.json
            ├── report.json
            ├── report.md
            └── report.pdf
```

These are intended outputs, not all implemented yet. Only create model
artifacts when modeling is appropriate. Keep the original dataset
immutable and have each stage write to its own artifact directory.

## Evidence-backed answers and reports

Keep these separate:

1.  **Executed code:** the notebook cells that actually ran.
2.  **Computed evidence:** structured outputs, statistics, model
    metrics, and chart files.
3.  **Interpretation:** the LLM's explanation of what the results mean.

The report should cover:

-   Executive summary and original question
-   Dataset overview and data quality
-   Cleaning decisions and their rationale
-   EDA methods, findings, and visualizations
-   Feature engineering and model choices, if applicable
-   Evaluation methodology and measured results
-   Final answer
-   Assumptions, limitations, and reproducibility details

Important claims should reference their supporting result artifact.
Observational associations should not be presented as causal
conclusions.

## Suggested project structure

This is a target structure and may need to be aligned with the
repository's current layout.

``` text
CloudFlow/
├── docker-compose.yml
├── .env
├── orchestrator/
│   ├── Dockerfile
│   ├── main.py
│   ├── pipeline.py
│   ├── planner.py
│   └── state_manager.py
├── agents/
│   ├── data-prep-agent/
│   │   ├── Dockerfile
│   │   ├── main.py
│   │   ├── profiler.py
│   │   ├── cleaner.py
│   │   └── notebook_runner.py
│   ├── eda-agent/
│   ├── ml-agent/
│   └── evaluation-agent/
├── shared/
│   ├── state.py
│   ├── schemas.py
│   └── artifact_utils.py
└── workspace/
```

## Current status

### Done / confirmed in the current development work

-   [x] Defined the intended four-stage pipeline and the orchestrator's
    role.

-   [x] Started a Data Prep profiler in
    `agents/data-prep-agent/profiler.py`.

-   [x] The profiler is intended to produce a `ColumnProfile` for each
    DataFrame column, including its name, dtype, missing-value count,
    unique-value count, and sample values.

-   [x] Run the profiler successfully end to end against
    `data/churn.csv`.

### In progress / not yet verified as complete

-   [ ] Define and validate shared request/response schemas for agents.
-   [ ] Implement the Data Prep API and actual cleaning/artifact
    outputs.
-   [ ] Add controlled notebook generation and execution.
-   [ ] Implement EDA, Feature Engineering & ML, and Evaluation & Report
    agents.
-   [ ] Implement the orchestrator's stage management, conditional
    execution, and error handling.
-   [ ] Configure Docker Compose and shared per-run artifact storage.
-   [ ] Add evidence validation and final report generation.
-   [ ] Add restricted execution for LLM-generated code.

**Status note:** The items above reflect what has been
attempted in the current development context. They are not a claim that
the code has been tested successfully or that the full pipeline is
running.
