# Exploratory Data Analysis (EDA) Agent

## Overview

The EDA Agent explores the prepared dataset to understand its structure,
distributions, relationships and patterns.

Its primary goal is to discover useful insights that help answer the
user's question. It should not simply generate a generic collection of
statistics and plots. Instead, the analysis should be guided by the
user's analytical objective.

The agent receives the prepared dataset from the Data Preparation Agent
and produces structured findings that can be used by the ML Agent and
the Evaluation & Report Writer Agent.

## Responsibilities

- Understand the dataset's structure and available columns.
- Generate descriptive statistics for relevant columns.
- Analyze distributions of numerical and categorical variables.
- Identify correlations and relationships between variables.
- Investigate patterns relevant to the user's question.
- Generate useful visualizations.
- Identify potential outliers and unusual patterns.
- Record important findings and supporting evidence.
- Save all outputs for downstream agents.

## Workflow

### Step 1: Receive Inputs

The agent receives the following information from the orchestrator:

- Run ID.
- User's question or problem statement.
- Path to the prepared dataset.
- Path to the final data preparation profile.
- Relevant metadata from the Data Preparation Agent.

The agent should use the prepared dataset rather than independently
cleaning the original dataset.

### Step 2: Understand the Analytical Objective

Examine the user's question and determine what needs to be investigated.

For example, if the question is:

"Which factors are associated with customer churn?"

The agent should prioritize:
- The distribution of the churn variable.
- Relationships between churn and relevant customer attributes.
- Differences between customers who churn and those who do not.
- Potentially useful patterns for subsequent predictive modelling.

The agent should not assume that every question requires machine learning.

### Step 3: Analyze the Dataset

Perform appropriate analyses based on the question and dataset.

Possible analyses include:

- Summary statistics for numerical columns.
- Frequency distributions for categorical columns.
- Histograms and box plots.
- Correlation analysis.
- Grouped comparisons.
- Relationships between selected variables.
- Analysis of class imbalance when a classification target exists.
- Identification of unusual distributions and potential outliers.

The agent should select analyses based on their relevance rather than
blindly applying every available technique.

### Step 4: Generate Visualizations

Create plots that help explain important findings.

Each visualization should have:
- A descriptive title.
- Clearly labelled axes.
- Appropriate scales.
- A short explanation of what it demonstrates.

Save plots as image files so they can be included in the final report.

### Step 5: Interpret Findings

For each important finding, record:

- What was observed.
- The relevant columns or variables.
- The supporting statistics or computed evidence.
- Why the observation matters for the user's question.
- Any limitations or uncertainty.

Distinguish observed relationships from causal explanations. Correlation
alone does not establish causation.

### Step 6: Save Outputs

Save the analysis results in a structured format for downstream agents.

Recommended outputs:

- `eda_report.json`: Structured findings, statistics, and observations.
- `eda_analysis.ipynb`: Notebook containing the analysis code and results.
- `plots/`: Directory containing generated visualizations.

## Expected Outputs

The EDA Agent should provide:

1. A summary of the dataset and its relevant characteristics.
2. Important patterns and relationships.
3. Supporting statistics for each major finding.
4. Paths to generated visualizations.
5. Observations that may be useful for machine learning.
6. Limitations or issues discovered during the analysis.

## Suggested Output Structure

The `eda_report.json` file may follow this structure:

{
    "run_id": "run_001",
    "summary": "Summary of the analysis",
    "findings": [
        {
            "title": "Finding title",
            "observation": "Description of the observation",
            "columns": ["Churn", "Contract"],
            "evidence": {
                "metric": "Example computed statistic",
                "value": 0.0
            },
            "relevance": "Why this matters for the question"
        }
    ],
    "visualizations": [
        {
            "title": "Visualization title",
            "path": "plots/example.png",
            "description": "What the plot demonstrates"
        }
    ],
    "limitations": []
}

The values above are illustrative. Actual values must come from the
analysis.

## Implementation Guidelines

- Use Python and Pandas for data analysis.
- Use Matplotlib or Seaborn for visualizations.
- Use Jupyter notebooks to organize executable analysis code and results.
- Save computed findings as structured JSON rather than relying only
  on notebook output.
- Use relative artifact paths within the current pipeline run.
- Do not modify the prepared dataset.
- Do not invent statistics or observations.
- Make the analysis reproducible.
- Handle unsupported analyses and insufficient data gracefully.

## Integration with CloudFlow

The EDA Agent receives the prepared dataset and relevant metadata from
the orchestrator.

Its findings should be made available to:
- The ML Agent, which can use relevant observations to guide modelling.
- The Evaluation & Report Writer Agent, which uses the findings and
  their supporting evidence to construct the final report.

The agent should return its output artifact paths and execution status
to the orchestrator.