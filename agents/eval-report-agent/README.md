# Evaluation & Report Writer Agent

## Overview

The Evaluation & Report Writer Agent is responsible for bringing
together the results produced by the other CloudFlow agents and
generating a coherent, evidence-based answer to the user's question.

It receives the original question, the dataset preparation results,
EDA findings, and ML evaluation results when available.

Its role is not simply to summarize every intermediate output.
It must determine which findings are relevant to the question,
check whether the available evidence supports the conclusions, and
produce a final report explaining the analysis and its results.

The agent should distinguish between computed facts, interpretations,
and limitations. It must not invent results or claim that an analysis
was performed when no supporting evidence exists.

## Responsibilities

- Collect and interpret outputs from upstream agents.
- Check that important claims are supported by computed results.
- Determine which findings are relevant to the user's question.
- Evaluate the reliability and limitations of the available evidence.
- Reconcile inconsistent or conflicting findings where possible.
- Generate a direct answer to the user's question.
- Summarize the pipeline's analysis and reasoning.
- Explain important observations and their significance.
- Document limitations, assumptions, and unsuccessful steps.
- Produce the final report in a structured format.

## Workflow

### Step 1: Receive Inputs

The agent receives the following information from the orchestrator:

- Run ID.
- Original user question.
- Data Preparation Agent results.
- EDA Agent report.
- ML Agent report, if applicable.
- Paths to relevant plots, metrics, and other artifacts.
- Pipeline execution status.

The agent should use the actual outputs generated during the current
run rather than relying on assumptions about what the other agents
were supposed to produce.

### Step 2: Collect the Evidence

Gather the findings needed to answer the user's question.

These may include:

- Dataset characteristics and preparation steps.
- Descriptive statistics.
- Patterns and relationships discovered by EDA.
- Model evaluation metrics.
- Predictions or other computed results.
- Visualizations that support important findings.
- Limitations reported by the upstream agents.

Only use outputs that actually exist and can be associated with the
current pipeline run.

### Step 3: Validate Important Findings

Before using a finding in the final report, examine whether it is
supported by the available evidence.

Check that:
- Numerical claims have corresponding computed values.
- The findings relate to the user's question.
- Model metrics are interpreted in the context of the evaluation
  strategy.
- Conclusions do not exceed what the evidence supports.
- Missing or failed upstream outputs are acknowledged.
- Contradictory findings are identified rather than silently ignored.

If an important result cannot be verified, qualify or omit the claim
and explain the limitation when relevant.

### Step 4: Construct the Answer

Begin by directly answering the user's original question.

The answer should:
- State the main conclusion.
- Include the most relevant supporting evidence.
- Explain important observations in understandable language.
- Include model results when modelling was performed and is relevant.
- Explain uncertainty or limitations that affect the conclusion.

The answer should be useful on its own, even if the user does not read
the complete report.

If the available evidence cannot answer the question, state this
clearly and explain what additional information or analysis is needed.

### Step 5: Write the Analysis Report

The report should explain what CloudFlow did and what it discovered.

Recommended sections:

1. Executive Summary
   - The original question.
   - The main findings.
   - The answer or conclusion.

2. Dataset Overview
   - Dataset dimensions and relevant characteristics.
   - Important data-quality issues.

3. Data Preparation
   - Cleaning and transformation steps performed.
   - Important changes to the dataset.
   - Validation results.

4. Exploratory Data Analysis
   - Relevant statistics and patterns.
   - Important relationships.
   - Supporting visualizations.

5. Machine Learning
   - Include this section only when modelling was performed.
   - Describe the problem type, target, and features.
   - Summarize the training and evaluation approach.
   - Present the relevant metrics and model limitations.

6. Final Findings
   - Answer the user's question.
   - Explain the supporting evidence.
   - Distinguish direct observations from interpretations.

7. Limitations
   - Data limitations.
   - Analytical assumptions.
   - Model limitations, if applicable.
   - Missing or failed analyses.
   - Areas where further investigation is needed.

### Step 6: Generate the Final Artifacts

Save the final answer and report in machine-readable and
human-readable formats.

Recommended outputs:

- `final_report.json`: Structured answer, findings, and report sections.
- `final_report.md`: Readable Markdown report.
- References to plots, metrics, and other supporting artifacts.

The orchestrator or frontend can display the final answer directly
and provide access to the full report.

## Expected Outputs

The Evaluation & Report Writer Agent should produce:

1. A direct answer to the original question.
2. A report summarizing the pipeline's work.
3. Supporting evidence for important conclusions.
4. References to relevant visualizations and model results.
5. An explanation of important limitations.
6. A summary of pipeline steps that succeeded or failed.

## Suggested Output Structure

The `final_report.json` file may follow this structure:

{
    "run_id": "run_001",
    "question": "The original user question",
    "answer": "A direct, evidence-based answer",
    "executive_summary": "Summary of the main findings",
    "sections": [
        {
            "title": "Data Preparation",
            "content": "Summary of the preparation process",
            "evidence": []
        },
        {
            "title": "Exploratory Data Analysis",
            "content": "Summary of relevant findings",
            "evidence": []
        }
    ],
    "key_findings": [
        {
            "finding": "Description of a finding",
            "evidence": [
                "Reference to a computed statistic or artifact"
            ],
            "limitations": []
        }
    ],
    "limitations": [],
    "pipeline_status": "completed"
}

The example is illustrative. Actual findings and values must come from
the current pipeline run.

## Implementation Guidelines

- Use structured outputs from upstream agents wherever possible.
- Keep the original user question available throughout report generation.
- Prioritize findings that answer the question.
- Do not invent statistics, plots, metrics, or model results.
- Do not claim that a model was trained if the ML Agent did not train one.
- Distinguish correlation from causation.
- Explain technical results in accessible language.
- Preserve links or relative paths to supporting artifacts.
- Record incomplete or failed analyses.
- Do not silently replace missing results with fabricated values.
- Ensure the report corresponds to the correct pipeline run.

## Integration with CloudFlow

The Evaluation & Report Writer Agent is the final analytical stage
of the CloudFlow pipeline.

It receives the outputs of the Data Preparation, EDA, and ML agents
through the orchestrator.

Its final answer and report are returned to the orchestrator, which
makes them available to the user through the frontend.

The agent should return its output artifact paths and execution status
to the orchestrator.