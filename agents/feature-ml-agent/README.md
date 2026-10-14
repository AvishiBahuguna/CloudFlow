# Feature Engineering & ML Agent

## Overview

The Feature Engineering & ML Agent prepares suitable features and
builds machine learning models when the user's question requires
predictive or other model-based analysis.

It receives the prepared dataset, the user's question, and relevant
metadata from the Data Preparation Agent. It may also use findings
from the EDA Agent to guide feature selection and modelling decisions.

The agent should determine whether machine learning is actually
necessary. Descriptive questions should not automatically trigger
model training.

Its primary goal is to build an appropriate, reproducible model when
required and produce reliable results that can be evaluated and
included in the final report.

## Responsibilities

- Determine whether the user's question requires machine learning.
- Identify the target variable when prediction is required.
- Determine the type of ML problem.
- Select relevant input features.
- Perform appropriate feature engineering.
- Handle model-specific preprocessing.
- Train suitable baseline and candidate models.
- Evaluate model performance using appropriate metrics.
- Compare candidate models.
- Save model artifacts, predictions, metrics, and findings.

## Workflow

### Step 1: Receive Inputs

The agent receives:

- Run ID.
- User's question or problem statement.
- Path to the prepared dataset.
- Dataset profile and relevant preparation results.
- Target-column information, if available.
- EDA findings, if available.

The agent should use the prepared dataset and respect the decisions
already made during data preparation.

### Step 2: Determine Whether ML Is Required

Interpret the user's question to determine whether a model is needed.

Examples:

- "Can we predict whether a customer will churn?"
  Requires predictive modelling.

- "Which customer group has the highest average spending?"
  Can be answered using data analysis and aggregation without
  necessarily training a model.

- "Which factors are associated with customer churn?"
  May be addressed using EDA, statistical analysis, or modelling,
  depending on the intended depth of the analysis.

If machine learning is unnecessary, return a structured result
indicating that no model was trained and explaining why.

Do not train a model merely because the agent is available.

### Step 3: Identify the Target and Problem Type

If prediction is required, identify the target column using the
user's question and the available dataset metadata.

Determine the problem type:

- Classification: Predict a category or class.
- Regression: Predict a numerical value.
- Clustering: Discover groups when the question calls for
  unsupervised grouping.
- Other tasks: Select an appropriate approach only when justified
  by the question.

If the target is ambiguous or cannot be identified reliably, request
clarification through the orchestrator rather than silently selecting
an arbitrary column.

For supervised learning, verify that the target exists in the dataset
and contains sufficient usable data.

### Step 4: Select Features

Determine which columns are suitable as model inputs.

Consider:
- Relevance to the target.
- Data types and missing values.
- High-cardinality categorical variables.
- Identifier columns that are unlikely to generalize.
- Columns that could introduce target leakage.
- Columns that would not be available at prediction time.

Do not automatically use every column as a feature.

The target must not accidentally be included among the input features.

### Step 5: Perform Feature Engineering

Apply transformations appropriate to the selected model and dataset.

Possible operations include:

- Encoding categorical variables.
- Scaling numerical features.
- Creating useful derived features.
- Selecting relevant features.
- Removing unsuitable input columns.
- Applying appropriate transformations to skewed variables.

Feature engineering should be guided by the dataset and the user's
objective, not by a fixed sequence of transformations.

Where preprocessing learns parameters from the data, fit those
transformations using training data only. Use a suitable preprocessing
pipeline to avoid information leakage.

### Step 6: Split the Data

For supervised learning, divide the dataset into appropriate training
and evaluation sets.

Depending on the problem, use:
- A random train/test split.
- A stratified split for classification when appropriate.
- A time-based split for time-dependent prediction.
- Cross-validation when appropriate.

The test set should not be used to select features, tune
hyperparameters, or repeatedly optimize the model.

Choose the splitting strategy based on the problem and dataset.

### Step 7: Train Models

Start with a simple baseline model.

Then, if useful, train and compare additional candidate models.

Examples include:

Classification:
- Logistic Regression.
- Decision Tree.
- Random Forest.

Regression:
- Linear Regression.
- Decision Tree Regressor.
- Random Forest Regressor.

These are examples, not mandatory models. Choose suitable algorithms
based on the problem, data, and computational constraints.

Avoid training many models without a clear reason.

### Step 8: Evaluate Performance

Select metrics appropriate to the problem.

Classification metrics may include:
- Accuracy.
- Precision.
- Recall.
- F1-score.
- Confusion matrix.
- ROC-AUC, where appropriate.

Regression metrics may include:
- Mean Absolute Error (MAE).
- Mean Squared Error (MSE).
- Root Mean Squared Error (RMSE).
- R-squared.

For imbalanced classification problems, do not rely on accuracy alone.

Compare candidate models against the baseline and explain the
trade-offs between their results.

### Step 9: Save Outputs

Save the model results and all relevant artifacts.

Recommended outputs:

- `ml_report.json`: Structured modelling results.
- `ml_analysis.ipynb`: Notebook containing feature engineering,
  training, and evaluation code.
- `model.joblib`: Saved model and preprocessing pipeline, when a
  model is trained and persistence is appropriate.
- `predictions.csv`: Actual and predicted values for the evaluation
  set, where appropriate.
- `metrics.json`: Model evaluation metrics.

Do not create model artifacts when no model was trained.

## Expected Outputs

The ML Agent should provide:

1. Whether machine learning was required.
2. The selected problem type.
3. The target column, if applicable.
4. The features used and important exclusions.
5. The feature engineering and preprocessing steps.
6. The training and evaluation approach.
7. Baseline and candidate model results.
8. The selected model and justification, if applicable.
9. Paths to saved models, predictions, and other artifacts.
10. Limitations and warnings.

## Suggested Output Structure

The `ml_report.json` file may follow this structure:

{
    "run_id": "run_001",
    "ml_required": true,
    "problem_type": "classification",
    "target_column": "Churn",
    "features_used": [
        "Age",
        "MonthlyCharges",
        "Contract"
    ],
    "preprocessing_steps": [
        "Encoded categorical variables"
    ],
    "evaluation": {
        "split_strategy": "stratified_train_test_split",
        "metrics": {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0
        }
    },
    "selected_model": "ExampleClassifier",
    "model_path": "model.joblib",
    "limitations": []
}

The example metric values are placeholders, not expected performance.

When ML is not required, return an appropriate result, for example:

{
    "run_id": "run_001",
    "ml_required": false,
    "reason": "The question can be answered using descriptive analysis."
}

## Implementation Guidelines

- Use Pandas and scikit-learn where appropriate.
- Organize preprocessing and modelling into reproducible pipelines.
- Keep training and evaluation data separate.
- Prevent target leakage and preprocessing leakage.
- Establish a baseline before comparing more complex models.
- Select metrics based on the problem type.
- Record random seeds when applicable.
- Do not invent model results.
- Save model artifacts only when appropriate.
- Handle insufficient data and training failures explicitly.
- Do not modify the shared prepared dataset.

## Integration with CloudFlow

The ML Agent receives inputs and artifact paths from the orchestrator.

Its results should be made available to the Evaluation & Report Writer
Agent, which will assess the findings and incorporate the model's
performance, limitations, and conclusions into the final report.

The agent should return its output artifact paths and execution status
to the orchestrator.