
export const DATASET_CONFIG = {
  maxFileSizeMB: 10,
  allowedExtensions: [".csv"],
  previewRows: 5,
};

export const OBJECTIVE_SUGGESTIONS = [
  {
    label: "Classification",
    text: "Predict whether a customer will churn based on the available customer features.",
  },
  {
    label: "Regression",
    text: "Predict house prices using the numerical and categorical features in the dataset.",
  },
  {
    label: "Data Analysis",
    text: "Analyze this dataset, identify important patterns, and explain which features are most influential.",
  },
];
