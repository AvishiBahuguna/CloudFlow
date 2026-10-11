export const mockPipeline = {
  runId: "DEMO-CHURN-001",
  stage: "Human approval",
  status: "waiting",
  retryCount: 0,
  maxRetries: 3,

  agents: {
    data_understanding: {
      status: "completed",
      summary:
        "Inspected the dataset, identified missing values and removed duplicate records.",
      decisions: [
        "Impute missing Age values using median.",
        "Remove one duplicate row.",
      ],
    },

    analysis_features: {
      status: "completed",
      summary:
        "Analyzed feature distributions and examined class balance.",
      decisions: [
        "Use stratified train-test splitting.",
        "Prioritize F1-score for evaluation.",
      ],
    },

    modeling: {
      status: "pending",
      summary:
        "Waiting for the approved training plan.",
      decisions: [],
    },

    evaluation: {
      status: "pending",
      summary:
        "Evaluation will begin after model training.",
      decisions: [],
    },

    reporting: {
      status: "pending",
      summary:
        "The final report will be generated after evaluation.",
      decisions: [],
    },

    human_gate: {
      status: "waiting",
      summary:
        "The pipeline is waiting for human approval before training.",
      decisions: [
        "Review the prediction target.",
        "Approve preprocessing and model choices.",
      ],
    },
  },

  metrics: {
    rowsBefore: 5,
    rowsAfter: 4,
    missingBefore: 1,
    missingAfter: 0,
    duplicatesBefore: 1,
    duplicatesAfter: 0,
  },

  logs: [
    {
      id: "log-1",
      time: "10:42:11",
      agent: "DATA",
      message: "Missing values detected in Age.",
    },
    {
      id: "log-2",
      time: "10:42:13",
      agent: "DATA",
      message: "One duplicate row removed.",
    },
    {
      id: "log-3",
      time: "10:42:20",
      agent: "ANALYSIS",
      message: "Class imbalance detected.",
    },
    {
      id: "log-4",
      time: "10:42:34",
      agent: "ORCH",
      message: "Waiting for human approval.",
    },
  ],
};
