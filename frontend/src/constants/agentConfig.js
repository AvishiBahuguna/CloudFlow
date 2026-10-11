
export const AGENTS = [
  {
    id: "data_understanding",
    name: "Data Understanding",
    shortName: "DATA",
    color: "#22D3EE",
    x: 50,
    y: 14,
  },
  {
    id: "analysis_features",
    name: "Analysis & Features",
    shortName: "ANALYSIS",
    color: "#8B7CFF",
    x: 82,
    y: 32,
  },
  {
    id: "modeling",
    name: "Modeling",
    shortName: "MODEL",
    color: "#34D399",
    x: 50,
    y: 86,
  },
  {
    id: "evaluation",
    name: "Evaluation & Diagnostics",
    shortName: "EVAL",
    color: "#FBBF24",
    x: 18,
    y: 68,
  },
  {
    id: "reporting",
    name: "Reporting",
    shortName: "REPORT",
    color: "#F472B6",
    x: 18,
    y: 32,
  },
];

export const HUMAN_GATE = {
  id: "human_gate",
  name: "Human Approval",
  shortName: "GATE",
  color: "#FBBF24",
  x: 82,
  y: 68,
};

export const AGENT_STATUSES = {
  PENDING: "pending",
  RUNNING: "running",
  COMPLETED: "completed",
  FAILED: "failed",
  WAITING: "waiting",
};
