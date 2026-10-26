import {
  BrainCircuit,
  Check,
  Circle,
  Database,
  FileText,
  FlaskConical,
  ScanSearch,
  ShieldCheck,
  UserRoundCheck,
} from "lucide-react";

const stages = [
  ["data_understanding", "Data understanding", Database, "Profile & clean"],
  [
    "analysis_features",
    "Analysis & features",
    ScanSearch,
    "Explore & transform",
  ],
  ["human_review", "Your review", UserRoundCheck, "Validate the plan"],
  ["modeling", "Modeling", BrainCircuit, "Train & compare"],
  ["evaluation", "Evaluation", FlaskConical, "Diagnose & improve"],
  ["reporting", "Reporting", FileText, "Explain & deliver"],
];

export default function Workflow({
  agents = {},
  selected,
  onSelect,
  compact = false,
}) {
  return (
    <div className={`workflow ${compact ? "workflow-compact" : ""}`}>
      {stages.map(([id, label, Icon, caption], index) => {
        const status = agents[id]?.status || "pending";
        return (
          <button
            key={id}
            type="button"
            onClick={() => onSelect?.(id)}
            className={`workflow-node node-${status} ${selected === id ? "is-selected" : ""}`}
            aria-label={`${label}: ${status}`}
            aria-pressed={selected === id}
          >
            <div className="node-top">
              <span className="node-number">0{index + 1}</span>
              {status === "completed" ? (
                <Check size={13} />
              ) : status === "waiting" ? (
                <ShieldCheck size={13} />
              ) : (
                <Circle size={7} fill="currentColor" />
              )}
            </div>
            <div className="node-icon">
              <Icon size={23} strokeWidth={1.7} />
            </div>
            <strong>{label}</strong>
            <small>{caption}</small>
            <span className="node-state">{status}</span>
          </button>
        );
      })}
    </div>
  );
}
