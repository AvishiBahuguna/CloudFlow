
import {
  Check,
  AlertCircle,
  LoaderCircle,
  Clock3,
} from "lucide-react";

function AgentNode({
  agent,
  status = "pending",
  selected = false,
  onSelect,
}) {
  function getStatusIcon() {
    switch (status) {
      case "completed":
        return <Check size={20} />;

      case "running":
        return <LoaderCircle size={20} className="node-spinner" />;

      case "failed":
        return <AlertCircle size={20} />;

      case "waiting":
        return <Clock3 size={20} />;

      default:
        return <span className="node-dot" />;
    }
  }

  return (
    <button
      type="button"
      className={`agent-node agent-node--${status} ${
        selected ? "agent-node--selected" : ""
      }`}
      style={{
        left: `${agent.x}%`,
        top: `${agent.y}%`,
        "--node-color": agent.color,
      }}
      onClick={() => onSelect(agent.id)}
      aria-label={`${agent.name}: ${status}`}
      aria-pressed={selected}
      title={`${agent.name} — ${status}`}
    >
      <span className="agent-node-circle">
        {getStatusIcon()}
      </span>

      <span className="agent-node-label">
        {agent.shortName}
      </span>
    </button>
  );
}

export default AgentNode;
