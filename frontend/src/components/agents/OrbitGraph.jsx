
import { AGENTS, HUMAN_GATE } from "../../constants/agentConfig";
import AgentNode from "./AgentNode";

import "./OrbitGraph.css";

function OrbitGraph({
  pipeline,
  selectedAgentId,
  onSelectAgent,
}) {
  const nodes = [...AGENTS, HUMAN_GATE];

  return (
    <section className="orbit-panel">
      <div className="orbit-panel-header">
        <span>ORBIT / AGENT WORKFLOW</span>
        <span>{pipeline.status.toUpperCase()}</span>
      </div>

      <div className="orbit-container">
        <svg
          className="orbit-lines"
          viewBox="0 0 100 100"
          preserveAspectRatio="none"
          aria-hidden="true"
        >
          <ellipse
            cx="50"
            cy="50"
            rx="32"
            ry="36"
            fill="none"
            stroke="#2A3040"
            strokeWidth="0.4"
          />

          {nodes.map((agent) => (
            <line
              key={agent.id}
              x1="50"
              y1="50"
              x2={agent.x}
              y2={agent.y}
              stroke="#2A3040"
              strokeWidth="0.3"
              strokeDasharray="1 1"
            />
          ))}
        </svg>

        <div className="orchestrator">
          <span className="orchestrator-name">ORCH</span>
          <span className="orchestrator-label">
            ORCHESTRATOR
          </span>
        </div>

        {nodes.map((agent) => (
          <AgentNode
            key={agent.id}
            agent={agent}
            status={
              pipeline.agents[agent.id]?.status ?? "pending"
            }
            selected={selectedAgentId === agent.id}
            onSelect={onSelectAgent}
          />
        ))}
      </div>

      <div className="orbit-panel-footer">
        <span>RETRIES</span>
        <strong>
          {pipeline.retryCount} / {pipeline.maxRetries}
        </strong>
      </div>
    </section>
  );
}

export default OrbitGraph;
