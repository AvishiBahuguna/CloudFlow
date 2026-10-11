
import { AGENTS, HUMAN_GATE } from "../../constants/agentConfig";

function AgentDossier({ agentId, pipeline }) {
  const allAgents = [...AGENTS, HUMAN_GATE];

  const agent = allAgents.find(
    (item) => item.id === agentId
  );

  const details = pipeline.agents[agentId];

  if (!agent || !details) {
    return (
      <section className="dossier-panel">
        <p>No agent information available.</p>
      </section>
    );
  }

  return (
    <section className="dossier-panel">
      <div className="dossier-heading">
        <span>DOSSIER / AGENT DETAILS</span>

        <span
          className="dossier-status"
          style={{ color: agent.color }}
        >
          {details.status.toUpperCase()}
        </span>
      </div>

      <div className="dossier-agent-title">
        <span
          className="dossier-color-dot"
          style={{ backgroundColor: agent.color }}
        />

        <h2>{agent.name}</h2>
      </div>

      <p className="dossier-summary">
        {details.summary}
      </p>

      <div className="dossier-section">
        <h3>Decisions and actions</h3>

        {details.decisions.length > 0 ? (
          <ul className="dossier-decisions">
            {details.decisions.map((decision, index) => (
              <li key={`${index}-${decision}`}>
                {decision}
              </li>
            ))}
          </ul>
        ) : (
          <p className="dossier-empty">
            No decisions recorded yet.
          </p>
        )}
      </div>
    </section>
  );
}

export default AgentDossier;
