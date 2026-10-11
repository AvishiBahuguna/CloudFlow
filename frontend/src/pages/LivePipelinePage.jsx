
import { useState } from "react";
import { Link, useParams } from "react-router-dom";

import OrbitGraph from "../components/agents/OrbitGraph";
import AgentDossier from "../components/agents/AgentDossier";

import { mockPipeline } from "../data/mockPipeline.js";

function LivePipelinePage() {
  const { runId } = useParams();

  const [selectedAgentId, setSelectedAgentId] =
    useState("data_understanding");

  // Demo data is used ONLY for the demonstration route.
  const isDemo = !runId;

  const pipeline = isDemo ? mockPipeline : null;

  if (!pipeline) {
    return (
      <section className="live-pipeline-page">
        <p className="page-label mono">
          CLOUDFLOW / LIVE PIPELINE
        </p>

        <h1>Pipeline Run: {runId}</h1>

        <p className="page-description">
          The dashboard is ready for backend integration.
          Run-status fetching will be implemented next.
        </p>

        <Link to="/new-run">
          Return to New Run
        </Link>
      </section>
    );
  }

  return (
    <section className="live-pipeline-page">
      <header className="pipeline-header">
        <div>
          <p className="page-label mono">
            RUN / {pipeline.runId}
          </p>

          <h1>{pipeline.stage}</h1>

          <p className="page-description">
            Monitor agents and inspect their decisions.
          </p>
        </div>

        <span className="pipeline-demo-label">
          DEMONSTRATION DATA
        </span>
      </header>

      <div className="pipeline-dashboard">
        <OrbitGraph
          pipeline={pipeline}
          selectedAgentId={selectedAgentId}
          onSelectAgent={setSelectedAgentId}
        />

        <AgentDossier
          agentId={selectedAgentId}
          pipeline={pipeline}
        />
      </div>
    </section>
  );
}

export default LivePipelinePage;
