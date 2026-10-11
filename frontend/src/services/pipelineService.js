
import { apiRequest } from "./apiClient";

export async function createPipelineRun({ file, objective }) {
  const formData = new FormData();

  formData.append("file", file);
  formData.append("objective", objective.trim());

  const response = await apiRequest("/runs", {
    method: "POST",
    body: formData,
  });

  const runId = response?.run_id ?? response?.id;

  if (!runId) {
    throw new Error(
      "The backend did not return a pipeline run ID."
    );
  }

  return {
    runId,
    status: response.status ?? "created",
  };
}

export async function getPipelineStatus(runId) {
  return apiRequest(
    `/runs/${encodeURIComponent(runId)}`
  );
}

export async function getPipelineHistory() {
  return apiRequest("/runs");
}
