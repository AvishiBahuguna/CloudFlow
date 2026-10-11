
const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL?.replace(/\/$/, "") || "";

export async function createPipelineRun({
  file,
  objective,
  signal,
}) {
  if (!API_BASE_URL) {
    throw new Error(
      "Backend API is not configured yet. Set VITE_API_BASE_URL after confirming the FastAPI endpoint."
    );
  }

  const formData = new FormData();

  formData.append("file", file);
  formData.append("objective", objective.trim());

  const response = await fetch(
    `${API_BASE_URL}/runs`,
    {
      method: "POST",
      body: formData,
      signal,
    }
  );

  if (!response.ok) {
    let message = `Request failed (${response.status}).`;

    try {
      const errorBody = await response.json();

      if (typeof errorBody.detail === "string") {
        message = errorBody.detail;
      }
    } catch {
      // Keep the fallback error message.
    }

    throw new Error(message);
  }

  const result = await response.json();

  const runId = result.run_id ?? result.id;

  if (!runId) {
    throw new Error(
      "Backend response did not contain a run identifier."
    );
  }

  return {
    runId,
    status: result.status ?? "created",
  };
}
