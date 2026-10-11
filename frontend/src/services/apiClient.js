
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL?.replace(/\/$/, "") || "";

export async function apiRequest(endpoint, options = {}) {
  if (!API_BASE_URL) {
    throw new Error(
      "Backend URL is not configured. Set VITE_API_BASE_URL."
    );
  }

  let response;

  try {
    response = await fetch(
      `${API_BASE_URL}${endpoint}`,
      options
    );
  } catch (error) {
    if (error.name === "AbortError") {
      throw error;
    }

    throw new Error(
      "Unable to connect to the backend. Check that FastAPI is running."
    );
  }

  if (!response.ok) {
    let message = `Request failed with status ${response.status}.`;

    try {
      const body = await response.json();

      if (typeof body.detail === "string") {
        message = body.detail;
      } else if (typeof body.message === "string") {
        message = body.message;
      }
    } catch {
      // Keep the default error message.
    }

    throw new Error(message);
  }

  if (response.status === 204) {
    return null;
  }

  return response.json();
}
