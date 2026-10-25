# CloudFlow

A multi-container, question-driven data analysis pipeline. Give it a dataset and a plain-English question; it analyzes the data, answers with evidence, and generates a report covering its workflow, decisions, findings, and limitations.

> **Status:** Early prototype. The FastAPI backend, frontend integration, and Dockerized Data Prep Agent are implemented. The remaining analysis stages and complete end-to-end pipeline are still under development.

## What It Does

**Input:** A dataset such as `churn.csv` and a question such as:

> Which factors are associated with customer churn, and can we predict who will leave?

**Output:** A data preparation result, structured report, and supporting artifacts. The intended final output includes an evidence-backed answer, relevant findings, and the notebooks and artifacts needed to inspect and reproduce the analysis.

Not every question requires machine learning. A descriptive question may only need exploratory data analysis or a verified calculation.

## Architecture and Workflow

```text
User: Dataset + Question
          |
          v
Frontend (React + Vite)
          |
          v
FastAPI Backend
          |
          v
Orchestrator / Planner (planned)
          |
          v
1. Data Prep Agent
   - Profiles the dataset
   - Plans and executes cleaning operations
   - Saves the cleaned dataset and preparation report
          |
          v
2. EDA Agent (planned)
   - Question-relevant statistics and charts
          |
          v
3. Feature Engineering & ML Agent (planned)
   - Feature preparation, model training and evaluation
          |
          v
4. Evaluation & Report Agent (planned)
   - Verifies findings and generates the final report
          |
          v
Answer + Report + Supporting Artifacts
```

Currently, the FastAPI backend launches the Data Prep Agent in a separate Docker container for each pipeline run. The container processes the uploaded dataset and writes its outputs to a shared workspace.

The remaining agents and orchestrator will extend this workflow as development progresses.

### Agent Outputs

| Stage | Key outputs | Status |
|---|---|---|
| Data Prep | Cleaned dataset, initial profile, operation plan, preparation report, notebook artifacts | Implemented |
| EDA | Findings, statistics, charts, notebook | Planned |
| Feature Engineering & ML | Features, model, predictions, evaluation results, notebook | Planned |
| Evaluation & Report | Final answer, report, evidence references | Planned |

---

# Setup and Installation

## Prerequisites

Install the following:

- Python 3.10+
- Git
- Node.js and npm
- Docker Engine
- An API key from a supported OpenAI-compatible LLM provider, such as [Groq](https://console.groq.com/)

Docker must be running, and the user running the FastAPI backend must have permission to execute Docker commands.

On Windows, use **WSL2** and preferably keep the repository inside the WSL filesystem.

The instructions below assume Ubuntu or another Linux environment.

## 1. Clone the Repository

```bash
git clone https://github.com/ananyast81/CloudFlow.git
cd CloudFlow
```

## 2. Create a Python Virtual Environment

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

If creating the virtual environment fails on Ubuntu, install the required packages:

```bash
sudo apt update
sudo apt install python3-venv python3-pip
```

Verify that the virtual environment is active:

```bash
which python
```

The returned path should point to the repository's `.venv` directory.

## 3. Install Backend Dependencies

With the virtual environment activated, run:

```bash
pip install -r requirements.txt
```

The root requirements file contains dependencies needed by the development environment and backend.

## 4. Configure the LLM

Create a `.env` file in the repository root:

```bash
touch .env
```

Add your provider configuration:

```env
# Groq
LLM_BASE_URL=https://api.groq.com/openai/v1
LLM_API_KEY=your_api_key
LLM_MODEL=openai/gpt-oss-120b
```

Replace `your_api_key` with your actual API key.

Model availability and names can change. If the provider returns a `model_not_found` error, check its current list of supported models.

The backend loads the root `.env` file, and the Dockerized Data Prep Agent receives the configured LLM environment variables when the backend launches it.

**Keep `.env` private. Do not commit API keys or other secrets to Git.**

## 5. Build the Data Prep Agent Docker Image

Make sure Docker is running:

```bash
docker info
```

Build the image from the **repository root**:

```bash
docker build \
  -f agents/data_prep_agent/Dockerfile \
  -t cloudflow-data-prep:latest \
  .
```

The final `.` is important: it sets the repository root as the Docker build context, allowing the Dockerfile to copy the agent and shared modules.

Verify that the image was created:

```bash
docker images cloudflow-data-prep
```

The backend uses this image to execute the Data Prep Agent for each pipeline run.

You only need to rebuild the image when you change code or dependencies included in the image.

### Docker Ignore Configuration

The repository root should contain a `.dockerignore` file to exclude local files that should not be included in the build context.

Example:

```text
.venv/
.git/
data/
workspace/
out/
__pycache__/
*.pyc
.env
```

This keeps local datasets, run artifacts, environment files, and development files out of the image build context.

The `.env` file is supplied to the container through the backend's environment configuration rather than being copied into the image.

---

# Running CloudFlow

Start the backend and frontend in separate terminals. Build the Docker image before starting a pipeline run.

## 6. Start the FastAPI Backend

Open a terminal and navigate to the repository root:

```bash
cd ~/CloudFlow
source .venv/bin/activate
```

Start the backend:

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

The backend will be available at:

- **API base URL:** `http://localhost:8000`
- **Interactive API documentation:** `http://localhost:8000/docs`
- **Health check:** `http://localhost:8000/health`

The `--reload` option automatically restarts the development server when Python files change.

### Verify the Backend

In another terminal, run:

```bash
curl http://localhost:8000/health
```

Expected response:

```json
{
  "status": "ok"
}
```

You can also open `http://localhost:8000/docs` in your browser to inspect the available API endpoints.

### Available Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/health` | Check whether the backend is responding |
| `POST` | `/runs` | Upload a dataset, provide an objective, and start a pipeline run |
| `GET` | `/runs` | Retrieve the run records held by the backend |
| `GET` | `/runs/{run_id}` | Retrieve the status and results of a specific run |

### How a Pipeline Run Executes

When a user submits a dataset and objective through `POST /runs`:

1. The backend validates the objective and accepts the uploaded CSV.
2. The uploaded dataset is saved under `workspace/<run_id>/input.csv`.
3. The backend creates a run record and returns a run ID.
4. A background task launches the `cloudflow-data-prep:latest` Docker image.
5. The container processes the dataset and writes its outputs to the shared workspace.
6. The backend reads the container status and preparation report.
7. The run record is updated with its final status and results.

The run status generally progresses through:

```text
queued -> running -> completed -> failed
```

The frontend can poll `GET /runs/{run_id}` to retrieve the current status and results.

**Important:** Keep Docker running while executing pipeline runs. If you change the Data Prep Agent's code, rebuild its Docker image before testing the changes.

## 7. Start the Frontend

Open a second terminal:

```bash
cd ~/CloudFlow/frontend
npm install
npm run dev
```

Vite will print the local development URL, typically:

```text
http://localhost:5173
```

Open that address in your browser.

Ensure the FastAPI backend is running on port `8000` and that the frontend's API configuration points to the correct backend URL.

---

# Development and Troubleshooting

## Check Docker

If a pipeline run fails to start, verify that Docker is available:

```bash
docker info
```

Check whether the agent image exists:

```bash
docker images cloudflow-data-prep
```

Inspect running containers:

```bash
docker ps
```

Because the backend runs the agent with automatic container removal, a successfully completed container may no longer appear in `docker ps`.

## Inspect Run Artifacts

After submitting a dataset, inspect the corresponding workspace directory:

```bash
ls -R workspace/<run_id>
```

Replace `<run_id>` with the actual ID returned by the API.

Check the container status:

```bash
cat workspace/<run_id>/container_status.json
```

Inspect the preparation report:

```bash
cat workspace/<run_id>/data-prep/preparation_report.json
```

These files can help distinguish a data preparation failure from an API or container-launching problem.

## Inspect Backend Logs

The backend terminal displays request logs and output captured from the Data Prep Agent container.

If a run fails, inspect these logs first. They may contain the underlying Python exception or Docker error.

## Common Issues

| Problem | What to check |
|---|---|
| `ModuleNotFoundError` | Activate `.venv` and install the root requirements |
| Docker image not found | Build `cloudflow-data-prep:latest` from the repository root |
| Docker permission denied | Verify Docker is running and the backend user has Docker permissions |
| LLM authentication error | Check `LLM_API_KEY` in the root `.env` file |
| LLM model not found | Verify that `LLM_MODEL` is supported by your provider |
| Pipeline run fails | Inspect backend logs and `container_status.json` |
| `/runs` returns no records after restarting | Run records are currently in memory and are cleared when the backend restarts |
| Frontend cannot reach the backend | Confirm that Uvicorn is running on port `8000` and check the frontend API configuration |

---


# Roadmap

Planned development includes:

- Implementing the EDA Agent.
- Implementing the Feature Engineering and ML Agent.
- Implementing evaluation and final report generation.
- Building the orchestrator to coordinate the agents.
- Adding persistent run tracking.
- Improving execution isolation, error handling, and resource management.
- Integrating the complete pipeline with the frontend.
- Adding Docker Compose configuration for easier multi-service development.

---