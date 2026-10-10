# CloudFlow

A multi-container, question-driven data analysis pipeline. Give it a dataset and a plain-English question; it analyzes the data, answers with evidence, and writes a report covering its workflow, decisions, findings, and limitations.

> **Status:** early prototype. Architecture is defined and the Data Prep profiler runs. The full four-stage pipeline is not implemented yet.

## What it does

**Input:** `churn.csv` + *"Which factors are associated with churn, and can we predict who will leave?"*

**Output:** an evidence-backed answer, a report, and the notebooks and artifacts behind it, so the analysis can be inspected and reproduced.

Not every question goes through machine learning. A descriptive question may only need EDA or a verified calculation.

## Architecture

```text
User: dataset + question
          |
          v
 Orchestrator / Planner     (creates run ID, plans stages, tracks state)
          |
          v
 1. Data Prep Agent         (profile, clean, flag target candidates)
          |
          v
 2. EDA Agent               (question-relevant stats and charts)
          |
          v
 3. Feature Eng. & ML Agent (features, baseline, train/evaluate (if needed))
          |
          v
 4. Evaluation & Report     (verify claims, answer, write report)
          |
          v
 Answer + report + artifacts
```

Each agent is a separate Docker container with a small API (FastAPI). The orchestrator coordinates them and does no analysis itself. Agents pass data through a shared per-run workspace.

| Stage | Key outputs |
|---|---|
| Data Prep | cleaned dataset, data profile, cleaning summary, notebook |
| EDA | findings JSON, charts, notebook |
| Feature Eng. & ML | model, predictions, experiment results, notebook |
| Evaluation & Report | final answer, report, evidence references |

## How agents work

Each analysis agent uses an LLM to plan analyses and write Python, runs that code in a Jupyter notebook, reads the real outputs, and iterates.

```text
plan -> write code -> execute notebook -> read results -> interpret -> save artifacts
```

Three things are kept separate: **executed code** (the notebook), **computed evidence** (structured results and metrics), and **interpretation** (the LLM's explanation). Numerical claims in the final answer are checked against stored artifacts, and associations are never presented as causation.

**Security:** LLM-generated code is arbitrary code. It must run in a restricted worker with no credentials, no Docker socket, no network, resource limits, and a read-only copy of the input. Containers help but are not a complete sandbox.

## Setup

> Only the Data Prep profiler runs today. These steps prepare a dev environment and verify what exists.

**Prerequisites:** Python 3.10+, Git, Docker (optional for now), and a free API key from an OpenAI-compatible provider such as [Groq](https://console.groq.com) or [Google AI Studio](https://aistudio.google.com). No OpenAI account needed. On Windows, use **WSL2** and keep the repo inside the WSL filesystem (e.g. `~/CloudFlow`).

## 1. Clone and create a virtual environment

```bash
git clone https://github.com/ananyast81/CloudFlow.git && cd CloudFlow
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

On Ubuntu you may need `sudo apt install python3-venv python3-pip` first. Check isolation with `which python` (should be inside `.venv/`) and `echo $PYTHONPATH` (should be empty).


## 2. Install dependencies

```bash
pip install -r requirements.txt   # or: pip install pandas pydantic openai
```

## 3. Configure the LLM

Create a `.env` in the repo root (gitignored, never commit it):

```env
# Groq
LLM_BASE_URL=https://api.groq.com/openai/v1
LLM_API_KEY=your_key
LLM_MODEL=llama-3.3-70b-versatile

# Gemini (alternative)
# LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
# LLM_MODEL=gemini-2.5-flash
```

Model names change; if you get "model not found", check the provider's model list. Then load the file:

```bash
set -a && source .env && set +a
```

## 4. Add a dataset

Save the Telco Customer Churn CSV as `data/churn.csv` (the `data/` folder is gitignored). Don't send real customer data through free-tier LLMs.

## 5. Docker (in progress)

Build from the **repo root** so `shared/` is included:

```bash
docker build -f agents/data-prep-agent/Dockerfile -t cloudflow-data-prep .
```

Create a `.dockerignore` in the repo root with:

```text
.venv/
.git/
data/
out/
workspace/
__pycache__/
.env
```

A `docker-compose.yml` for the full pipeline is planned.