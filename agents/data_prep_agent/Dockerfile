FROM python:3.10-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

COPY agents/data_prep_agent/requirements.txt ./requirements.txt

RUN python -m pip install --no-cache-dir -r requirements.txt

COPY agents/ ./agents/
COPY shared/ ./shared/

RUN mkdir -p /workspace

CMD ["python", "-m", "agents.data_prep_agent.container_entrypoint"]