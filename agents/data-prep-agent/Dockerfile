FROM python:3.10-slim
WORKDIR /app
COPY agents/data-prep-agent/requirements.txt .
RUN pip install --no-cache-dir requirements.txt
COPY shared/ ./shared/
COPY agents/data-prep-agent ./agent/
ENV PYTHONPATH=/app
CMD ["python", "agent/main.py"]