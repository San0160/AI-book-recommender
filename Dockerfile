# syntax=docker/dockerfile:1
ARG PYTHON_VERSION=3.10

# ---------- Stage 1: train the model ----------
FROM python:${PYTHON_VERSION}-slim AS trainer
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app/src LOG_TO_FILE=0
WORKDIR /app

COPY requirements.txt requirements-train.txt ./
RUN pip install --no-cache-dir -r requirements-train.txt

COPY src ./src
COPY config ./config
COPY scripts ./scripts
COPY params.yaml main.py ./
RUN python main.py && python scripts/check_quality.py


# ---------- Stage 2: slim serving image ----------
FROM python:${PYTHON_VERSION}-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app/src \
    LOG_TO_FILE=0 PORT=8080
WORKDIR /app

RUN useradd --create-home --uid 10001 app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    && mkdir -p /app/artifacts && chown -R app:app /app

COPY --chown=app:app src ./src
COPY --chown=app:app config ./config
COPY --chown=app:app templates ./templates
COPY --chown=app:app params.yaml app.py ./
COPY --chown=app:app static ./static
COPY --from=trainer --chown=app:app /app/artifacts/data_preprocessing/processed_data.csv artifacts/data_preprocessing/
COPY --from=trainer --chown=app:app /app/artifacts/data_transformation/feature_matrix.pkl artifacts/data_transformation/
COPY --from=trainer --chown=app:app /app/artifacts/model_trainer/model.pkl artifacts/model_trainer/
COPY --from=trainer --chown=app:app /app/artifacts/model_evaluation/metrics.json artifacts/model_evaluation/

USER app
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8080/ready', timeout=3)"

CMD ["sh", "-c", "exec uvicorn app:app --host 0.0.0.0 --port ${PORT:-8080}"]