FROM python:3.12-slim-trixie
COPY --from=ghcr.io/astral-sh/uv:0.9.6 /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml uv.lock ./
COPY services/ingest/pyproject.toml ./services/ingest/pyproject.toml
COPY services/capture ./services/capture

RUN uv sync --locked --project services/capture

WORKDIR /app/services/capture

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
