FROM python:3.11-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Copy the full project (server needs adaptive_nav package)
COPY . /app

# Install Python deps
RUN pip install --no-cache-dir -r adaptive_nav/server/requirements.txt

# Expose the OpenEnv default port
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "adaptive_nav.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
