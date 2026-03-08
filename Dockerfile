FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV MODEL_PATH=s3://world-model-v1/biometric_model/final_model_20260307_200122.pth

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash app

WORKDIR /home/app

# Copy source code and install dependencies
COPY pyproject.toml poetry.lock ./
COPY src/ ./src/
RUN pip install poetry && \
    poetry config virtualenvs.in-project true && \
    poetry install --only=main

USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["poetry", "run", "python", "-m", "biometric_recognition.api.serve"]
