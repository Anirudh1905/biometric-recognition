FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1

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
COPY --chmod=755 src/ ./src/
RUN pip install poetry && \
    poetry config virtualenvs.in-project true && \
    poetry install --with=api --without=dev,pipeline

USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["poetry", "run", "uvicorn", "biometric_recognition.api.serve:app", "--host", "0.0.0.0", "--port", "8000"]
