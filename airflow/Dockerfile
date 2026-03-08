FROM apache/airflow:2.8.1-python3.11

USER root

# Install system dependencies for ML packages
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file with proper permissions
COPY --chown=airflow:root requirements-airflow.txt /tmp/requirements-airflow.txt

USER airflow

# Install Python dependencies
RUN pip install --no-cache-dir -r /tmp/requirements-airflow.txt
