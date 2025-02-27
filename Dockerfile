FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create necessary directories
RUN mkdir -p data/vectors data/indices

# Set environment variables
ENV PYTHONPATH=/app
ENV FLASK_APP=api/coordinator_api.py

# Default command (can be overridden in docker-compose.yml)
CMD ["python", "api/coordinator_api.py"]
