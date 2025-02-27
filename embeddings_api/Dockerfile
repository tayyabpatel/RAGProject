# Use official Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (to speed up installs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker layer caching
COPY requirements.txt /app/

# Install dependencies in a virtual environment
RUN python3 -m venv /app/venv && \
    /app/venv/bin/pip install --no-cache-dir --upgrade pip && \
    /app/venv/bin/pip install --no-cache-dir -r requirements.txt

# Copy source files
COPY embeddings_processor.py embeddings.py vector_store.py data_processing.py api.py /app/

# Copy environment variables **only if they exist**
RUN if [ -f .env ]; then cp .env /app/; fi

# Set environment variables
ENV QDRANT_HOST=qdrant
ENV QDRANT_PORT=6333
ENV PATH="/app/venv/bin:$PATH"

# Expose API port
EXPOSE 5001

# Run FastAPI server
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "5001"]
