# Use official Python image
FROM python:3.11

# Set working directory
WORKDIR /app

# Install system dependencies using apt
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first to install dependencies properly
COPY requirements.txt /app/

# Install necessary dependencies
RUN python3 -m pip install --no-cache-dir --upgrade pip \
    && python3 -m pip install --no-cache-dir pandas fastavro \
    && python3 -m pip install --no-cache-dir transformers torch einops 'numpy<2' \
    && if [ -f "requirements.txt" ]; then python3 -m pip install --no-cache-dir -r requirements.txt; fi

# Copy project files selectively (avoid unnecessary files)
COPY embeddings.py vector_database.py api.py /app/
COPY data_processing.py /app/

# Expose API port
EXPOSE 8000

# Command to run FastAPI app
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]

# Set up cache directory for Hugging Face models
ENV TRANSFORMERS_CACHE=/app/huggingface_cache
