# Use official Python image
FROM python:3.11

# Set working directory
WORKDIR /app

# Install system dependencies using apt
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-dev \
    python3-setuptools \
    python3-numpy \
    python3-scipy \
    python3-requests \
    python3-torch \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . /app

# Upgrade pip and install Python dependencies
RUN python3 -m pip install --no-cache-dir --upgrade pip \
    && python3 -m pip install --no-cache-dir pandas \
    && python3 -m pip install --no-cache-dir fastavro \
    && python3 -m pip install --no-cache-dir -r requirements.txt  # Ensure python-multipart gets installed

# Expose API port
EXPOSE 8000

# Command to run FastAPI app
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]

# Set up cache directory for Hugging Face models
ENV TRANSFORMERS_CACHE=/app/huggingface_cache
