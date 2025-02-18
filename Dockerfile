# Use official Python image
FROM python:3.11

# Set working directory
WORKDIR /app

# Install system dependencies using apt
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    python3-venv \
    python3-setuptools \
    python3-numpy \
    python3-scipy \
    python3-pandas \
    python3-requests \
    python3-torch \
    python3-nltk \
    python3-sklearn \
    python3-fastapi \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . /app

# Upgrade pip and install remaining Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Expose API port
EXPOSE 8000

# Command to run FastAPI app
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]

