# Use official Python image
FROM python:3.11

# Set working directory
WORKDIR /app

# Copy requirements first
COPY requirements.txt /app/

# Install dependencies
RUN python3 -m pip install --no-cache-dir --upgrade pip \
    && python3 -m pip install --no-cache-dir -r requirements.txt

# Copy source files
COPY embeddings_processor.py embeddings.py vector_store.py data_processing.py api.py /app/

# Copy environment variables
COPY .env /app/

# Set environment variables
ENV QDRANT_HOST=qdrant
ENV QDRANT_PORT=6333

# Expose API port for embeddings service
EXPOSE 5001

# Run as background worker & API
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "5001"]
