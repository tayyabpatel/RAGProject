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
COPY database.py /app/

# Set environment variables dynamically (No .env file copied)
ENV QDRANT_HOST=${QDRANT_HOST:-qdrant}
ENV QDRANT_PORT=${QDRANT_PORT:-6333}
ENV QDRANT_COLLECTION=${QDRANT_COLLECTION:-news_articles}

# Expose API port
EXPOSE 6000

# Run FastAPI
CMD ["uvicorn", "database:app", "--host", "0.0.0.0", "--port", "6000"]
