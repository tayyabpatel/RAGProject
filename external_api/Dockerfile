# Use official Python image
FROM python:3.11

# Set working directory
WORKDIR /app

# Copy requirements first
COPY requirements.txt /app/

# Install dependencies
RUN python3 -m pip install --no-cache-dir --upgrade pip \
    && python3 -m pip install --no-cache-dir -r requirements.txt

# Copy API source files
COPY api.py /app/

# Expose API port
EXPOSE 8000

# Run FastAPI server
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
