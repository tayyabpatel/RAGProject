# Use official Python image as base
FROM python:3.9

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app code
COPY . .

# Set environment variables (modify as needed)
ENV CONFIG_PATH="/app/config.json"

# Run the streaming listener
CMD ["python", "main.py"]
