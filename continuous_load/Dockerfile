# Use Python 3.9 as base image
FROM python:3.9

# Set working directory inside the container
WORKDIR /app

# Copy requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all module files into the container
COPY . .

# Command to run when the container starts
CMD ["python", "main.py"]

