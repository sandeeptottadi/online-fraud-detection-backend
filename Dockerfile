FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy all application files
COPY . .

# Debug: List directory contents
RUN echo "Contents of /app:" && ls -la

# Set environment variables
ENV PORT=8000

# Start the application with the correct module path
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]