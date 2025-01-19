FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy only necessary files
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application files
COPY model.pkl .
COPY scaler.pkl .

# Verify files are copied
RUN ls -la

# Set environment variables
ENV PORT=8000

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]