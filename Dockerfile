FROM python:3.9-slim

WORKDIR /

# Install system dependencies including gcc
RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application files
COPY main.py .
COPY model.pkl .
COPY scaler.pkl .

# Set environment variables
ENV PORT=8000

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]