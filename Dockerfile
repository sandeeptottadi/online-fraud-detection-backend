FROM python:3.9.21-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first
COPY requirements.txt .

# Install Python packages with exact versions
RUN pip install -r requirements.txt

# Copy application files
COPY . .

# Verify files
RUN ls -la && python -c "import pickle; import sklearn; print(f'Python version: {pickle.format_version}')"

# Set environment variables
ENV PORT=8000

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]