FROM python:3.9-slim

WORKDIR /

# Install system dependencies including gcc
RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy everything to root
COPY . .

# Install Python packages
RUN pip install -r requirements.txt

# Set environment variables
ENV PORT=8000

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]