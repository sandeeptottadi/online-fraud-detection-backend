FROM python:3.9.21-slim

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Debug information
RUN python -c "import sys; print(f'Python version: {sys.version}')"
RUN python -c "import sklearn; print(f'sklearn version: {sklearn.__version__}')"
RUN ls -la

ENV PORT=8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]