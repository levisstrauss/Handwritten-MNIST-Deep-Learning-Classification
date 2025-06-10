# ========================================================================
# DOCKERFILE
# ========================================================================


FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY models/ ./models/

# Create necessary directories
RUN mkdir -p data logs

# Set environment variables
ENV PYTHONPATH=/app

# Expose port for potential web service
EXPOSE 8080

# Default command
CMD ["python", "src/train.py", "--help"]
