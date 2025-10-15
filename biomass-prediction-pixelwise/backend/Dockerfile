FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements_simple.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements_simple.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/data/processed /app/models /app/outputs

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "app.main_simple:app", "--host", "0.0.0.0", "--port", "8000"]