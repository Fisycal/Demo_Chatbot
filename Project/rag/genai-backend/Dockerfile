# Base
FROM python:3.12-slim

EXPOSE 8080

ARG MONGO_CONNECTION_STRING
ENV MONGO_CONNECTION_STRING=${MONGO_CONNECTION_STRING}

# Install system dependencies
# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY . .
WORKDIR .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Run with Uvicorn
CMD ["python", "main.py"]

