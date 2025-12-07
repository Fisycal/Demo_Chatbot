# Base
FROM python:3.12-slim

EXPOSE 8080

ARG MONGO_CONNECTION_STRING
ENV MONGO_CONNECTION_STRING=${MONGO_CONNECTION_STRING}

COPY . .
WORKDIR .
RUN pip install -r requirements.txt

# Run with Uvicorn
CMD ["python", "main.py"]

