
# Start containers
```bash
cd ollama-qdrant
docker compose up -d
```

# Stop containers
```bash
docker compose down
```

# Pull ollama images
```bash
docker exec -it ollama ollama pull mxbai-embed-large:335m
docker exec -it ollama ollama pull llama3.2:latest
# docker exec -it ollama ollama pull mistral:latest
```
# Verify containers are running
```bash
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"