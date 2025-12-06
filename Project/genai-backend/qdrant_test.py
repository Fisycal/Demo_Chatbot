from qdrant_client import QdrantClient

# Connect to your local Qdrant instance
q_client = QdrantClient(host="localhost", port=6333)

# Make sure a collection exists with a named vector
q_client.recreate_collection(
    collection_name="test_collection",
    vectors_config={
        "dense": {"size": 4, "distance": "Cosine"}
    }
)

# Insert a dummy point with a named vector
q_client.upsert(
    collection_name="test_collection",
    points=[
        {
            "id": 1,
            "vector": {"dense": [0.1, 0.2, 0.3, 0.4]},
            "payload": {"doc": "hello world"}
        }
    ]
)

# Query using vector_name
hits = q_client.query_points(
    collection_name="test_collection",
    query=[0.1, 0.2, 0.3, 0.4],
    vector_name="dense",
    limit=1,
    with_payload=True
)

print("Query result:", hits)