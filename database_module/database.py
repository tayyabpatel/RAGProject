import os
import numpy as np
import logging
from fastapi import FastAPI, HTTPException
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load environment variables
QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "news_articles")

# Initialize Qdrant Client
client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

app = FastAPI()

class InsertRequest(BaseModel):
    embeddings: list
    metadata: dict

class QueryRequest(BaseModel):
    query_vector: list
    top_k: int = 5

@app.post("/create_collection/")
def create_collection():
    """
    Creates or recreates a Qdrant collection for storing article embeddings.
    """
    try:
        client.recreate_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config={"embedding": VectorParams(size=1024, distance=Distance.COSINE)},
        )
        logging.info(f"✅ Collection '{QDRANT_COLLECTION}' created successfully.")
        return {"status": "success", "message": f"Collection '{QDRANT_COLLECTION}' created."}
    except Exception as e:
        logging.error(f"❌ Error creating collection: {e}")
        raise HTTPException(status_code=500, detail="Error creating collection.")

@app.post("/insert_vectors/")
def insert_vectors(request: InsertRequest):
    """
    Inserts article embeddings into Qdrant.
    """
    try:
        points = []
        for i, embedding in enumerate(request.embeddings):
            vector_id = abs(hash(str(embedding))) % (10**12)  # Unique vector ID
            points.append(
                PointStruct(
                    id=vector_id,
                    vector={"embedding": embedding},
                    payload=request.metadata
                )
            )

        if points:
            client.upsert(QDRANT_COLLECTION, points)
            logging.info(f"✅ Inserted {len(points)} vectors into Qdrant.")
            return {"status": "success", "message": f"Inserted {len(points)} vectors."}
        else:
            logging.warning("⚠️ No valid vectors to insert.")
            return {"status": "warning", "message": "No valid vectors."}
    
    except Exception as e:
        logging.error(f"❌ Error inserting vectors: {e}")
        raise HTTPException(status_code=500, detail="Error inserting vectors.")

@app.post("/search_vectors/")
def search_vectors(request: QueryRequest):
    """
    Searches for similar articles in Qdrant based on a query embedding.
    """
    try:
        results = client.search(
            collection_name=QDRANT_COLLECTION,
            query_vector={"embedding": request.query_vector},
            limit=request.top_k,
            with_payload=True
        )
        logging.info(f"🔍 Search returned {len(results)} results.")
        return {"results": results}
    except Exception as e:
        logging.error(f"❌ Error searching vectors: {e}")
        raise HTTPException(status_code=500, detail="Error searching Qdrant.")
