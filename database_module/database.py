import os
import numpy as np
import logging
from fastapi import FastAPI, HTTPException
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams, NamedVector
from pydantic import BaseModel
import pandas as pd

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
    metadata: list  # Each embedding should have corresponding metadata

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
            vectors_config={
                "embedding": VectorParams(size=1024, distance=Distance.COSINE)
            },
        )
        logging.info(f"✅ Collection '{QDRANT_COLLECTION}' created successfully.")
        return {"status": "success", "message": f"Collection '{QDRANT_COLLECTION}' created."}
    except Exception as e:
        logging.error(f"❌ Error creating collection: {e}")
        raise HTTPException(status_code=500, detail=f"Error creating collection: {str(e)}")

@app.post("/insert_vectors/")
def insert_vectors(request: InsertRequest):
    """
    Inserts batch article embeddings into Qdrant.
    """
    if not request.embeddings or not request.metadata:
        raise HTTPException(status_code=400, detail="Invalid input: Embeddings or metadata missing.")

    if len(request.embeddings) != len(request.metadata):
        raise HTTPException(status_code=400, detail="Mismatch between embeddings and metadata lengths.")

    try:
        points = []
        for i, (embedding, meta) in enumerate(zip(request.embeddings, request.metadata)):
            vector_id = abs(hash(str(embedding))) % (10**12)  # Unique vector ID
            
            points.append(
                PointStruct(
                    id=vector_id,
                    vector={"embedding": embedding},  # ✅ Named vector field
                    payload=meta  # ✅ Ensure metadata is properly stored
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
        raise HTTPException(status_code=500, detail=f"Error inserting vectors: {str(e)}")

@app.post("/search_vectors/")
def search_vectors(request: QueryRequest):
    """
    Searches for similar articles in Qdrant based on a query embedding.
    """
    try:
        results = client.search(
            collection_name=QDRANT_COLLECTION,
            query_vector=NamedVector(name="embedding", vector=request.query_vector),  # ✅ Specify vector name
            limit=request.top_k,
            with_payload=True
        )
        logging.info(f"🔍 Search returned {len(results)} results.")
        
        formatted_results = [
            {
                "an": res.payload.get("an", "Unknown"),
                "publication_datetime": res.payload.get("publication_datetime", "Unknown"),
                "content_text": res.payload.get("content_text", "No Content Available")
            }
            for res in results
        ]

        return {"results": formatted_results}
    except Exception as e:
        logging.error(f"❌ Error searching vectors: {e}")
        raise HTTPException(status_code=500, detail=f"Error searching Qdrant: {str(e)}")
