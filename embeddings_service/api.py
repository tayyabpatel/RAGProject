from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import logging
from embeddings import generate_query_embedding

app = FastAPI()

# Load environment variables
QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))

logging.basicConfig(level=logging.INFO)

class QueryRequest(BaseModel):
    query: str

@app.post("/generate_embedding/")
async def generate_embedding(request: QueryRequest):
    """
    Accepts a user query and generates an embedding using JinaAI model.
    """
    try:
        logging.info(f"🔍 Received embedding request: {request.query}")

        # Generate query embedding
        embedding = generate_query_embedding(request.query)
        if not embedding:
            raise HTTPException(status_code=500, detail="Embedding generation failed.")

        logging.info("✅ Embedding generated successfully.")
        return {"embedding": embedding}

    except Exception as e:
        logging.error(f"❌ Error generating embedding: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")
