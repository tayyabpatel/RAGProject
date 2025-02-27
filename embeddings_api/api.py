from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import logging
from embeddings import generate_query_embedding, generate_article_embeddings
import pandas as pd

app = FastAPI()

# Load environment variables
QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))

logging.basicConfig(level=logging.INFO)

class QueryRequest(BaseModel):
    query: str

@app.post("/embed_query/")
async def embed_query(request: QueryRequest):
    """ Generates an embedding for a search query (dummy embeddings) """
    try:
        if not request.query.strip():  # Reject empty or whitespace-only queries
            raise HTTPException(status_code=400, detail="Query text cannot be empty.")

        embedding = generate_query_embedding(request.query)
        return {"embedding": embedding}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating query embedding: {str(e)}")

@app.post("/embed_text/")
async def embed_text(request: QueryRequest):
    """ Generates an embedding for full articles (dummy embeddings) """
    try:
        if not request.query.strip():  # Reject empty or whitespace-only queries
            raise HTTPException(status_code=400, detail="Query text cannot be empty.")

        embedding = generate_article_embeddings(pd.DataFrame([{"full_text": request.query}]))
        return {"embedding": embedding["embedding"].tolist()[0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating text embedding: {str(e)}")
