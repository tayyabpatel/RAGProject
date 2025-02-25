from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import os
import logging
from vector_database import search_vectors

app = FastAPI()

# Load environment variables
EMBEDDINGS_SERVICE_URL = os.getenv("EMBEDDINGS_SERVICE_URL", "http://localhost:5001")

logging.basicConfig(level=logging.INFO)

class QueryRequest(BaseModel):
    query: str

@app.post("/search/")
async def search_articles(request: QueryRequest):
    """
    Accepts a user query, calls the embeddings API, retrieves relevant articles, and returns them.
    """
    try:
        logging.info(f"🔍 Received search request: {request.query}")

        # Step 1: Call embeddings service
        response = requests.post(f"{EMBEDDINGS_SERVICE_URL}/embed_query", json={"query": request.query})
        
        if response.status_code != 200:
            logging.error(f"❌ Failed to get embeddings: {response.json()}")
            raise HTTPException(status_code=500, detail="Error generating query embedding.")

        query_embedding = response.json().get("embedding")
        if not query_embedding:
            raise HTTPException(status_code=500, detail="Embedding generation failed.")

        logging.info("✅ Embedding received successfully.")

        # Step 2: Search Qdrant
        search_results = search_vectors(query_embedding, top_k=5)

        if not search_results:
            return {"error": "No relevant articles found."}

        # Step 3: Format response
        response = []
        for result in search_results:
            response.append({
                "an": result.payload.get("an"),
                "publication_datetime": result.payload.get("publication_datetime", "Unknown"),
                "content_text": result.payload.get("content_text", "No Content Available")
            })

        return {"results": response}

    except Exception as e:
        logging.error(f"❌ Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")
