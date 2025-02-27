from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import os
import logging
from vector_database import search_vectors  # Ensure this function exists

app = FastAPI()

# Load environment variables
EMBEDDINGS_SERVICE_URL = os.getenv("EMBEDDINGS_SERVICE_URL", "http://localhost:5001")
QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")
QDRANT_PORT = os.getenv("QDRANT_PORT", "6333")

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
        try:
            response = requests.post(
                f"{EMBEDDINGS_SERVICE_URL}/embed_query", 
                json={"query": request.query}, 
                timeout=10  # Set timeout to prevent hanging
            )
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.Timeout:
            logging.error("❌ Timeout error while calling Embeddings API.")
            raise HTTPException(status_code=504, detail="Embeddings service timeout.")
        except requests.exceptions.RequestException as e:
            logging.error(f"❌ Failed to reach Embeddings API: {e}")
            raise HTTPException(status_code=502, detail="Error reaching embeddings service.")

        # Extract embedding from response
        query_embedding = data.get("embedding")
        if not query_embedding:
            raise HTTPException(status_code=500, detail="Embedding generation failed.")

        logging.info("✅ Embedding received successfully.")

        # Step 2: Search Qdrant
        search_results = search_vectors(query_embedding, qdrant_host=QDRANT_HOST, qdrant_port=QDRANT_PORT, top_k=5)

        if not search_results:
            return {"error": "No relevant articles found."}

        # Step 3: Format response
        response = []
        for result in search_results:
            response.append({
                "an": result.payload.get("an"),
                "publication_datetime": result.payload.get("publication_datetime", "Unknown"),
                "content_text": result.payload.get("content_text", "No Content Available"),
            })

        return {"results": response}

    except Exception as e:
        logging.error(f"❌ Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")
