from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from embeddings import generate_query_embedding
from vector_database import search_vectors

app = FastAPI()

class QueryRequest(BaseModel):
    query: str

@app.post("/search/")
async def search_articles(request: QueryRequest):
    """
    API endpoint to process a query:
    - Embed the query
    - Search Qdrant for the closest 5 articles
    - Return the AN number and 5 relevant chunks from each article
    """
    try:
        # Generate query embedding
        query_embedding = generate_query_embedding(request.query)

        # Search for closest articles
        search_results = search_vectors(query_embedding, top_k=5)

        if not search_results:
            raise HTTPException(status_code=404, detail="No relevant articles found.")

        response_data = []
        for result in search_results:
            payload = result.payload
            an_number = payload.get("an", "N/A")  # Extract AN number
            content_text = payload.get("content_text", "")

            # Extract 5 chunks of relevant text
            chunks = content_text.split(". ")  # Split by sentence
            relevant_chunks = chunks[:5] if len(chunks) >= 5 else chunks  # Take first 5

            response_data.append({
                "an_number": an_number,
                "relevant_chunks": relevant_chunks
            })

        return {"results": response_data}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")
