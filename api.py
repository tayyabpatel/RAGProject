from fastapi import FastAPI
from pydantic import BaseModel
from embeddings import generate_query_embedding
from vector_database import search_vectors

app = FastAPI()

class QueryRequest(BaseModel):
    query: str

@app.post("/search/")
async def search_articles(request: QueryRequest):
    """
    API endpoint to search for articles similar to the given query.

    Args:
        request (QueryRequest): JSON payload containing the search query.

    Returns:
        dict: JSON response containing the top 5 relevant articles and their relevant content chunks.
    """
    query_embedding = generate_query_embedding(request.query)

    if query_embedding is None:
        return {"error": "Failed to generate query embedding."}

    search_results = search_vectors(query_embedding, top_k=5)

    if not search_results:
        return {"error": "No relevant articles found."}

    response = []
    for result in search_results:
        an_number = result.payload.get("an")
        content_text = result.payload.get("content_text")
        publication_datetime = result.payload.get("publication_datetime", "Unknown")

        response.append({
            "an_number": an_number,
            "relevant_chunks": [
                {"text": content_text, "publication_datetime": publication_datetime}
            ]
        })

    return {"results": response}
