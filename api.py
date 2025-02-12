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
        dict: JSON response containing relevant articles and their relevant content chunks.
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
        publication_datetime = result.payload.get("publication_datetime", "Unknown")
        headline = result.payload.get("headline", "No Headline Available")
        content_chunks = result.payload.get("content_text")

        # If content_chunks is a list with multiple chunks, keep relevant_chunks structure
        if isinstance(content_chunks, list) and len(content_chunks) > 1:
            response.append({
                "an_number": an_number,
                "relevant_chunks": [
                    {"text": chunk, "publication_datetime": publication_datetime}
                    for chunk in content_chunks
                ]
            })
        else:
            # If only one chunk (or string), flatten response structure
            response.append({
                "an": an_number,
                "publication_datetime": publication_datetime,
                "headline": headline,
                "text": content_chunks if isinstance(content_chunks, str) else content_chunks[0]
            })

    return {"results": response}
