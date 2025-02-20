from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from io import BytesIO
import logging
from data_processing import process_avro_data
from vector_database import insert_vectors, search_vectors
from embeddings import generate_query_embedding

app = FastAPI()

logging.basicConfig(level=logging.INFO)

class QueryRequest(BaseModel):
    query: str

@app.post("/search/")
async def search_articles(request: QueryRequest):
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

        if isinstance(content_chunks, list) and len(content_chunks) > 1:
            response.append({
                "an_number": an_number,
                "relevant_chunks": [
                    {"text": chunk, "publication_datetime": publication_datetime}
                    for chunk in content_chunks
                ]
            })
        else:
            response.append({
                "an": an_number,
                "publication_datetime": publication_datetime,
                "headline": headline,
                "text": content_chunks if isinstance(content_chunks, str) else content_chunks[0]
            })

    return {"results": response}

@app.post("/upload/")
async def upload_avro(file: UploadFile = File(...)):
    if not file.filename.endswith(".avro"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an AVRO file.")

    try:
        contents = await file.read()
        bytes_reader = BytesIO(contents)

        df = process_avro_data(bytes_reader)

        if df is None or df.empty:
            raise HTTPException(status_code=400, detail="Invalid AVRO file or no records found.")

        insert_vectors(df)

        return JSONResponse(content={"message": "File processed and stored successfully."}, status_code=200)
    except Exception as e:
        logging.error(f"Error processing AVRO: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")
