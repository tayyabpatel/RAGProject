from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import numpy as np
import pandas as pd
import uuid
import os

# Qdrant connection settings
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")  # Change if using Qdrant Cloud
COLLECTION_NAME = "news_articles"

client = QdrantClient(QDRANT_URL)

# Create collection schema (if not exists)
def create_collection(vector_size=768):
    """
    Creates a Qdrant collection for storing embeddings if it does not already exist.
    """
    existing_collections = client.get_collections().collections
    if COLLECTION_NAME not in [col.name for col in existing_collections]:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        print(f"✅ Collection '{COLLECTION_NAME}' created successfully.")
    else:
        print(f"✅ Collection '{COLLECTION_NAME}' already exists.")

# Insert data into Qdrant
def insert_embeddings(df):
    """
    Inserts articles with their embeddings into Qdrant.
    
    Args:
        df (pd.DataFrame): DataFrame containing 'embedding' and metadata fields.
    """
    points = []
    for _, row in df.iterrows():
        if isinstance(row["embedding"], list):  # Ensure embedding is valid
            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),  # Unique ID
                    vector=row["embedding"],  # Embedding vector
                    payload={  # Metadata
                        "title": row.get("title", ""),
                        "date": row.get("publication_datetime", ""),
                        "content": row.get("content_text", ""),
                    },
                )
            )

    if points:
        client.upsert(collection_name=COLLECTION_NAME, points=points)
        print(f"✅ Inserted {len(points)} articles into Qdrant.")
    else:
        print("⚠️ No valid embeddings found in DataFrame.")

# Search function using cosine similarity
def search_articles(query_embedding, top_k=5):
    """
    Searches for the most relevant articles using cosine similarity.

    Args:
        query_embedding (list): The embedding vector of the search query.
        top_k (int): Number of results to return.

    Returns:
        list: Top-k matching articles with metadata.
    """
    search_results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_embedding,
        limit=top_k,
    )

    return [
        {"title": hit.payload["title"], "content": hit.payload["content"], "score": hit.score}
        for hit in search_results
    ]

