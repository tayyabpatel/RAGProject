from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams
import numpy as np
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

# Connect to Qdrant
client = QdrantClient("localhost", port=6333)

# Define Qdrant collection schema
COLLECTION_NAME = "news_articles"

def create_collection():
    """
    Creates a Qdrant collection with the correct schema if it doesn't exist.
    """
    try:
        client.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
        )
        logging.info(f"✅ Collection '{COLLECTION_NAME}' created successfully.")
    except Exception as e:
        logging.error(f"❌ Error creating collection: {e}")

def insert_vectors(df):
    """
    Inserts article embeddings into Qdrant.

    Args:
        df (pd.DataFrame): DataFrame containing 'embedding' and 'publication_datetime' columns.
    """
    if df is None or "embedding" not in df.columns:
        logging.error("❌ Error: DataFrame is None or missing 'embedding' column.")
        return
    
    try:
        points = []
        for i, row in df.iterrows():
            publication_datetime = row.get("publication_datetime", "Unknown")

            # Convert NaT values to None, otherwise store as a string
            if pd.isna(publication_datetime) or publication_datetime == "NaT":
                publication_datetime = "Unknown"
            else:
                publication_datetime = str(publication_datetime)

            for chunk in row["content_chunks"]:
                points.append(
                    PointStruct(
                        id=i, 
                        vector=row["embedding"], 
                        payload={
                            "an": row["an"], 
                            "content_text": chunk,
                            "publication_datetime": publication_datetime  
                        }
                    )
                )
        
        logging.info(f"Inserting {len(points)} vectors into Qdrant.")
        client.upsert(COLLECTION_NAME, points)
        logging.info("✅ Data inserted successfully.")
    except Exception as e:
        logging.error(f"❌ Error inserting vectors: {e}")

def search_vectors(query_vector, top_k=5):
    """
    Searches for similar articles using the query embedding.

    Args:
        query_vector (list): The query embedding vector.
        top_k (int): Number of results to retrieve.

    Returns:
        list: Retrieved documents with similarity scores.
    """
    try:
        results = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=top_k,
            with_payload=True
        )
        logging.info(f"Search returned {len(results)} results.")
        return results
    except Exception as e:
        logging.error(f"❌ Error searching vectors: {e}")
        return None
