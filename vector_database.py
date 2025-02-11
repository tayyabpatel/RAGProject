from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams
import numpy as np

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
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE),  # Updated to 1024
        )
        print(f"✅ Collection '{COLLECTION_NAME}' created successfully.")
    except Exception as e:
        print(f"❌ Error creating collection: {e}")

def insert_vectors(df):
    """
    Inserts article embeddings into Qdrant.

    Args:
        df (pd.DataFrame): DataFrame containing 'embedding' column.
    """
    if df is None or "embedding" not in df.columns:
        print("❌ Error: DataFrame is None or missing 'embedding' column.")
        return
    
    try:
        points = []
        for i, row in df.iterrows():
            for chunk in row["content_chunks"]:  # Insert each chunk separately
                points.append(
                    PointStruct(
                        id=i, 
                        vector=row["embedding"], 
                        payload={"an": row["an"], "content_text": chunk}
                    )
                )

        client.upsert(COLLECTION_NAME, points)
        print("✅ Data inserted successfully.")
    except Exception as e:
        print(f"❌ Error inserting vectors: {e}")

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
            with_payload=True  # Ensure metadata (title, an, content) is returned
        )
        return results
    except Exception as e:
        print(f"❌ Error searching vectors: {e}")
        return None
