from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams
import numpy as np
import uuid

# Define Qdrant collection name
COLLECTION_NAME = "news_articles"

def connect_qdrant():
    """
    Establishes a connection to the Qdrant vector database.
    
    Returns:
        QdrantClient: Connected Qdrant client instance.
    """
    return QdrantClient("localhost", port=6333)

def create_collection(client, collection_name=COLLECTION_NAME):
    """
    Creates a Qdrant collection if it doesn't exist.

    Args:
        client (QdrantClient): Qdrant client instance.
        collection_name (str): Name of the collection.
    """
    try:
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE),
        )
        print(f"Collection '{collection_name}' created successfully.")
    except Exception as e:
        print(f"Error creating collection: {e}")

def insert_embeddings(client, collection_name, df):
    """
    Inserts article embeddings into Qdrant.

    Args:
        client (QdrantClient): Qdrant client instance.
        collection_name (str): Name of the collection.
        df (pd.DataFrame): DataFrame containing 'embedding' column and metadata.
    """
    if df is None or "embedding" not in df.columns:
        print("Error: DataFrame is None or missing 'embedding' column.")
        return
    
    try:
        points = [
            PointStruct(
                id=str(uuid.uuid4()),  # Generate unique ID for each vector
                vector=vec.tolist() if isinstance(vec, np.ndarray) else vec,  # Ensure vector is list format
                payload={
                    "title": df.iloc[i].get("title", "No Title"),
                    "publication_date": str(df.iloc[i].get("publication_date", "Unknown")),
                    "source": df.iloc[i].get("source_name", "Unknown"),
                    "content": df.iloc[i].get("content_text", "No Content"),
                }
            )
            for i, vec in enumerate(df["embedding"])
        ]

        client.upsert(
            collection_name=collection_name,
            points=points
        )
        print(f"Inserted {len(points)} embeddings into '{collection_name}'")
    except Exception as e:
        print(f"Error inserting embeddings: {e}")

def search_vectors(client, collection_name, query_vector, top_k=5):
    """
    Searches for similar articles using the query embedding.

    Args:
        client (QdrantClient): Qdrant client instance.
        collection_name (str): Name of the collection.
        query_vector (list): The query embedding vector.
        top_k (int): Number of results to retrieve.

    Returns:
        list: Retrieved documents with similarity scores.
    """
    try:
        results = client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=top_k,
            query_filter=None  # Allow searching the whole dataset
        )
        return results
    except Exception as e:
        print(f"Error searching vectors: {e}")
        return None
