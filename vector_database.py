import os
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams
import numpy as np
import pandas as pd
import logging
from embeddings import generate_article_embeddings  # Ensure correct function import

# Configure logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

# Connect to Qdrant (Ensure correct Docker service name)
client = QdrantClient(
    host=os.getenv("QDRANT_HOST", "qdrant"),  # Ensure it matches docker-compose
    port=int(os.getenv("QDRANT_PORT", "6333")),
)

# Define Qdrant collection schema
COLLECTION_NAME = "news_articles"

def create_collection():
    """
    Creates a Qdrant collection with an explicit named vector field.
    """
    try:
        client.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={"embedding": VectorParams(size=1024, distance=Distance.COSINE)},
        )
        logging.info(f"✅ Collection '{COLLECTION_NAME}' created successfully with named vector field.")
    except Exception as e:
        logging.error(f"❌ Error creating collection: {e}")

def insert_vectors(df):
    """
    Inserts article embeddings into Qdrant.

    Args:
        df (pd.DataFrame): DataFrame containing 'title', 'snippet', 'body', and other metadata.
    """
    if df is None or "title" not in df.columns or "snippet" not in df.columns or "body" not in df.columns:
        logging.error("❌ Error: DataFrame is missing required columns.")
        return
    
    try:
        # Generate embeddings for full articles
        df = generate_article_embeddings(df)

        if "embedding" not in df.columns:
            logging.error("❌ Error: Embeddings were not generated.")
            return
        
        logging.debug(f"Generated embeddings:\n{df[['full_text', 'embedding']].head()}")  # Debugging

        points = []
        for i, row in df.iterrows():
            vector_id = abs(hash(row["full_text"])) % (10**12)  # Prevent negative IDs

            embedding = np.array(row["embedding"]).flatten().tolist()

            points.append(
                PointStruct(
                    id=vector_id, 
                    vector={"embedding": embedding},  # ✅ Correctly formatted named vector
                    payload={"an": row.get("an", "Unknown"), "content_text": row["full_text"]}
                )
            )
        
        if points:
            client.upsert(COLLECTION_NAME, points)
            logging.info("✅ Data inserted successfully.")
        else:
            logging.warning("⚠️ No valid vectors to insert.")
            
    except Exception as e:
        logging.error(f"❌ Error inserting vectors: {e}")
