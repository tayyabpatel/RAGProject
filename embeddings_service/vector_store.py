import os
import logging
import numpy as np
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load environment variables
QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))

# Initialize Qdrant Client
client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

# Define Collection Name
COLLECTION_NAME = "news_articles"

def create_collection():
    """ Creates a Qdrant collection with a named vector field for embeddings. """
    try:
        client.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={"embedding": VectorParams(size=1024, distance=Distance.COSINE)},
        )
        logging.info(f"✅ Collection '{COLLECTION_NAME}' created successfully.")
    except Exception as e:
        logging.error(f"❌ Error creating collection: {e}")

def insert_vectors(df):
    """ Inserts batch of article embeddings into Qdrant. """
    if df is None or "embedding" not in df.columns:
        logging.error("❌ Error: DataFrame is None or missing 'embedding' column.")
        return

    try:
        points = []
        for _, row in df.iterrows():
            vector_id = abs(hash(row["full_text"])) % (10**12)  # Unique vector ID
            embedding = np.array(row["embedding"]).flatten().tolist()

            points.append(
                PointStruct(
                    id=vector_id,
                    vector={"embedding": embedding},  # ✅ Fixed: Added named vector
                    payload={"an": row.get("an", "Unknown"), "content_text": row["full_text"]}
                )
            )

        if points:
            client.upsert(COLLECTION_NAME, points)
            logging.info(f"✅ Successfully inserted {len(points)} vectors into Qdrant.")
        else:
            logging.warning("⚠️ No valid vectors to insert.")

    except Exception as e:
        logging.error(f"❌ Error inserting vectors: {e}")
