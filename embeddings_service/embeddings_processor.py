import os
import fastavro
import pandas as pd
import logging
from io import BytesIO
from data_processing import preprocess_dataframe
from embeddings import generate_article_embeddings
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Qdrant Client
client = QdrantClient(
    host=os.getenv("QDRANT_HOST", "qdrant"),
    port=int(os.getenv("QDRANT_PORT", "6333")),
)

# Qdrant Collection Name
COLLECTION_NAME = "news_articles"

def create_collection():
    """
    Creates a Qdrant collection with a named vector field.
    """
    try:
        client.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={"embedding": VectorParams(size=1024, distance=Distance.COSINE)},
        )
        logging.info(f"✅ Collection '{COLLECTION_NAME}' created successfully.")
    except Exception as e:
        logging.error(f"❌ Error creating collection: {e}")

def load_avro_to_dataframe(avro_source):
    """
    Loads an AVRO file into a Pandas DataFrame.
    """
    try:
        if isinstance(avro_source, str):  # Local file path
            with open(avro_source, "rb") as f:
                reader = fastavro.reader(f)
                records = [record for record in reader]
        elif isinstance(avro_source, BytesIO):  # Uploaded file
            avro_source.seek(0)
            reader = fastavro.reader(avro_source)
            records = [record for record in reader]
        else:
            raise ValueError("Invalid AVRO source type.")

        df = pd.DataFrame(records)
        logging.info(f"✅ Loaded AVRO file with {len(df)} records.")
        return df

    except Exception as e:
        logging.error(f"❌ Error loading AVRO file: {e}")
        return None

def process_and_store_embeddings(avro_file_path):
    """
    Loads AVRO, processes data, generates embeddings, and stores them in Qdrant.
    """
    df = load_avro_to_dataframe(avro_file_path)
    if df is None or df.empty:
        logging.error("❌ Error: DataFrame is empty or invalid.")
        return

    # Preprocess articles
    df = preprocess_dataframe(df)

    # Generate embeddings
    df = generate_article_embeddings(df)

    if "embedding" not in df.columns:
        logging.error("❌ Error: Embeddings were not generated.")
        return

    # Store in Qdrant
    points = []
    for i, row in df.iterrows():
        vector_id = abs(hash(row["full_text"])) % (10**12)  # Unique ID
        embedding = np.array(row["embedding"]).flatten().tolist()

        points.append(
            PointStruct(
                id=vector_id,
                vector={"embedding": embedding},
                payload={"an": row.get("an", "Unknown"), "content_text": row["full_text"]},
            )
        )

    if points:
        client.upsert(COLLECTION_NAME, points)
        logging.info("✅ Data stored in Qdrant successfully.")
    else:
        logging.warning("⚠️ No valid vectors to insert.")

