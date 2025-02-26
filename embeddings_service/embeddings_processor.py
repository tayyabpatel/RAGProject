import os
import fastavro
import pandas as pd
import logging
from io import BytesIO
from data_processing import preprocess_dataframe
from embeddings import generate_article_embeddings
from vector_store import insert_vectors

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

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
    insert_vectors(df)
