import fastavro
import pandas as pd
import numpy as np
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

# Load AVRO file and convert to Pandas DataFrame
def load_avro_to_dataframe(avro_file_path):
    try:
        with open(avro_file_path, "rb") as f:
            reader = fastavro.reader(f)
            records = [record for record in reader]
        df = pd.DataFrame(records)
        logging.info(f"Loaded AVRO file: {avro_file_path} with {len(df)} records.")
    except Exception as e:
        logging.error(f"❌ Error loading AVRO file: {e}")
        return None

    return df

# Function to chunk content by words
def chunk_text_by_words(text, max_words=700):
    words = text.split()
    if len(words) <= max_words:
        return [text]  # If the text is within the limit, return as a single chunk
    
    chunks = [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]
    return chunks

# Function to convert timestamps correctly
def convert_datetime_column(df, column_name, unit='ms'):
    """
    Converts EPOCH timestamp columns into readable datetime format.
    """
    if column_name in df.columns:
        logging.info(f"Checking values in {column_name} before conversion:")
        logging.info(df[column_name].head())

        # Convert timestamp using the correct unit
        df[column_name] = pd.to_datetime(df[column_name], unit=unit, errors='coerce')

        logging.info(f"Converted column {column_name} to datetime. Sample values:")
        logging.info(df[column_name].head())

    return df

# Preprocess DataFrame
def preprocess_dataframe(df):
    if df is None:
        logging.error("❌ Error: DataFrame is None. Ensure AVRO file is loaded correctly.")
        return None

    df.fillna(value=np.nan, inplace=True)  # Replace nulls with NaN

    # Convert timestamps using 'ms' for publication_datetime
    date_columns = ["ingestion_datetime", "availability_datetime", "modification_datetime", "publication_datetime", "publication_date"]
    for col in date_columns:
        df = convert_datetime_column(df, col, unit='ms')

    # Convert word_count to integer
    if "word_count" in df.columns:
        df["word_count"] = pd.to_numeric(df["word_count"], errors='coerce').fillna(0).astype(int)

    # Merge snippet and body into a new column 'content_text'
    if "snippet" in df.columns and "body" in df.columns:
        df["content_text"] = df["snippet"].fillna("") + " " + df["body"].fillna("")
        df.drop(columns=["snippet", "body"], inplace=True)

    # Apply chunking
    df["content_chunks"] = df["content_text"].apply(lambda x: chunk_text_by_words(x, max_words=700))

    # Ensure publication_datetime is properly formatted before inserting into Qdrant
    if "publication_datetime" in df.columns:
        df["publication_datetime"] = df["publication_datetime"].apply(
            lambda x: str(x) if pd.notna(x) else "Unknown"
        )

    logging.info("Final check on publication_datetime values:")
    logging.info(df["publication_datetime"].head())

    return df
