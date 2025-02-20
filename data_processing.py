import fastavro
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from io import BytesIO

# Configure logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

def load_avro_to_dataframe(avro_source):
    """
    Loads AVRO file and converts to Pandas DataFrame.
    Supports both local file paths and file-like objects from API uploads.
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
        logging.info(f"Loaded AVRO file with {len(df)} records.")
        return df

    except Exception as e:
        logging.error(f"❌ Error loading AVRO file: {e}")
        return None

def chunk_text_by_words(text, max_words=700):
    """
    Splits text into chunks of at most `max_words` words.
    """
    words = text.split()
    if len(words) <= max_words:
        return [text]
    return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]

def convert_datetime_column(df, column_name, unit='ms'):
    """
    Converts EPOCH timestamp columns into readable datetime format.
    """
    if column_name in df.columns:
        logging.info(f"Checking values in {column_name} before conversion:")
        logging.info(df[column_name].head())

        df[column_name] = pd.to_datetime(df[column_name], unit=unit, errors='coerce')

        logging.info(f"Converted column {column_name} to datetime. Sample values:")
        logging.info(df[column_name].head())

    return df

def preprocess_dataframe(df):
    """
    Prepares DataFrame for embedding and storage:
    - Converts timestamps
    - Merges text fields
    - Splits text into chunks
    """
    if df is None:
        logging.error("❌ Error: DataFrame is None. Ensure AVRO file is loaded correctly.")
        return None

    df.fillna(value=np.nan, inplace=True)

    # Convert timestamps
    date_columns = ["ingestion_datetime", "availability_datetime", "modification_datetime", "publication_datetime", "publication_date"]
    for col in date_columns:
        df = convert_datetime_column(df, col, unit='ms')

    # Convert word_count to integer
    if "word_count" in df.columns:
        df["word_count"] = pd.to_numeric(df["word_count"], errors='coerce').fillna(0).astype(int)

    # Merge snippet and body into a single column
    if "snippet" in df.columns and "body" in df.columns:
        df["content_text"] = df["snippet"].fillna("") + " " + df["body"].fillna("")
        df.drop(columns=["snippet", "body"], inplace=True)

    # Apply chunking
    df["content_chunks"] = df["content_text"].apply(lambda x: chunk_text_by_words(x, max_words=700))

    # Format publication_datetime as string for storage
    if "publication_datetime" in df.columns:
        df["publication_datetime"] = df["publication_datetime"].apply(lambda x: str(x) if pd.notna(x) else "Unknown")

    logging.info("Final check on publication_datetime values:")
    logging.info(df["publication_datetime"].head())

    return df
