import fastavro
import pandas as pd
import numpy as np
from datetime import datetime

# Load AVRO file and convert to Pandas DataFrame
def load_avro_to_dataframe(avro_file_path):
    """
    Loads an AVRO file into a Pandas DataFrame and performs basic cleaning.

    Args:
        avro_file_path (str): Path to the AVRO file.

    Returns:
        pd.DataFrame: DataFrame containing the data.
    """
    try:
        with open(avro_file_path, "rb") as f:
            reader = fastavro.reader(f)
            records = [record for record in reader]
        df = pd.DataFrame(records)
    except Exception as e:
        print(f"❌ Error loading AVRO file: {e}")
        return None

    return df

# Function to chunk content by words (not characters)
def chunk_text_by_words(text, max_words=700):
    """
    Splits text into chunks based on word count.

    Args:
        text (str): The text to be split.
        max_words (int): Maximum words per chunk.

    Returns:
        list: A list of text chunks.
    """
    words = text.split()
    if len(words) <= max_words:
        return [text]  # If the text is within the limit, return as a single chunk
    
    chunks = []
    for i in range(0, len(words), max_words):
        chunks.append(" ".join(words[i:i + max_words]))  # Create chunks of `max_words`
    
    return chunks

# Preprocess DataFrame
def preprocess_dataframe(df):
    """
    Cleans and preprocesses the DataFrame:
    - Converts timestamps to datetime
    - Converts word_count to integer
    - Concatenates 'snippet' and 'body' into 'content_text'
    - Splits 'content_text' into chunks if it exceeds 700 words

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: Preprocessed DataFrame with chunked text.
    """
    if df is None:
        print("❌ Error: DataFrame is None. Ensure AVRO file is loaded correctly.")
        return None

    df.fillna(value=np.nan, inplace=True)  # Replace nulls with NaN

    # Convert long timestamps to datetime
    date_columns = ["ingestion_datetime", "availability_datetime", "modification_datetime", "publication_datetime", "publication_date"]
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], unit='s', errors='coerce')

    # Convert word_count to integer
    if "word_count" in df.columns:
        df["word_count"] = pd.to_numeric(df["word_count"], errors='coerce').fillna(0).astype(int)

    # Merge snippet and body into a new column 'content_text'
    if "snippet" in df.columns and "body" in df.columns:
        df["content_text"] = df["snippet"].fillna("") + " " + df["body"].fillna("")
        df.drop(columns=["snippet", "body"], inplace=True)  # Drop original columns

    # Apply chunking if necessary
    df["content_chunks"] = df["content_text"].apply(lambda x: chunk_text_by_words(x, max_words=700))

    return df
