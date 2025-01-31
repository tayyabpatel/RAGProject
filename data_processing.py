import fastavro
import pandas as pd
import numpy as np
# from jina import JinaEmbeddings  # Commented out for now
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
    with open(avro_file_path, "rb") as f:
        reader = fastavro.reader(f)
        records = [record for record in reader]

    df = pd.DataFrame(records)

    return df

# Preprocess DataFrame
def preprocess_dataframe(df):
    """
    Cleans and preprocesses the DataFrame:
    - Converts timestamps to datetime
    - Converts word_count to integer
    - Concatenates 'snippet' and 'body' into 'content_text'
    - Handles missing values
    
    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: Preprocessed DataFrame.
    """
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

    return df

# Commented out for now since the embeddings module isn't implemented
# def generate_embeddings(df, embedding_model="jina-embeddings-v3"):
#     """
#     Generates text embeddings for the content_text column using JinaAI.
# 
#     Args:
#         df (pd.DataFrame): DataFrame containing the articles.
#         embedding_model (str): The embedding model to use.
# 
#     Returns:
#         pd.DataFrame: DataFrame with embeddings added.
#     """
#     embedder = JinaEmbeddings(embedding_model)
# 
#     if "content_text" in df.columns:
#         df["embedding"] = df["content_text"].apply(lambda x: embedder.encode([x])[0] if pd.notna(x) and x.strip() else None)
# 
#     return df

# Main function to process AVRO and return DataFrame
def process_avro(avro_file_path):
    """
    Processes an AVRO file and returns a DataFrame.

    Args:
        avro_file_path (str): Path to the AVRO file.

    Returns:
        pd.DataFrame: Final processed DataFrame.
    """
    df = load_avro_to_dataframe(avro_file_path)
    df = preprocess_dataframe(df)
    
    # Commented out for now since embeddings aren't implemented yet
    # df = generate_embeddings(df)
    
    return df
