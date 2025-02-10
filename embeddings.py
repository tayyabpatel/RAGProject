import torch
from sentence_transformers import SentenceTransformer
import pandas as pd

# Load the embedding model from Hugging Face
embedding_model = SentenceTransformer("jinaai/jina-embeddings-v3", trust_remote_code=True)

def generate_article_embeddings(df):
    """
    Generates embeddings for news articles and stores them in a new column.

    Args:
        df (pd.DataFrame): DataFrame containing the news articles.

    Returns:
        pd.DataFrame: DataFrame with an additional 'embedding' column.
    """
    if df is None or "content_text" not in df.columns:
        print("Error: DataFrame is None or missing 'content_text' column.")
        return None

    # Ensure empty or NaN values do not break the embedding process
    df["content_text"] = df["content_text"].fillna("").astype(str)

    try:
        # Compute embeddings for all articles
        embeddings = embedding_model.encode(df["content_text"].tolist(),
                                            convert_to_tensor=True,
                                            batch_size=32,
                                            show_progress_bar=True)

        # Convert embeddings to a list for storage in DataFrame
        df["embedding"] = embeddings.cpu().tolist()
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        return None
    
    return df

def generate_query_embedding(query):
    """
    Generates an embedding for an incoming search query.

    Args:
        query (str): The query text.

    Returns:
        list: Embedding vector for the query.
    """
    try:
        query_embedding = embedding_model.encode(query, convert_to_tensor=True).cpu().tolist()
    except Exception as e:
        print(f"Error generating query embedding: {e}")
        return None

    return query_embedding
