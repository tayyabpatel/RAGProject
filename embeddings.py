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
    if "content_text" in df.columns:
        # Ensure empty or NaN values do not break the embedding process
        df["content_text"] = df["content_text"].fillna("").astype(str)

        # Compute embeddings for all articles
        embeddings = embedding_model.encode(
            df["content_text"].tolist(), 
            convert_to_tensor=True, 
            show_progress_bar=True
        )

        # Convert embeddings to a list for storage in DataFrame
        df["embedding"] = embeddings.cpu().numpy().tolist()

    return df

def generate_query_embedding(query):
    """
    Generates an embedding for an incoming search query.

    Args:
        query (str): The query text.

    Returns:
        list: Embedding vector for the query.
    """
    if not query or not isinstance(query, str):
        raise ValueError("Query must be a non-empty string")

    query_embedding = embedding_model.encode(
        query, convert_to_tensor=True
    ).cpu().numpy().tolist()

    return query_embedding
