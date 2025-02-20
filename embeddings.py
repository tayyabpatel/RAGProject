import torch
from sentence_transformers import SentenceTransformer
import pandas as pd
from tqdm import tqdm

# Load the embedding model
embedding_model = SentenceTransformer("jinaai/jina-embeddings-v3", trust_remote_code=True)

def generate_article_embeddings(df, batch_size=32, max_length=512):
    """
    Generates embeddings for news articles and stores them in a new column using batch processing.

    Args:
        df (pd.DataFrame): DataFrame containing the news articles.
        batch_size (int): Number of samples to process at a time.
        max_length (int): Maximum token length per document to avoid memory overflow.

    Returns:
        pd.DataFrame: DataFrame with an additional 'embedding' column.
    """
    if "content_text" not in df.columns:
        print("❌ Error: 'content_text' column not found in DataFrame.")
        return df

    df["content_text"] = df["content_text"].fillna("").astype(str)

    # Split into batches
    embeddings = []
    for i in tqdm(range(0, len(df), batch_size), desc="Generating embeddings in batches"):
        batch_texts = df["content_text"].iloc[i:i+batch_size].tolist()

        # Truncate texts that exceed max_length
        batch_texts = [text[:max_length] for text in batch_texts]

        # Encode in smaller batches to prevent memory issues
        batch_embeddings = embedding_model.encode(
            batch_texts,
            convert_to_tensor=False  # Ensures output is always a list
        )

        # Ensure embeddings are always lists of floats
        batch_embeddings = [list(emb) if isinstance(emb, (list, torch.Tensor)) else [emb] for emb in batch_embeddings]

        embeddings.extend(batch_embeddings)

    df["embedding"] = embeddings
    return df

def generate_query_embedding(query, max_length=512):
    """
    Generates an embedding for an incoming search query.

    Args:
        query (str): The query text.
        max_length (int): Maximum token length for query.

    Returns:
        list: Embedding vector for the query.
    """
    query = query[:max_length]  # Truncate if too long
    query_embedding = embedding_model.encode(query, convert_to_tensor=False)  # Always returns a list

    # Ensure query embedding is always a list of floats
    return list(query_embedding) if isinstance(query_embedding, (list, torch.Tensor)) else [query_embedding]
