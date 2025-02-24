import torch
from transformers import AutoModel, AutoTokenizer
import pandas as pd
from tqdm import tqdm

# Load the embedding model from the locally saved directory
model = AutoModel.from_pretrained("jinaai/jina-embeddings-v3", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("jinaai/jina-embeddings-v3", trust_remote_code=True)

def generate_article_embeddings(df, batch_size=8, max_length=512):
    """
    Generates embeddings for news articles using AutoModel.
    Concatenates title, snippet, and body before encoding.

    Args:
        df (pd.DataFrame): DataFrame containing 'title', 'snippet', and 'body'.
        batch_size (int): Number of samples to process at a time.
        max_length (int): Maximum token length per document.

    Returns:
        pd.DataFrame: DataFrame with an additional 'embedding' column.
    """
    required_columns = ["title", "snippet", "body"]
    if not all(col in df.columns for col in required_columns):
        print("❌ Error: Required columns ('title', 'snippet', 'body') not found.")
        return df

    df.fillna("", inplace=True)
    df["full_text"] = df["title"] + " " + df["snippet"] + " " + df["body"]

    embeddings = []
    for i in tqdm(range(0, len(df), batch_size), desc="Generating embeddings in batches"):
        batch_texts = df["full_text"].iloc[i:i+batch_size].tolist()

        inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        with torch.no_grad():
            batch_embeddings = model(**inputs, task="retrieval.passage").last_hidden_state.mean(dim=1)

        embeddings.extend(batch_embeddings.tolist())

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
    inputs = tokenizer(query, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    with torch.no_grad():
        query_embedding = model(**inputs, task="retrieval.query").last_hidden_state.mean(dim=1)

    return query_embedding.tolist()
