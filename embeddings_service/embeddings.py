import torch
from transformers import AutoModel, AutoTokenizer
import pandas as pd
from tqdm import tqdm

# Load JinaAI Embedding Model
model = AutoModel.from_pretrained("jinaai/jina-embeddings-v3", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("jinaai/jina-embeddings-v3", trust_remote_code=True)

def generate_article_embeddings(df, batch_size=8, max_length=512):
    """
    Generates embeddings for news articles using AutoModel.
    """
    if "full_text" not in df.columns:
        print("❌ Error: 'full_text' column not found.")
        return df

    embeddings = []
    for i in tqdm(range(0, len(df), batch_size), desc="Generating embeddings"):
        batch_texts = df["full_text"].iloc[i : i + batch_size].tolist()

        inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        with torch.no_grad():
            batch_embeddings = model(**inputs, task="retrieval.passage").last_hidden_state.mean(dim=1)

        embeddings.extend(batch_embeddings.tolist())

    df["embedding"] = embeddings
    return df

def generate_query_embedding(query, max_length=512):
    """
    Generates an embedding for a search query.
    """
    inputs = tokenizer(query, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    with torch.no_grad():
        query_embedding = model(**inputs, task="retrieval.query").last_hidden_state.mean(dim=1)

    return query_embedding.tolist()
