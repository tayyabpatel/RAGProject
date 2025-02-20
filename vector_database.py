import os
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams
import numpy as np
import pandas as pd
import logging
from embeddings import generate_article_embeddings  # Ensure correct function import

# Configure logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

# Connect to Qdrant (Ensure correct Docker service name)
client = QdrantClient(
    host=os.getenv("QDRANT_HOST", "qdrant"),  # Ensure it matches docker-compose
    port=int(os.getenv("QDRANT_PORT", "6333")),
)

# Define Qdrant collection schema
COLLECTION_NAME = "news_articles"

def create_collection():
    """
    Creates a Qdrant collection with an explicit named vector field.
    """
    try:
        client.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={"embedding": VectorParams(size=1024, distance=Distance.COSINE)},
        )
        logging.info(f"✅ Collection '{COLLECTION_NAME}' created successfully with named vector field.")
    except Exception as e:
        logging.error(f"❌ Error creating collection: {e}")

def insert_vectors(df):
    """
    Inserts article embeddings into Qdrant.

    Args:
        df (pd.DataFrame): DataFrame containing 'content_chunks' and other metadata.
    """
    if df is None or "content_chunks" not in df.columns:
        logging.error("❌ Error: DataFrame is None or missing 'content_chunks' column.")
        return
    
    try:
        # Ensure embeddings are generated
        df = generate_article_embeddings(df)

        if "embedding" not in df.columns:
            logging.error("❌ Error: Embeddings were not generated.")
            return
        
        logging.debug(f"Generated embeddings:\n{df[['content_text', 'embedding']].head()}")  # Debugging

        points = []
        for i, row in df.iterrows():
            publication_datetime = row.get("publication_datetime", "Unknown")
            an_number = row.get("an", "Unknown")

            # Convert NaT values to None, otherwise store as a string
            if pd.isna(publication_datetime) or publication_datetime == "NaT":
                publication_datetime = "Unknown"
            else:
                publication_datetime = str(publication_datetime)

            for chunk, embedding in zip(row["content_chunks"], row["embedding"]):
                vector_id = abs(hash(chunk)) % (10**12)  # Prevent negative IDs
                
                # ✅ Ensure embeddings are **always** a flat list
                embedding = np.array(embedding).flatten().tolist()

                # ✅ Validate embedding type
                if not isinstance(embedding, list) or not all(isinstance(x, float) for x in embedding):
                    logging.error(f"❌ Invalid embedding format for chunk: {chunk}")
                    continue  # Skip inserting this vector
                
                logging.debug(f"Inserting vector ID {vector_id}: {embedding[:5]}...")  # Log first 5 values
                
                points.append(
                    PointStruct(
                        id=vector_id, 
                        vector=embedding, 
                        payload={
                            "an": an_number, 
                            "content_text": chunk,
                            "publication_datetime": publication_datetime  
                        }
                    )
                )
        
        if points:
            logging.info(f"Inserting {len(points)} vectors into Qdrant.")
            client.upsert(COLLECTION_NAME, points)
            logging.info("✅ Data inserted successfully.")
        else:
            logging.warning("⚠️ No valid vectors to insert.")
            
    except Exception as e:
        logging.error(f"❌ Error inserting vectors: {e}")

def search_vectors(query_vector, top_k=5):
    """
    Searches for similar articles using the query embedding.

    Args:
        query_vector (list): The query embedding vector.
        top_k (int): Number of results to retrieve.

    Returns:
        list: Retrieved documents with similarity scores.
    """
    try:
        results = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=top_k,
            with_payload=True
        )
        logging.info(f"Search returned {len(results)} results.")
        return results
    except Exception as e:
        logging.error(f"❌ Error searching vectors: {e}")
        return None
