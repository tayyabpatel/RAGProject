# RAG Project

## Overview
This project implements a **Retrieval-Augmented Generation (RAG) System** using:
- **Google Compute Engine VM** for deployment
- **FastAPI** for serving API endpoints
- **Qdrant Vector Database** for semantic search
- **JinaAI Embeddings** for high-quality text representation
- **FastAvro & Pandas** for processing AVRO data
- **Docker & Docker Compose** for containerization

This README covers:
1. **How to run the project on Google VM via SSH**
2. **Example API calls with expected responses**
3. **Data processing of AVRO files**
4. **How embeddings work and why we use JinaAI**
5. **How Qdrant vector database works & why we use it**
6. **Query embedding for relevant article retrieval**

---

## 1️⃣ Running the Project via SSH on Google VM

To deploy and run the RAG system on a Google Compute Engine (GCE) VM, follow these steps:

### **Step 1: SSH into the VM**
```bash
ssh -i /path/to/private-key username@your-vm-external-ip
```

### **Step 2: Navigate to the Project Directory**
```bash
cd /opt/RAGProject
```

### **Step 3: Build and Start the Docker Containers**
```bash
docker compose down && docker compose up -d --build
```

### **Step 4: Verify that the Containers are Running**
```bash
docker ps
```
Expected output:
```
CONTAINER ID   IMAGE               STATUS          PORTS
abcdef123456   ragproject-app      Up 2 minutes   0.0.0.0:8000->8000/tcp
xyz789654321   qdrant/qdrant       Up 2 minutes   0.0.0.0:6333->6333/tcp
```

### **Step 5: Initialize the Qdrant Collection**
```bash
docker exec -it rag_api python3 -c 'from vector_database import create_collection; create_collection()'
```

Your project is now running!

---

## 2️⃣ Example API Calls

### **Search Endpoint**
#### **Request**
```bash
curl -X 'POST' 'http://YOUR_VM_IP:8000/search/' \
-H 'Content-Type: application/json' \
-d '{"query": "Latest AI developments"}'
```

#### **Expected Response**
```json
{
  "results": [
    {
      "title": "AI Breakthroughs in 2025",
      "content": "Scientists have developed...",
      "similarity_score": 0.92
    },
    {
      "title": "Deep Learning Advances",
      "content": "A new deep learning framework...",
      "similarity_score": 0.87
    }
  ]
}
```

---

## 3️⃣ Data Processing: AVRO to DataFrame

We process AVRO files using **FastAvro** and **Pandas**:

### **Step 1: Load AVRO File**
```python
import fastavro
import pandas as pd

def load_avro_to_dataframe(file_path):
    with open(file_path, "rb") as f:
        reader = fastavro.reader(f)
        records = [record for record in reader]
    return pd.DataFrame(records)
```

### **Step 2: Preprocess the Data**
- Convert timestamps
- Merge text fields (snippet + body)
- Split text into 700-word chunks

```python
def preprocess_dataframe(df):
    df["publication_datetime"] = pd.to_datetime(df["publication_datetime"], unit="ms", errors="coerce")
    df["content_text"] = df["snippet"].fillna("") + " " + df["body"].fillna("")
    df["content_chunks"] = df["content_text"].apply(lambda x: x.split(" ")[:700])
    return df
```

---

## 4️⃣ How Embeddings Work & Why JinaAI

We use **JinaAI Embeddings** for:
- **High-quality text representation**
- **Efficient similarity search**
- **Support for multilingual queries**

### **Generating Embeddings**
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("jinaai/jina-embeddings-v3", trust_remote_code=True)

def generate_article_embeddings(df):
    df["embedding"] = df["content_chunks"].apply(lambda x: model.encode(x, convert_to_tensor=True).tolist())
    return df
```

---

## 5️⃣ How Qdrant Works & Why We Use It

Qdrant is a **high-performance vector search engine** with:
- **Fast nearest-neighbor search**
- **Support for filtering and payload metadata**
- **Multilingual support**

### **Qdrant Schema**
```python
from qdrant_client.models import Distance, VectorParams

def create_collection():
    client.recreate_collection(
        collection_name="news_articles",
        vectors_config={"embedding": VectorParams(size=1024, distance=Distance.COSINE)}
    )
```

---

## 6️⃣ Query Embedding & Retrieval

When a user searches, we:
1. **Embed the query**
2. **Find similar embeddings in Qdrant**
3. **Return the most relevant articles**

```python
def generate_query_embedding(query):
    return model.encode(query, convert_to_tensor=True).tolist()

def search_vectors(query_vector, top_k=5):
    return client.search(
        collection_name="news_articles",
        query_vector=query_vector,
        limit=top_k,
        with_payload=True
    )
```

---

## Why Use a RAG System?

A **Retrieval-Augmented Generation (RAG) System**:
- **Improves Accuracy**: Retrieves relevant data to ground AI responses.
- **Reduces Hallucinations**: Ensures AI bases responses on real information.
- **Enhances Search**: Uses semantic similarity instead of simple keyword matching.
- **Multilingual Support**: Works across multiple languages using Qdrant & JinaAI.

---

## Summary
- **Deploy on Google VM using SSH** ✅
- **Run the project via Docker Compose** ✅
- **Preprocess AVRO data & generate embeddings** ✅
- **Use Qdrant for fast vector search** ✅
- **Retrieve the most relevant articles based on semantic similarity** ✅

🚀 **Enjoy your RAG-powered search system!**







