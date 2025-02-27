import os
import time
import requests
import subprocess

# Define services
SERVICES = ["database", "qdrant_db", "embeddings_service", "query_api"]

def run_command(command):
    """Runs a shell command and returns output."""
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    return result.stdout, result.stderr

def start_containers():
    """Starts all services using Docker Compose."""
    print("🚀 Starting all containers...")
    stdout, stderr = run_command("docker compose up -d --build")
    if stderr:
        print(f"⚠️ Warning: {stderr}")
    print(stdout)

def check_containers():
    """Checks if all containers are running."""
    print("🔍 Checking container status...")
    stdout, _ = run_command("docker ps --format '{{.Names}}'")
    running_containers = stdout.split("\n")

    missing_services = [s for s in SERVICES if s not in running_containers]
    if missing_services:
        print(f"❌ Some containers are missing: {missing_services}")
        exit(1)
    print("✅ All required containers are running.")

def wait_for_qdrant():
    """Waits until Qdrant is ready."""
    print("⏳ Waiting for Qdrant to be ready...")
    for _ in range(10):
        try:
            response = requests.get("http://localhost:6333/collections")
            if response.status_code == 200:
                print("✅ Qdrant is ready!")
                return
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(5)
    print("❌ Qdrant did not start in time.")
    exit(1)

def trigger_initial_load():
    """Triggers the Initial Load API to process and store news data."""
    print("📤 Sending Initial Load request...")
    payload = {"avro_file_path": "/app/data/news_data100.avro"}
    response = requests.post("http://localhost:5001/process_and_store/", json=payload)

    if response.status_code == 200:
        print(f"✅ Initial Load successful: {response.json()}")
    else:
        print(f"❌ Initial Load failed: {response.status_code}, {response.text}")
        exit(1)

def verify_data_stored():
    """Queries Qdrant to check if embeddings have been stored."""
    print("🔍 Verifying stored embeddings...")
    query = {
        "query_vector": [0.0] * 1024,  # Dummy vector
        "top_k": 3
    }
    response = requests.post("http://localhost:6000/search_vectors/", json=query)

    if response.status_code == 200:
        results = response.json().get("results", [])
        if results:
            print(f"✅ Found {len(results)} stored articles in Qdrant!")
        else:
            print("⚠️ No results found in Qdrant.")
    else:
        print(f"❌ Qdrant query failed: {response.status_code}, {response.text}")
        exit(1)

if __name__ == "__main__":
    start_containers()
    check_containers()
    wait_for_qdrant()
    trigger_initial_load()
    verify_data_stored()
    print("🎉 System setup complete!")
