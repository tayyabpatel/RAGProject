import logging
from listener import start_streaming

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def main():
    """Starts the Factiva streaming module."""
    logging.info("🚀 Starting Factiva Streaming Module...")
    start_streaming()

if __name__ == "__main__":
    main()
