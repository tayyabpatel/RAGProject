import json
import logging
import time
from listener import StreamListener
from utils import load_config

# Load configuration
config = load_config()

# Set up logging
logging.basicConfig(level=logging.INFO, filename=config["log_file"], format="%(asctime)s - %(levelname)s - %(message)s")

def main():
    """Initialize and start the HTTP polling listener."""
    listener = StreamListener(config["stream_url"], config["auth_token"], config["polling_interval"], config["process_avro_api"])
    listener.listen()

if __name__ == "__main__":
    main()
