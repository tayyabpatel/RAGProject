import json
import os

def load_config():
    """Load configuration from JSON file."""
    config_path = os.getenv("CONFIG_PATH", "config.json")
    with open(config_path, "r") as file:
        return json.load(file)
