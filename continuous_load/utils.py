import json

def load_config(file="customer_config.json"):
    """Loads the JSON configuration file."""
    with open(file, "r") as f:
        return json.load(f)

