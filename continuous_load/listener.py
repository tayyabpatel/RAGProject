import logging
import requests
import time
import json
from utils import load_config

class StreamListener:
    def __init__(self, stream_url, auth_token, polling_interval, process_avro_api):
        self.stream_url = stream_url
        self.auth_token = auth_token
        self.polling_interval = polling_interval
        self.process_avro_api = process_avro_api
        self.headers = {"Authorization": f"Bearer {self.auth_token}"}

    def fetch_articles(self):
        """Fetch new articles from the streaming API using HTTP polling."""
        try:
            response = requests.get(self.stream_url, headers=self.headers)

            if response.status_code == 200:
                articles = response.json()
                for article in articles:
                    self.process_article(article)
            else:
                logging.error(f"Failed to fetch articles. Status Code: {response.status_code}, Response: {response.text}")

        except requests.RequestException as e:
            logging.error(f"Error fetching articles: {e}")

    def process_article(self, article):
        """Send the article to the Process AVRO Module."""
        try:
            response = requests.post(self.process_avro_api, json=article)

            if response.status_code == 200:
                logging.info("Article successfully sent to Process AVRO Module")
            else:
                logging.error(f"Failed to send article. Status: {response.status_code}, Response: {response.text}")

        except requests.RequestException as e:
            logging.error(f"Error sending article: {e}")

    def listen(self):
        """Continuously fetch articles at the specified polling interval."""
        while True:
            self.fetch_articles()
            time.sleep(self.polling_interval)

# Load config and start the listener
if __name__ == "__main__":
    config = load_config()
    listener = StreamListener(config["stream_url"], config["auth_token"], config["polling_interval"], config["process_avro_api"])
    listener.listen()
