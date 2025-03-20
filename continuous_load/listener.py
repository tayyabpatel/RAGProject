from dnaStreaming.listener import Listener
import logging

def callback(message, subscription_id):
    """Handles incoming messages from Factiva stream."""
    logging.info(f"📥 Received Message from Subscription {subscription_id}: {message}")
    return True  # Returning False would stop the listener.

def start_streaming():
    """Starts the streaming listener using `dnaStreaming`."""
    try:
        logging.info("🔗 Connecting to Factiva Stream...")
        listener = Listener(config_file="customer_config.json")
        listener.listen(callback)
    except Exception as e:
        logging.error(f"❌ Error in Factiva Streaming: {e}")

if __name__ == "__main__":
    start_streaming()

