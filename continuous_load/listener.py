import os
import logging
from dnaStreaming.listener import Listener

# Load environment variables
USER_KEY = os.getenv("USER_KEY")
SUBSCRIPTION_ID = os.getenv("SUBSCRIPTION_ID")

if not USER_KEY or not SUBSCRIPTION_ID:
    raise ValueError("❌ Missing USER_KEY or SUBSCRIPTION_ID. Check your environment variables.")

logging.basicConfig(level=logging.INFO)
logging.info(f"🔑 USER_KEY loaded. 🔒")
logging.info(f"📡 SUBSCRIPTION_ID: {SUBSCRIPTION_ID}")

# Correct callback with 2 arguments
def on_message_callback(message, subscription_id):
    print(f"📰 Message from {subscription_id}:\n{message}")

# Start streaming
def start_streaming():
    listener = Listener(user_key=USER_KEY)
    listener.listen(
        on_message_callback=on_message_callback,
        subscription_id=SUBSCRIPTION_ID
    )
