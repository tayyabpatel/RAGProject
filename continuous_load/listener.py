import asyncio
import websockets
import logging

class StreamListener:
    def __init__(self, stream_url, auth_token):
        self.stream_url = stream_url
        self.auth_token = auth_token

    async def connect(self):
        """Connect to the WebSocket stream."""
        async with websockets.connect(self.stream_url, extra_headers={"Authorization": f"Bearer {self.auth_token}"}) as websocket:
            logging.info("Connected to stream.")
            await self.receive_messages(websocket)

    async def receive_messages(self, websocket):
        """Listen for incoming messages."""
        try:
            async for message in websocket:
                logging.info(f"Received message: {message}")
                self.process_message(message)
        except Exception as e:
            logging.error(f"Error receiving messages: {e}")

    def process_message(self, message):
        """Process incoming messages (Modify as needed)."""
        logging.info(f"Processing message: {message}")

    def listen(self):
        """Start listening to the stream."""
        asyncio.run(self.connect())
