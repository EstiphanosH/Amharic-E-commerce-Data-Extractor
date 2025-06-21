import os
import asyncio
import json
import re
import configparser
from datetime import datetime
from telethon import TelegramClient
from telethon.tl.types import MessageMediaPhoto, MessageMediaDocument
import pandas as pd
import logging


class TelegramScraper:
    """Core Telegram scraping functionality"""

    def __init__(self, config_path="credentials.ini"):
        self.config_path = config_path
        self.api_id = None
        self.api_hash = None
        self.phone = None
        self.data_dir = "../data/raw/telegram_data"  # Default, can be overridden
        self._load_config()
        self._setup_logging()

    def _setup_logging(self):
        """Configure logging"""
        self.logger = logging.getLogger("TelegramScraper")
        self.logger.setLevel(logging.INFO)

        # Create logs directory if needed
        log_dir = os.path.join(self.data_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)

        # File handler
        log_file = os.path.join(log_dir, "scraper.log")
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    def _load_config(self):
        """Load and validate configuration"""
        config = configparser.ConfigParser()

        if not os.path.exists(self.config_path):
            raise FileNotFoundError(
                f"Config file not found at: {self.config_path}\n"
                "Create file with:\n[Telegram]\n"
                "api_id = YOUR_API_ID\n"
                "api_hash = YOUR_API_HASH\n"
                "phone = YOUR_PHONE\n"
            )

        try:
            config.read(self.config_path)
            self.api_id = int(config["Telegram"]["api_id"])
            self.api_hash = config["Telegram"]["api_hash"]
            self.phone = config["Telegram"]["phone"]
            self.logger.info("Credentials loaded successfully")
        except (KeyError, ValueError) as e:
            self.logger.error(f"Configuration error: {str(e)}")
            raise ValueError(f"Configuration error: {str(e)}") from e

    async def start_client(self):
        """Initialize Telegram client"""
        try:
            self.client = TelegramClient(
                session=str(self.phone), api_id=self.api_id, api_hash=self.api_hash
            )
            await self.client.start(phone=self.phone)
            self.logger.info("Client initialized successfully")
            return self.client
        except Exception as e:
            self.logger.error(f"Client initialization failed: {str(e)}")
            raise

    async def _process_message(self, message):
        """Extract structured data from message"""
        # Handle media files
        media_path = None
        media_type = None
        if message.media:
            try:
                media_path, media_type = await self._download_media(message)
            except Exception as e:
                self.logger.error(f"Media download error: {str(e)}")

        # Process reactions
        reactions = []
        if getattr(message, "reactions", None):
            for r in message.reactions.results:
                try:
                    emoticon = (
                        r.reaction.emoticon
                        if hasattr(r.reaction, "emoticon")
                        else str(r.reaction)
                    )
                    reactions.append({"emoticon": emoticon, "count": r.count})
                except AttributeError:
                    self.logger.warning(
                        f"Unexpected reaction format in message {message.id}"
                    )

        # Get channel ID safely
        channel_id = None
        try:
            if hasattr(message.peer_id, "channel_id"):
                channel_id = message.peer_id.channel_id
        except AttributeError:
            pass

        return {
            "id": message.id,
            "text": message.text or "",
            "date": message.date.isoformat() if message.date else None,
            "views": getattr(message, "views", None),
            "forwards": getattr(message, "forwards", None),
            "sender_id": getattr(message, "sender_id", None),
            "media_path": media_path,
            "media_type": media_type,
            "reactions": reactions,
            "url": f"https://t.me/c/{channel_id}/{message.id}" if channel_id else None,
        }

    async def _download_media(self, message):
        """Download media from message"""
        media_dir = os.path.join(self.data_dir, "media")
        os.makedirs(media_dir, exist_ok=True)

        # Create safe filename
        timestamp = int(message.date.timestamp()) if message.date else 0
        filename = f"{message.id}_{timestamp}"

        if isinstance(message.media, MessageMediaPhoto):
            filename += ".jpg"
            path = os.path.join(media_dir, filename)
            await message.download_media(file=path)
            return path, "photo"

        elif isinstance(message.media, MessageMediaDocument):
            doc = message.media.document
            ext = doc.mime_type.split("/")[-1] if doc.mime_type else "bin"
            filename += f".{ext}"
            path = os.path.join(media_dir, filename)
            await message.download_media(file=path)
            return path, "document"

        return None, None

    async def scrape_channel(self, channel_handle, limit=100):
        """Scrape messages from a channel"""
        self.logger.info(f"Starting scrape: {channel_handle}")
        channel_data = []

        try:
            entity = await self.client.get_entity(channel_handle)
            # Sanitize channel name for directory
            channel_name = re.sub(r"[^\w\-_]", "", entity.title.replace(" ", "_"))
            channel_dir = os.path.join(self.data_dir, channel_name)
            os.makedirs(channel_dir, exist_ok=True)

            async for message in self.client.iter_messages(entity, limit=limit):
                try:
                    processed = await self._process_message(message)
                    channel_data.append(processed)
                except Exception as e:
                    self.logger.error(
                        f"Error processing message {message.id}: {str(e)}"
                    )

            # Save data
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            json_path = os.path.join(channel_dir, f"{channel_name}_{timestamp}.json")
            csv_path = os.path.join(channel_dir, f"{channel_name}_{timestamp}.csv")

            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(channel_data, f, ensure_ascii=False, indent=2)

            if channel_data:
                pd.DataFrame(channel_data).to_csv(
                    csv_path, index=False, encoding="utf-8"
                )

            self.logger.info(f"Saved {len(channel_data)} messages to {channel_dir}")
            return channel_data

        except Exception as e:
            self.logger.error(f"Channel scraping failed: {str(e)}")
            return None

    async def close(self):
        """Cleanup client connection"""
        if self.client and self.client.is_connected():
            await self.client.disconnect()
            self.logger.info("Client disconnected")
