import re
import emoji
import unicodedata
import pandas as pd
import numpy as np


class AmharicPreprocessor:
    """Comprehensive preprocessing pipeline for Amharic Telegram data"""

    def __init__(self):
        # Initialize patterns and rules
        self.currency_pattern = re.compile(r"(\d+)(ብር)", re.IGNORECASE)
        self.price_pattern = re.compile(
            r"(ዋጋ|በ|ብር|br|birr|price)\s*[:]?\s*(\d[\d,.]*)", re.IGNORECASE
        )
        self.location_keywords = [
            "ቦታ",
            "አድራሻ",
            "location",
            "place",
            "address",
            "ወደ",
            "ከ",
            "በ",
        ]

    def normalize_amharic(self, text):
        """Handle Amharic-specific linguistic normalization"""
        if not text or not isinstance(text, str):
            return ""

        # Unicode normalization
        text = unicodedata.normalize("NFC", text)

        # Standardize currency
        text = self.currency_pattern.sub(r"\1 ብር", text)

        # Handle abbreviations
        replacements = {
            r"ሜትር": "ሜትር",
            r"ኪ\.ሜ\.": "ኪሎሜትር",
            r"ኪ\.ግ\.": "ኪሎግራም",
            r"ሴ\.ሜ\.": "ሴንቲሜትር",
            r"ፒ\.ሲ\.": "ፒሲ",
            r"ኤም\.": "ኤም",
            r"ቲ\.ቪ\.": "ቲቪ",
        }
        for pattern, replacement in replacements.items():
            text = re.sub(pattern, replacement, text)

        return text

    def clean_text(self, text):
        """Clean text while preserving Amharic content"""
        if not text or not isinstance(text, str):
            return ""

        # Remove emojis and URLs
        text = emoji.replace_emoji(text, replace="")
        text = re.sub(r"http\S+|www\S+|https\S+", "", text)
        text = re.sub(r"@\w+", "", text)
        text = re.sub(r"(\+251|0)?9\d{8}\b", "", text)
        text = re.sub(r"[^\w\s\u1200-\u137F.,!?;:ብር/]", "", text)
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def tokenize_amharic(self, text):
        """Linguistically-aware tokenization"""
        if not text:
            return []

        text = re.sub(r"(\d+)(ብር)", r"\1 \2", text)
        return re.findall(r"[\w\u1200-\u137F']+|[.,!?;:ብር/]", text)

    def extract_features(self, text):
        """Extract Amharic-specific features"""
        features = {
            "contains_price": 0,
            "contains_location": 0,
            "contains_product": 0,
            "price_value": None,
            "location_mentioned": None,
        }

        if not text:
            return features

        # Price detection
        price_match = self.price_pattern.search(text)
        if price_match:
            features["contains_price"] = 1
            try:
                features["price_value"] = float(price_match.group(2).replace(",", ""))
            except:
                pass

        # Location detection
        if any(keyword in text for keyword in self.location_keywords):
            features["contains_location"] = 1
            location_candidates = re.findall(r"([\u1200-\u137F]{3,})", text)
            features["location_mentioned"] = (
                location_candidates[:3] if location_candidates else None
            )

        # Product detection
        if re.search(r"(ሽያጭ|ይገኛል|ተሸጧል|ዋጋ|ገዢ)", text):
            features["contains_product"] = 1

        return features

    def preprocess_message(self, message):
        """Full preprocessing pipeline for a message"""
        # Metadata extraction
        metadata = {
            "message_id": message.get("id"),
            "channel": message.get("channel"),
            "timestamp": message.get("date"),
            "views": message.get("views", 0),
            "media_path": message.get("media"),
            "original_length": len(message.get("text", "")),
        }

        # Text processing
        raw_text = message.get("text", "")
        normalized_text = self.normalize_amharic(raw_text)
        cleaned_text = self.clean_text(normalized_text)
        tokens = self.tokenize_amharic(cleaned_text)

        # Feature extraction
        features = self.extract_features(cleaned_text)

        # Content data
        content = {
            "raw_text": raw_text,
            "normalized_text": normalized_text,
            "cleaned_text": cleaned_text,
            "tokens": tokens,
            "token_count": len(tokens),
        }

        return {**metadata, **content, **features}
