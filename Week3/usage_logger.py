
import os
import time
from datetime import datetime, timezone

from pymongo import MongoClient
from pymongo.errors import PyMongoError


class UsageLogger:

    def __init__(self):
        mongodb_uri = os.getenv("MONGODB_URI")

        if not mongodb_uri:
            raise ValueError("MONGODB_URI is missing from .env")

        self.client = MongoClient(
            mongodb_uri,
            serverSelectionTimeoutMS=5000,
            connectTimeoutMS=5000,
            socketTimeoutMS=10000,
            retryWrites=True
        )

        self.client.admin.command("ping")

        db = self.client["llm_monitoring"]
        self.collection = db["api_usage"]

        print("Connected to MongoDB successfully!")

    def log_call(self, record):
        document = record.copy()
        document["timestamp"] = datetime.now(timezone.utc)

        for attempt in range(3):
            try:
                self.collection.insert_one(document)
                print("Usage record saved to MongoDB!")
                return

            except PyMongoError as error:
                print(f"MongoDB logging attempt {attempt + 1} failed: {error}")

                if attempt < 2:
                    time.sleep(2 ** attempt)

        print("Warning: Could not save usage record.")

    def close(self):
        self.client.close()
        print("MongoDB connection closed.")
