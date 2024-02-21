import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    DATABASE_URL = os.getenv("DATABASE_URL")
    SECRET_KEY = os.getenv("SECRET_KEY")
    AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
    ESG_API_KEY = os.getenv("ESG_API_KEY")
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")  # Default to INFO if not set

config = Config()