
#Instead of every file reading .env or calling os.getenv(...) on its own, config.py loads .env once and exposes all settings through a single settings object.
import os
from pathlib import Path

from dotenv import load_dotenv

#Finds the project root and loads .env
BASE_DIR = Path(__file__).resolve().parent.parent

# Load environment variables from the root .env file
load_dotenv(BASE_DIR / ".env")


# Application configuration
class Settings:

    APP_NAME: str = os.getenv(
        "APP_NAME",
        "Helpdesk RAG Service"
    )

    ENVIRONMENT: str = os.getenv(
        "ENVIRONMENT",
        "development"
    )

    HOST: str = os.getenv(
        "HOST",
        "127.0.0.1"
    )

    PORT: int = int(os.getenv("PORT") or "8000")

    # Database configuration
    DATABASE_URL: str = os.getenv("DATABASE_URL", "")

    # Gemini embedding configuration
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL") or "gemini-embedding-001"
    EMBEDDING_DIMENSIONS: int = 768

    # LLM configuration
    LLM_MODEL: str = os.getenv("LLM_MODEL") or "gemini-3.5-flash-lite"
    LLM_API_KEY: str = os.getenv("LLM_API_KEY") or os.getenv("GEMINI_API_KEY", "")
    INTERNAL_API_SECRET: str = os.getenv("INTERNAL_API_SECRET", "")


# Create a single settings object
settings = Settings()
