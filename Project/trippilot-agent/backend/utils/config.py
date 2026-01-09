from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # OpenAI
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-4o-mini"

    # Session storage (optional - uses in-memory if not set)
    MONGO_URI: str = ""
    MONGO_DB: str = "vacation_agent"

    # Long-term memory (optional - uses noop if not set)
    QDRANT_URL: str = "http://18.222.92.149:6333"
    QDRANT_COLLECTION: str = "user_prefs"

    # Click-out link short domain (optional)
    SHORTLINK_BASE: str = "https://go.example.com/r"

    # Google Calendar OAuth (optional)
    GOOGLE_OAUTH_CLIENT_JSON_PATH: str = "client_secret.json"
    GOOGLE_OAUTH_TOKEN_JSON_PATH: str = "google_token.json"
    GOOGLE_CALENDAR_ID: str = "primary"
    GOOGLE_TIMEZONE: str = "America/Chicago"

    # CORS
    CORS_ALLOW_ORIGINS: List[str] = ["*"]
    CORS_ALLOW_METHODS: List[str] = ["*"]
    CORS_ALLOW_HEADERS: List[str] = ["*"]
    CORS_ALLOW_CREDENTIALS: bool = True

settings = Settings()
