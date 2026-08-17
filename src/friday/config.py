from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()

@dataclass(frozen=True)
class Settings:
    db_host: str = os.getenv("FRIDAY_DB_HOST", "localhost")
    db_port: int = int(os.getenv("FRIDAY_DB_PORT", "5432"))
    db_name: str = os.getenv("FRIDAY_DB_NAME", "Friday")
    db_user: str = os.getenv("FRIDAY_DB_USER", "friday_app")
    db_password: str = os.getenv("FRIDAY_DB_PASSWORD", "")
    ollama_base_url: str = os.getenv("FRIDAY_OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("FRIDAY_OLLAMA_MODEL", "qwen3:8b")
    ai_mode: str = os.getenv("FRIDAY_AI_MODE", "local")

settings = Settings()
