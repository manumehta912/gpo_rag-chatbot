import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).parent
load_dotenv(BASE_DIR / ".env")

API_KEY = os.getenv("GEMINI_API_KEY")
CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-3.5-flash-lite")
EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001")

PDF_PATH = BASE_DIR / "data" / "GPO-Anual-report.pdf"
INDEX_DIR = BASE_DIR / "index"

CHUNK_SIZE = 1000        # characters
CHUNK_OVERLAP = 150      # characters
EMBED_DIM = 768
TOP_K = 5
