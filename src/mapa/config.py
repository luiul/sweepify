import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

SPOTIFY_CLIENT_ID = os.getenv("SPOTIPY_CLIENT_ID", "")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIPY_CLIENT_SECRET", "")
SPOTIFY_REDIRECT_URI = os.getenv("SPOTIPY_REDIRECT_URI", "http://localhost:8888/callback")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

DB_DIR = Path(os.getenv("MAPA_DB_DIR", Path.home() / ".mapa"))
DB_PATH = DB_DIR / "mapa.db"

PLAYLIST_PREFIX = os.getenv("MAPA_PLAYLIST_PREFIX", "mapa:")
