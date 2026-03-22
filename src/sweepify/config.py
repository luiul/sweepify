import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

SPOTIFY_CLIENT_ID = os.getenv("SPOTIPY_CLIENT_ID", "")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIPY_CLIENT_SECRET", "")
SPOTIFY_REDIRECT_URI = os.getenv("SPOTIPY_REDIRECT_URI", "http://127.0.0.1:8888/callback")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
AWS_REGION = os.getenv("AWS_REGION", "eu-west-1")
LLM_PROVIDER = os.getenv("SWEEPIFY_LLM_PROVIDER", "bedrock")

DB_DIR = Path(os.getenv("SWEEPIFY_DB_DIR", Path.home() / ".sweepify"))
DB_PATH = DB_DIR / "sweepify.db"

PLAYLIST_PREFIX = os.getenv("SWEEPIFY_PLAYLIST_PREFIX", "sweepify:")
