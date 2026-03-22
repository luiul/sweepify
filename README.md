# mapa

Classify your Spotify Liked Songs into playlists using AI.

mapa fetches your liked songs, sends their metadata to Claude for classification, and creates Spotify playlists based on the categories it discovers (genre, mood, energy, etc.).

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)
- A [Spotify Developer](https://developer.spotify.com/) account
- An [Anthropic API](https://console.anthropic.com/) key

## Setup

### 1. Clone and install

```bash
git clone https://github.com/luiul/mapa.git
cd mapa
uv sync
```

### 2. Create a Spotify app

1. Go to the [Spotify Developer Dashboard](https://developer.spotify.com/dashboard)
2. Click **Create App**
3. Set the **Redirect URI** to `http://localhost:8888/callback`
4. Note the **Client ID** and **Client Secret**

### 3. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` with your credentials:

```
SPOTIPY_CLIENT_ID=your_spotify_client_id
SPOTIPY_CLIENT_SECRET=your_spotify_client_secret
SPOTIPY_REDIRECT_URI=http://localhost:8888/callback
ANTHROPIC_API_KEY=your_anthropic_api_key
```

## Usage

### Full pipeline

```bash
uv run mapa run
```

This runs all three steps in sequence: fetch, classify, and create playlists.

### Individual commands

```bash
uv run mapa fetch      # Fetch liked songs from Spotify
uv run mapa classify   # Classify songs using Claude
uv run mapa create     # Create Spotify playlists
uv run mapa status     # Show song and classification counts
uv run mapa reset      # Clear classifications (keeps songs)
```

### First run

On your first run, a browser window will open for Spotify authorization. Grant the requested permissions and you'll be redirected back. The auth token is cached locally for subsequent runs.

## Configuration

| Variable | Default | Description |
|---|---|---|
| `SPOTIPY_CLIENT_ID` | | Spotify app client ID |
| `SPOTIPY_CLIENT_SECRET` | | Spotify app client secret |
| `SPOTIPY_REDIRECT_URI` | `http://localhost:8888/callback` | OAuth redirect URI |
| `ANTHROPIC_API_KEY` | | Anthropic API key |
| `MAPA_PLAYLIST_PREFIX` | `mapa:` | Prefix for created playlist names |
| `MAPA_DB_DIR` | `~/.mapa` | Directory for the local SQLite database |

## How it works

1. **Fetch** — Retrieves all Liked Songs via the Spotify API with pagination, enriches them with artist genre data, and stores everything in a local SQLite database.
2. **Classify** — Sends unclassified songs (in batches of ~200) to Claude, which groups them into 5-15 categories based on genre, mood, and thematic coherence. Categories stay consistent across batches.
3. **Create** — Creates private Spotify playlists for each category and adds the songs. Re-running is safe: existing playlists are reused, and already-classified songs are skipped.

## Development

```bash
uv run pytest           # Run tests
uv run ruff check .     # Lint
uv run ruff format .    # Format
```
