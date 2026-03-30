# sweepify

AI-powered curator for your Spotify music — sweep your songs into playlists.

sweepify fetches your liked songs (or any playlist), sends their metadata to Claude for classification, and creates Spotify playlists based on the categories it discovers (genre, mood, energy, etc.).

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)
- A [Spotify Developer](https://developer.spotify.com/) account
- An [Anthropic API](https://console.anthropic.com/) key or AWS Bedrock access

## Setup

### 1. Clone and install

```bash
git clone https://github.com/luiul/sweepify.git
cd sweepify
uv sync
```

### 2. Create a Spotify app

1. Go to the [Spotify Developer Dashboard](https://developer.spotify.com/dashboard)
2. Click **Create App**
3. Set the **Redirect URI** to `http://127.0.0.1:8888/callback`
4. Note the **Client ID** and **Client Secret**

### 3. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` with your credentials:

```shell
SPOTIPY_CLIENT_ID=your_spotify_client_id
SPOTIPY_CLIENT_SECRET=your_spotify_client_secret
SPOTIPY_REDIRECT_URI=http://127.0.0.1:8888/callback
SWEEPIFY_LLM_PROVIDER=bedrock   # or "anthropic"
ANTHROPIC_API_KEY=               # only if using anthropic provider
```

## Usage

### Full pipeline

```bash
uv run sweepify run
```

This runs all five steps in sequence: fetch, enrich, classify, refine, and create playlists.

### Target a specific playlist

```bash
uv run sweepify run -p "My Playlist"
```

### Individual commands

```bash
uv run sweepify fetch      # Fetch liked songs from Spotify
uv run sweepify enrich     # Add AI metadata (mood, BPM, vibe, related artists)
uv run sweepify classify   # Rough-classify songs in parallel
uv run sweepify refine     # Consolidate rough categories into final playlists
uv run sweepify create     # Create Spotify playlists
uv run sweepify status     # Show song and classification counts
uv run sweepify reset      # Clear classifications (keeps songs)
uv run sweepify clear      # Remove sweepify playlists from Spotify + reset
```

### Genre-based playlists

Create playlists from songs matching specific genres:

```bash
uv run sweepify playlist -g "lo-fi house, minimal techno"             # Claude picks the name
uv run sweepify playlist -g "corrido, banda" --name "Corridos Mix"    # custom name
uv run sweepify playlist -g "latin, reggaeton" -n 3                   # split into 3 playlists
```

Use `-n 0` with `run` or `classify` to classify songs only into existing sweepify playlists:

```bash
uv run sweepify run -n 0
```

### Database explorer

Browse your songs, explore genre breakdowns, build playlist commands, and run ad-hoc SQL queries in an interactive Streamlit UI:

```bash
uv run --group ui sweepify ui
```

The Playlist Builder tab lets you select genres, preview matching songs, and generates the CLI command to run.

### First run

On your first run, a browser window will open for Spotify authorization. Grant the requested permissions and you'll be redirected back. The auth token is cached locally for subsequent runs.

## Configuration

| Variable                   | Default                          | Description                                       |
| -------------------------- | -------------------------------- | ------------------------------------------------- |
| `SPOTIPY_CLIENT_ID`        |                                  | Spotify app client ID                             |
| `SPOTIPY_CLIENT_SECRET`    |                                  | Spotify app client secret                         |
| `SPOTIPY_REDIRECT_URI`     | `http://127.0.0.1:8888/callback` | OAuth redirect URI                                |
| `SWEEPIFY_LLM_PROVIDER`    | `bedrock`                        | LLM provider: `bedrock` or `anthropic`            |
| `ANTHROPIC_API_KEY`        |                                  | Anthropic API key (only for `anthropic` provider) |
| `AWS_REGION`               | `eu-west-1`                      | AWS region (only for `bedrock` provider)          |
| `SWEEPIFY_PLAYLIST_PREFIX` | `sweepify:`                      | Prefix for created playlist names                 |
| `SWEEPIFY_DB_DIR`          | `~/.sweepify`                    | Directory for the local SQLite database           |

## How it works

1. **Fetch** — Retrieves all Liked Songs (or a specific playlist) via the Spotify API with pagination, enriches them with artist genre data, and stores everything in a local SQLite database.
2. **Enrich** — Sends unenriched songs (in parallel batches of ~50) to Claude to generate metadata: mood, BPM estimate, vibe phrase, and related artists. Up to 4 batches run concurrently. Progress is saved after each batch.
3. **Classify** — Rough-classifies songs in parallel batches (see below). Songs can appear in up to 4 playlists. Results are saved immediately.
4. **Refine** — Consolidates rough categories into coherent final playlists (see below). Can be re-run independently if interrupted.
5. **Create** — Creates private Spotify playlists for each category and adds the songs. Re-running is safe: existing playlists are reused, and already-classified songs are skipped.

### Classification pipeline

Classification uses a two-step approach to balance speed and quality:

**`classify`** — Songs are split into batches of ~100 and all batches are sent to Claude concurrently (up to 4 workers). Each batch independently produces its own categories. Results are saved to the database immediately, so progress is preserved if interrupted. Overlapping or inconsistent category names across batches are expected at this stage.

**`refine`** — All rough categories are sent in one API call to Claude, which returns a mapping of which rough categories to merge and what to rename them. Song reassignment happens locally (no song IDs are sent to Claude), so this step is fast. Can be re-run independently if it fails.

Short-circuits:

- **<=100 songs**: `classify` produces final categories in a single API call. `refine` is not needed.
- **Fixed categories** (`-n 0`): `classify` batches run in parallel, assigning to predetermined categories. `refine` is not needed.

## Development

```bash
uv run pytest           # Run tests
uv run ruff check .     # Lint
uv run ruff format .    # Format
uv sync --group ui      # Install UI dependencies (Streamlit, Plotly)
```
