import sqlite3

from mapa.config import DB_DIR, DB_PATH
from mapa.models import Playlist, Song

SCHEMA = """
CREATE TABLE IF NOT EXISTS songs (
    spotify_id   TEXT PRIMARY KEY,
    name         TEXT NOT NULL,
    artist       TEXT NOT NULL,
    album        TEXT,
    genres       TEXT,
    added_at     TEXT,
    energy       REAL,
    valence      REAL,
    tempo        REAL,
    danceability REAL,
    classified   INTEGER DEFAULT 0,
    playlist_id  TEXT,
    category     TEXT,
    fetched_at   TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS playlists (
    spotify_id   TEXT PRIMARY KEY,
    name         TEXT NOT NULL,
    created_at   TEXT DEFAULT CURRENT_TIMESTAMP
);
"""


def _ensure_db_dir() -> None:
    DB_DIR.mkdir(parents=True, exist_ok=True)


def get_connection() -> sqlite3.Connection:
    _ensure_db_dir()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with get_connection() as conn:
        conn.executescript(SCHEMA)


def upsert_songs(songs: list[Song]) -> int:
    """Insert songs, ignoring duplicates. Returns number of new songs added."""
    with get_connection() as conn:
        cursor = conn.executemany(
            """
            INSERT OR IGNORE INTO songs
                (spotify_id, name, artist, album, genres, added_at,
                 energy, valence, tempo, danceability)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    s.spotify_id,
                    s.name,
                    s.artist,
                    s.album,
                    s.genres,
                    s.added_at,
                    s.energy,
                    s.valence,
                    s.tempo,
                    s.danceability,
                )
                for s in songs
            ],
        )
        return cursor.rowcount


def get_unclassified_songs() -> list[Song]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM songs WHERE classified = 0",
        ).fetchall()
        return [_row_to_song(r) for r in rows]


def get_all_songs() -> list[Song]:
    with get_connection() as conn:
        rows = conn.execute("SELECT * FROM songs").fetchall()
        return [_row_to_song(r) for r in rows]


def mark_classified(song_ids: list[str], category: str, playlist_id: str) -> None:
    with get_connection() as conn:
        conn.executemany(
            """
            UPDATE songs
            SET classified = 1, category = ?, playlist_id = ?
            WHERE spotify_id = ?
            """,
            [(category, playlist_id, sid) for sid in song_ids],
        )


def upsert_playlist(playlist: Playlist) -> None:
    with get_connection() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO playlists (spotify_id, name)
            VALUES (?, ?)
            """,
            (playlist.spotify_id, playlist.name),
        )


def get_playlists() -> list[Playlist]:
    with get_connection() as conn:
        rows = conn.execute("SELECT * FROM playlists").fetchall()
        return [Playlist.model_validate(dict(r)) for r in rows]


def get_status() -> dict[str, int]:
    with get_connection() as conn:
        total = conn.execute("SELECT COUNT(*) FROM songs").fetchone()[0]
        classified = conn.execute(
            "SELECT COUNT(*) FROM songs WHERE classified = 1",
        ).fetchone()[0]
        playlists = conn.execute("SELECT COUNT(*) FROM playlists").fetchone()[0]
        categories = conn.execute(
            "SELECT COUNT(DISTINCT category) FROM songs WHERE category IS NOT NULL",
        ).fetchone()[0]
    return {
        "total": total,
        "classified": classified,
        "unclassified": total - classified,
        "playlists": playlists,
        "categories": categories,
    }


def reset_classifications() -> int:
    """Clear all classification data. Returns number of songs reset."""
    with get_connection() as conn:
        cursor = conn.execute(
            "UPDATE songs SET classified = 0, category = NULL, playlist_id = NULL WHERE classified = 1",
        )
        conn.execute("DELETE FROM playlists")
        return cursor.rowcount


def _row_to_song(row: sqlite3.Row) -> Song:
    return Song.model_validate(dict(row))
