import sqlite3

from mapa.config import DB_DIR, DB_PATH
from mapa.models import Playlist, Song, generate_create_table, get_insert_columns


def _ensure_db_dir() -> None:
    DB_DIR.mkdir(parents=True, exist_ok=True)


def get_connection() -> sqlite3.Connection:
    _ensure_db_dir()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with get_connection() as conn:
        conn.execute(generate_create_table(Song, "songs"))
        conn.execute(generate_create_table(Playlist, "playlists"))


# --- Songs ---

_SONG_INSERT_COLS = get_insert_columns(Song)
_SONG_INSERT_SQL = (
    f"INSERT OR IGNORE INTO songs ({', '.join(_SONG_INSERT_COLS)}) VALUES ({', '.join('?' for _ in _SONG_INSERT_COLS)})"
)


def upsert_songs(songs: list[Song]) -> int:
    """Insert songs, ignoring duplicates. Returns number of new songs added."""
    with get_connection() as conn:
        cursor = conn.executemany(
            _SONG_INSERT_SQL,
            [tuple(getattr(s, c) for c in _SONG_INSERT_COLS) for s in songs],
        )
        return cursor.rowcount


def get_unclassified_songs() -> list[Song]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM songs WHERE classified = 0",
        ).fetchall()
        return [Song.model_validate(dict(r)) for r in rows]


def get_all_songs() -> list[Song]:
    with get_connection() as conn:
        rows = conn.execute("SELECT * FROM songs").fetchall()
        return [Song.model_validate(dict(r)) for r in rows]


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


# --- Playlists ---

_PLAYLIST_INSERT_COLS = get_insert_columns(Playlist)
_PLAYLIST_INSERT_SQL = (
    f"INSERT OR REPLACE INTO playlists ({', '.join(_PLAYLIST_INSERT_COLS)}) "
    f"VALUES ({', '.join('?' for _ in _PLAYLIST_INSERT_COLS)})"
)


def upsert_playlist(playlist: Playlist) -> None:
    with get_connection() as conn:
        conn.execute(
            _PLAYLIST_INSERT_SQL,
            tuple(getattr(playlist, c) for c in _PLAYLIST_INSERT_COLS),
        )


def get_playlists() -> list[Playlist]:
    with get_connection() as conn:
        rows = conn.execute("SELECT * FROM playlists").fetchall()
        return [Playlist.model_validate(dict(r)) for r in rows]


def get_songs_by_category() -> dict[str, list[Song]]:
    """Get classified songs grouped by category, only those not yet added to a playlist."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM songs WHERE classified = 1 AND (playlist_id IS NULL OR playlist_id = '')",
        ).fetchall()
    by_cat: dict[str, list[Song]] = {}
    for r in rows:
        song = Song.model_validate(dict(r))
        by_cat.setdefault(song.category, []).append(song)
    return by_cat


def get_playlist_by_name(name: str) -> Playlist | None:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM playlists WHERE name = ?", (name,)
        ).fetchone()
        return Playlist.model_validate(dict(row)) if row else None


# --- Aggregates ---


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
