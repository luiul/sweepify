import json
import sqlite3

from sweepify.config import DB_DIR, DB_PATH
from sweepify.models import Playlist, Song, generate_create_table, get_insert_columns, _resolve_sqlite_type


def _ensure_db_dir() -> None:
    DB_DIR.mkdir(parents=True, exist_ok=True)


def get_connection() -> sqlite3.Connection:
    _ensure_db_dir()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _ensure_columns(conn: sqlite3.Connection, model: type, table_name: str) -> None:
    """Add any missing columns to an existing table based on the model."""
    existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table_name})").fetchall()}
    for name, info in model.model_fields.items():
        if name not in existing:
            col_type = _resolve_sqlite_type(info.annotation)
            default_clause = ""
            if info.default is not None and not info.is_required():
                if isinstance(info.default, bool):
                    default_clause = f" DEFAULT {int(info.default)}"
                elif isinstance(info.default, (int, float)):
                    default_clause = f" DEFAULT {info.default}"
            conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {name} {col_type}{default_clause}")


def _drop_removed_columns(conn: sqlite3.Connection, model: type, table_name: str) -> None:
    """Drop columns that exist in the table but are no longer in the model."""
    existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table_name})").fetchall()}
    model_fields = set(model.model_fields)
    for col in existing - model_fields:
        conn.execute(f"ALTER TABLE {table_name} DROP COLUMN {col}")


def _migrate_multi_category(conn: sqlite3.Connection) -> None:
    """Migrate from single category/playlist_id to multi-category categories/playlist_ids."""
    existing = {row[1] for row in conn.execute("PRAGMA table_info(songs)").fetchall()}
    if "category" not in existing:
        return
    # Migrate data: category -> categories JSON array, playlist_id -> playlist_ids JSON object
    rows = conn.execute(
        "SELECT spotify_id, category, playlist_id FROM songs WHERE category IS NOT NULL"
    ).fetchall()
    for row in rows:
        sid, category, playlist_id = row["spotify_id"], row["category"], row["playlist_id"]
        categories = json.dumps([category])
        playlist_ids = json.dumps({category: playlist_id}) if playlist_id else json.dumps({})
        conn.execute(
            "UPDATE songs SET categories = ?, playlist_ids = ? WHERE spotify_id = ?",
            (categories, playlist_ids, sid),
        )


def init_db() -> None:
    with get_connection() as conn:
        conn.execute(generate_create_table(Song, "songs"))
        conn.execute(generate_create_table(Playlist, "playlists"))
        _ensure_columns(conn, Song, "songs")
        _migrate_multi_category(conn)
        _drop_removed_columns(conn, Song, "songs")


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


def get_unenriched_songs() -> list[Song]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM songs WHERE enriched = 0",
        ).fetchall()
        return [Song.model_validate(dict(r)) for r in rows]


def get_all_songs() -> list[Song]:
    with get_connection() as conn:
        rows = conn.execute("SELECT * FROM songs").fetchall()
        return [Song.model_validate(dict(r)) for r in rows]


def mark_enriched(enrichments: list[dict]) -> None:
    """Update songs with AI-enriched metadata and set enriched = 1."""
    with get_connection() as conn:
        conn.executemany(
            """
            UPDATE songs
            SET mood = ?, bpm = ?, vibe = ?, related_artists = ?, enriched = 1
            WHERE spotify_id = ?
            """,
            [
                (e["mood"], e["bpm"], e["vibe"], e["related_artists"], e["spotify_id"])
                for e in enrichments
            ],
        )


def mark_classified(song_ids: list[str], category: str, playlist_id: str) -> None:
    """Add a category to each song's categories list and record the playlist_id mapping."""
    with get_connection() as conn:
        for sid in song_ids:
            row = conn.execute(
                "SELECT categories, playlist_ids FROM songs WHERE spotify_id = ?", (sid,)
            ).fetchone()
            if not row:
                continue

            # Parse existing arrays
            cats = json.loads(row["categories"]) if row["categories"] else []
            pids = json.loads(row["playlist_ids"]) if row["playlist_ids"] else {}

            # Append category if not already present
            if category not in cats:
                cats.append(category)

            # Record playlist_id mapping
            if playlist_id:
                pids[category] = playlist_id

            conn.execute(
                "UPDATE songs SET classified = 1, categories = ?, playlist_ids = ? WHERE spotify_id = ?",
                (json.dumps(cats), json.dumps(pids), sid),
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
    """Get classified songs grouped by category, only for categories not yet added to a playlist."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM songs WHERE classified = 1",
        ).fetchall()
    by_cat: dict[str, list[Song]] = {}
    for r in rows:
        song = Song.model_validate(dict(r))
        cats = json.loads(song.categories) if song.categories else []
        pids = json.loads(song.playlist_ids) if song.playlist_ids else {}
        # Include song in categories that don't have a playlist_id yet
        for cat in cats:
            pid = pids.get(cat)
            if not pid:
                by_cat.setdefault(cat, []).append(song)
    return by_cat


def get_playlist_by_name(name: str) -> Playlist | None:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM playlists WHERE name = ?", (name,)
        ).fetchone()
        return Playlist.model_validate(dict(row)) if row else None


# --- Genre queries ---


def get_songs_by_genres(genres: list[str]) -> list[Song]:
    """Get songs whose genres JSON array contains any of the given genres."""
    with get_connection() as conn:
        placeholders = " OR ".join("genres LIKE ?" for _ in genres)
        params = [f'%"{g}"%' for g in genres]
        rows = conn.execute(
            f"SELECT * FROM songs WHERE {placeholders}",  # noqa: S608
            params,
        ).fetchall()
        return [Song.model_validate(dict(r)) for r in rows]


# --- Aggregates ---


def get_status() -> dict[str, int]:
    with get_connection() as conn:
        total = conn.execute("SELECT COUNT(*) FROM songs").fetchone()[0]
        enriched = conn.execute(
            "SELECT COUNT(*) FROM songs WHERE enriched = 1",
        ).fetchone()[0]
        classified = conn.execute(
            "SELECT COUNT(*) FROM songs WHERE classified = 1",
        ).fetchone()[0]
        playlists = conn.execute("SELECT COUNT(*) FROM playlists").fetchone()[0]
        # Count distinct categories across all songs' JSON arrays
        rows = conn.execute(
            "SELECT categories FROM songs WHERE categories IS NOT NULL"
        ).fetchall()
    all_cats: set[str] = set()
    for r in rows:
        try:
            all_cats.update(json.loads(r["categories"]))
        except (json.JSONDecodeError, TypeError):
            pass
    return {
        "total": total,
        "enriched": enriched,
        "classified": classified,
        "unclassified": total - classified,
        "playlists": playlists,
        "categories": len(all_cats),
    }


def reset_classifications() -> int:
    """Clear all classification data. Returns number of songs reset."""
    with get_connection() as conn:
        cursor = conn.execute(
            "UPDATE songs SET classified = 0, categories = NULL, playlist_ids = NULL WHERE classified = 1",
        )
        conn.execute("DELETE FROM playlists")
        return cursor.rowcount


def reset_enrichments() -> int:
    """Clear all enrichment data. Returns number of songs reset."""
    with get_connection() as conn:
        cursor = conn.execute(
            "UPDATE songs SET enriched = 0, mood = NULL, bpm = NULL, vibe = NULL, related_artists = NULL "
            "WHERE enriched = 1",
        )
        return cursor.rowcount
