import os

import pytest

# Use in-memory-like temp DB for tests
os.environ["SWEEPIFY_DB_DIR"] = "/tmp/sweepify_test"

from sweepify import db
from sweepify.models import Playlist, Song


@pytest.fixture(autouse=True)
def fresh_db(tmp_path):
    """Use a fresh DB for each test."""
    os.environ["SWEEPIFY_DB_DIR"] = str(tmp_path)
    # Re-read config since DB_DIR/DB_PATH are set at import time
    db.DB_DIR = tmp_path
    db.DB_PATH = tmp_path / "sweepify.db"
    db.init_db()
    yield


def _make_song(id: str = "1", name: str = "Test Song", artist: str = "Artist") -> Song:
    return Song(spotify_id=id, name=name, artist=artist)


def test_ensure_columns_adds_missing():
    """Verify _ensure_columns adds new columns to an existing table."""
    with db.get_connection() as conn:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(songs)").fetchall()}
    # All new columns should exist after init_db
    assert "mood" in cols
    assert "bpm" in cols
    assert "vibe" in cols
    assert "related_artists" in cols
    assert "enriched" in cols


def test_upsert_songs_inserts():
    songs = [_make_song("1"), _make_song("2")]
    count = db.upsert_songs(songs)
    assert count == 2
    assert len(db.get_all_songs()) == 2


def test_upsert_songs_ignores_duplicates():
    db.upsert_songs([_make_song("1")])
    count = db.upsert_songs([_make_song("1"), _make_song("2")])
    # SQLite INSERT OR IGNORE: rowcount may vary, but total should be 2
    assert len(db.get_all_songs()) == 2


def test_get_unclassified_songs():
    db.upsert_songs([_make_song("1"), _make_song("2")])
    assert len(db.get_unclassified_songs()) == 2

    db.mark_classified(["1"], "Rock", "playlist_abc")
    unclassified = db.get_unclassified_songs()
    assert len(unclassified) == 1
    assert unclassified[0].spotify_id == "2"


def test_mark_classified():
    db.upsert_songs([_make_song("1")])
    db.mark_classified(["1"], "Chill", "pl_123")

    songs = db.get_all_songs()
    assert songs[0].classified is True
    assert songs[0].category == "Chill"
    assert songs[0].playlist_id == "pl_123"


def test_upsert_and_get_playlists():
    db.upsert_playlist(Playlist(spotify_id="pl_1", name="sweepify: Chill"))
    db.upsert_playlist(Playlist(spotify_id="pl_2", name="sweepify: Rock"))

    playlists = db.get_playlists()
    assert len(playlists) == 2
    names = {p.name for p in playlists}
    assert "sweepify: Chill" in names


def test_get_unenriched_songs():
    db.upsert_songs([_make_song("1"), _make_song("2")])
    assert len(db.get_unenriched_songs()) == 2

    db.mark_enriched([{
        "spotify_id": "1", "mood": "chill", "bpm": 90,
        "vibe": "late night", "related_artists": '["X"]',
    }])
    unenriched = db.get_unenriched_songs()
    assert len(unenriched) == 1
    assert unenriched[0].spotify_id == "2"


def test_mark_enriched():
    db.upsert_songs([_make_song("1")])
    db.mark_enriched([{
        "spotify_id": "1", "mood": "euphoric", "bpm": 128,
        "vibe": "festival anthem", "related_artists": '["A", "B"]',
    }])

    songs = db.get_all_songs()
    assert songs[0].enriched is True
    assert songs[0].mood == "euphoric"
    assert songs[0].bpm == 128
    assert songs[0].vibe == "festival anthem"
    assert songs[0].related_artists == '["A", "B"]'


def test_reset_enrichments():
    db.upsert_songs([_make_song("1"), _make_song("2")])
    db.mark_enriched([
        {"spotify_id": "1", "mood": "chill", "bpm": 90, "vibe": "v", "related_artists": "[]"},
        {"spotify_id": "2", "mood": "dark", "bpm": 70, "vibe": "v", "related_artists": "[]"},
    ])

    count = db.reset_enrichments()
    assert count == 2

    songs = db.get_all_songs()
    assert all(not s.enriched for s in songs)
    assert all(s.mood is None for s in songs)
    assert all(s.bpm is None for s in songs)


def test_get_status():
    db.upsert_songs([_make_song("1"), _make_song("2"), _make_song("3")])
    db.mark_classified(["1"], "Rock", "pl_1")
    db.mark_enriched([{
        "spotify_id": "2", "mood": "chill", "bpm": 90,
        "vibe": "v", "related_artists": "[]",
    }])
    db.upsert_playlist(Playlist(spotify_id="pl_1", name="sweepify: Rock"))

    s = db.get_status()
    assert s["total"] == 3
    assert s["enriched"] == 1
    assert s["classified"] == 1
    assert s["unclassified"] == 2
    assert s["playlists"] == 1
    assert s["categories"] == 1


def test_reset_classifications():
    db.upsert_songs([_make_song("1"), _make_song("2")])
    db.mark_classified(["1", "2"], "Pop", "pl_1")
    db.upsert_playlist(Playlist(spotify_id="pl_1", name="sweepify: Pop"))

    count = db.reset_classifications()
    assert count == 2

    songs = db.get_all_songs()
    assert all(not s.classified for s in songs)
    assert all(s.category is None for s in songs)
    assert len(db.get_playlists()) == 0
