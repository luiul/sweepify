import json
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
    assert "categories" in cols
    assert "playlist_ids" in cols


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


def test_mark_classified_single_category():
    db.upsert_songs([_make_song("1")])
    db.mark_classified(["1"], "Chill", "pl_123")

    songs = db.get_all_songs()
    assert songs[0].classified is True
    cats = json.loads(songs[0].categories)
    pids = json.loads(songs[0].playlist_ids)
    assert cats == ["Chill"]
    assert pids == {"Chill": "pl_123"}


def test_mark_classified_multiple_categories():
    db.upsert_songs([_make_song("1")])
    db.mark_classified(["1"], "Chill", "")
    db.mark_classified(["1"], "Late Night", "")
    db.mark_classified(["1"], "Chill", "pl_123")  # Add playlist_id for Chill

    songs = db.get_all_songs()
    cats = json.loads(songs[0].categories)
    pids = json.loads(songs[0].playlist_ids)
    assert sorted(cats) == ["Chill", "Late Night"]
    assert pids["Chill"] == "pl_123"
    assert "Late Night" not in pids  # No playlist_id yet


def test_mark_classified_no_duplicate_categories():
    db.upsert_songs([_make_song("1")])
    db.mark_classified(["1"], "Rock", "")
    db.mark_classified(["1"], "Rock", "pl_1")  # Same category, now with playlist_id

    songs = db.get_all_songs()
    cats = json.loads(songs[0].categories)
    assert cats == ["Rock"]  # Not duplicated


def test_get_songs_by_category():
    db.upsert_songs([_make_song("1"), _make_song("2")])
    db.mark_classified(["1"], "Rock", "")
    db.mark_classified(["1"], "Chill", "")
    db.mark_classified(["2"], "Rock", "")

    by_cat = db.get_songs_by_category()
    assert "Rock" in by_cat
    assert "Chill" in by_cat
    assert len(by_cat["Rock"]) == 2
    assert len(by_cat["Chill"]) == 1


def test_get_songs_by_category_excludes_fulfilled():
    db.upsert_songs([_make_song("1")])
    db.mark_classified(["1"], "Rock", "")
    db.mark_classified(["1"], "Chill", "")

    # Fulfill Rock
    db.mark_classified(["1"], "Rock", "pl_1")

    by_cat = db.get_songs_by_category()
    assert "Rock" not in by_cat  # Already has playlist_id
    assert "Chill" in by_cat


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
    db.mark_classified(["1"], "Chill", "")
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
    assert s["categories"] == 2  # Rock and Chill


def test_reset_classifications():
    db.upsert_songs([_make_song("1"), _make_song("2")])
    db.mark_classified(["1", "2"], "Pop", "pl_1")
    db.upsert_playlist(Playlist(spotify_id="pl_1", name="sweepify: Pop"))

    count = db.reset_classifications()
    assert count == 2

    songs = db.get_all_songs()
    assert all(not s.classified for s in songs)
    assert all(s.categories is None for s in songs)
    assert len(db.get_playlists()) == 0
