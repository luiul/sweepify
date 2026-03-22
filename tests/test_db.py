import os

import pytest

# Use in-memory-like temp DB for tests
os.environ["MAPA_DB_DIR"] = "/tmp/mapa_test"

from mapa import db
from mapa.models import Playlist, Song


@pytest.fixture(autouse=True)
def fresh_db(tmp_path):
    """Use a fresh DB for each test."""
    os.environ["MAPA_DB_DIR"] = str(tmp_path)
    # Re-read config since DB_DIR/DB_PATH are set at import time
    db.DB_DIR = tmp_path
    db.DB_PATH = tmp_path / "mapa.db"
    db.init_db()
    yield


def _make_song(id: str = "1", name: str = "Test Song", artist: str = "Artist") -> Song:
    return Song(spotify_id=id, name=name, artist=artist)


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
    db.upsert_playlist(Playlist(spotify_id="pl_1", name="mapa: Chill"))
    db.upsert_playlist(Playlist(spotify_id="pl_2", name="mapa: Rock"))

    playlists = db.get_playlists()
    assert len(playlists) == 2
    names = {p.name for p in playlists}
    assert "mapa: Chill" in names


def test_get_status():
    db.upsert_songs([_make_song("1"), _make_song("2"), _make_song("3")])
    db.mark_classified(["1"], "Rock", "pl_1")
    db.upsert_playlist(Playlist(spotify_id="pl_1", name="mapa: Rock"))

    s = db.get_status()
    assert s["total"] == 3
    assert s["classified"] == 1
    assert s["unclassified"] == 2
    assert s["playlists"] == 1
    assert s["categories"] == 1


def test_reset_classifications():
    db.upsert_songs([_make_song("1"), _make_song("2")])
    db.mark_classified(["1", "2"], "Pop", "pl_1")
    db.upsert_playlist(Playlist(spotify_id="pl_1", name="mapa: Pop"))

    count = db.reset_classifications()
    assert count == 2

    songs = db.get_all_songs()
    assert all(not s.classified for s in songs)
    assert all(s.category is None for s in songs)
    assert len(db.get_playlists()) == 0
