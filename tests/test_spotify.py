from unittest.mock import MagicMock

from mapa.spotify import fetch_liked_songs


def _make_track(track_id: str, name: str, artist_name: str, artist_id: str) -> dict:
    return {
        "track": {
            "id": track_id,
            "name": name,
            "artists": [{"name": artist_name, "id": artist_id}],
            "album": {"name": "Test Album"},
        },
        "added_at": "2025-01-01T00:00:00Z",
    }


def _make_artist(artist_id: str, genres: list[str]) -> dict:
    return {"id": artist_id, "genres": genres}


def test_fetch_liked_songs_single_page():
    sp = MagicMock()
    sp.current_user_saved_tracks.return_value = {
        "items": [
            _make_track("t1", "Song One", "Artist A", "a1"),
            _make_track("t2", "Song Two", "Artist B", "a2"),
        ],
        "next": None,
    }
    sp.artists.return_value = {
        "artists": [
            _make_artist("a1", ["rock", "indie"]),
            _make_artist("a2", ["pop"]),
        ]
    }

    songs = fetch_liked_songs(sp)

    assert len(songs) == 2
    assert songs[0].spotify_id == "t1"
    assert songs[0].name == "Song One"
    assert songs[0].artist == "Artist A"
    assert songs[0].album == "Test Album"
    assert "rock" in songs[0].genres
    assert songs[1].spotify_id == "t2"


def test_fetch_liked_songs_pagination():
    sp = MagicMock()
    sp.current_user_saved_tracks.side_effect = [
        {
            "items": [_make_track("t1", "Song One", "Artist A", "a1")],
            "next": "http://next-page",
        },
        {
            "items": [_make_track("t2", "Song Two", "Artist B", "a2")],
            "next": None,
        },
    ]
    sp.artists.return_value = {
        "artists": [
            _make_artist("a1", []),
            _make_artist("a2", []),
        ]
    }

    songs = fetch_liked_songs(sp)

    assert len(songs) == 2
    assert sp.current_user_saved_tracks.call_count == 2


def test_fetch_liked_songs_empty():
    sp = MagicMock()
    sp.current_user_saved_tracks.return_value = {"items": [], "next": None}

    songs = fetch_liked_songs(sp)

    assert songs == []


def test_genres_enrichment():
    sp = MagicMock()
    sp.current_user_saved_tracks.return_value = {
        "items": [
            _make_track("t1", "Collab", "Artist A", "a1"),
        ],
        "next": None,
    }
    # Add a second artist to the track
    sp.current_user_saved_tracks.return_value["items"][0]["track"]["artists"].append(
        {"name": "Artist B", "id": "a2"}
    )
    sp.artists.return_value = {
        "artists": [
            _make_artist("a1", ["rock"]),
            _make_artist("a2", ["electronic", "rock"]),
        ]
    }

    songs = fetch_liked_songs(sp)

    assert songs[0].artist == "Artist A, Artist B"
    assert "rock" in songs[0].genres
    assert "electronic" in songs[0].genres
