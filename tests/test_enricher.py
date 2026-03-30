import json
from unittest.mock import MagicMock

from sweepify.enricher import (
    EnrichmentResult,
    SongEnrichment,
    _format_songs_for_prompt,
    enrich_songs,
)
from sweepify.models import Song


def _make_song(id: str, name: str = "Song", artist: str = "Artist") -> Song:
    return Song(spotify_id=id, name=name, artist=artist, album="Album", genres='["rock"]')


def _mock_claude_response(enrichments: list[dict]) -> MagicMock:
    """Create a mock Anthropic message response."""
    result = {"songs": enrichments}
    content_block = MagicMock()
    content_block.text = json.dumps(result)
    response = MagicMock()
    response.content = [content_block]
    response.stop_reason = "end_turn"
    return response


def test_enrich_songs_single_batch():
    client = MagicMock()
    client.messages.create.return_value = _mock_claude_response([
        {
            "spotify_id": "t1",
            "mood": "chill",
            "bpm": 90,
            "vibe": "late night drive",
            "related_artists": ["Artist B", "Artist C"],
        },
        {
            "spotify_id": "t2",
            "mood": "euphoric",
            "bpm": 128,
            "vibe": "festival anthem",
            "related_artists": ["Artist D"],
        },
    ])

    songs = [_make_song("t1"), _make_song("t2")]
    result = enrich_songs(client, songs)

    assert len(result.songs) == 2
    assert result.songs[0].spotify_id == "t1"
    assert result.songs[0].mood == "chill"
    assert result.songs[0].bpm == 90
    assert result.songs[0].vibe == "late night drive"
    assert result.songs[0].related_artists == ["Artist B", "Artist C"]
    assert result.songs[1].mood == "euphoric"
    assert result.songs[1].bpm == 128


def test_enrich_songs_with_code_fences():
    content_block = MagicMock()
    content_block.text = '```json\n{"songs": [{"spotify_id": "t1", "mood": "dark", "bpm": 70, "vibe": "midnight walk", "related_artists": ["X"]}]}\n```'
    response = MagicMock()
    response.content = [content_block]
    response.stop_reason = "end_turn"

    client = MagicMock()
    client.messages.create.return_value = response

    songs = [_make_song("t1")]
    result = enrich_songs(client, songs)

    assert len(result.songs) == 1
    assert result.songs[0].mood == "dark"


def test_format_songs_for_prompt():
    songs = [
        Song(
            spotify_id="t1", name="My Song", artist="The Band", album="The Album",
            genres='["rock", "indie"]', release_date="2023-05-01",
        ),
    ]
    prompt = _format_songs_for_prompt(songs)

    assert "t1" in prompt
    assert "My Song" in prompt
    assert "The Band" in prompt
    assert "released: 2023-05-01" in prompt


def test_enrich_songs_callbacks():
    client = MagicMock()
    client.messages.create.return_value = _mock_claude_response([
        {
            "spotify_id": "t1",
            "mood": "chill",
            "bpm": 90,
            "vibe": "rainy day",
            "related_artists": ["A"],
        },
    ])

    progress_calls = []
    batch_done_calls = []

    def on_progress(batch, total, size):
        progress_calls.append((batch, total, size))

    def on_batch_done(batch_num, result):
        batch_done_calls.append(result)

    songs = [_make_song("t1")]
    enrich_songs(client, songs, on_progress=on_progress, on_batch_done=on_batch_done)

    assert len(progress_calls) == 1
    assert progress_calls[0] == (1, 1, 1)
    assert len(batch_done_calls) == 1
    assert len(batch_done_calls[0].songs) == 1
