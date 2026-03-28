import json
from unittest.mock import MagicMock

from sweepify.classifier import (
    ClassificationResult,
    _build_user_prompt,
    _format_songs_for_prompt,
    _merge_categories,
    Category,
    classify_songs,
)
from sweepify.models import Song


def _make_song(id: str, name: str = "Song", artist: str = "Artist") -> Song:
    return Song(spotify_id=id, name=name, artist=artist, album="Album", genres='["rock"]')


def _mock_claude_response(categories: list[dict]) -> MagicMock:
    """Create a mock Anthropic message response."""
    result = {"categories": categories}
    content_block = MagicMock()
    content_block.text = json.dumps(result)
    response = MagicMock()
    response.content = [content_block]
    return response


def test_classify_songs_single_batch():
    client = MagicMock()
    client.messages.create.return_value = _mock_claude_response([
        {"name": "Rock Anthems", "description": "Classic rock", "song_ids": ["t1", "t2"]},
        {"name": "Chill Vibes", "description": "Relaxing", "song_ids": ["t3"]},
    ])

    songs = [_make_song("t1"), _make_song("t2"), _make_song("t3")]
    result = classify_songs(client, songs)

    assert len(result.categories) == 2
    assert result.categories[0].name == "Rock Anthems"
    assert result.categories[0].song_ids == ["t1", "t2"]
    assert result.categories[1].song_ids == ["t3"]


def test_classify_songs_with_code_fences():
    """Claude sometimes wraps JSON in markdown code fences."""
    content_block = MagicMock()
    content_block.text = '```json\n{"categories": [{"name": "Pop", "description": "Pop music", "song_ids": ["t1"]}]}\n```'
    response = MagicMock()
    response.content = [content_block]

    client = MagicMock()
    client.messages.create.return_value = response

    songs = [_make_song("t1")]
    result = classify_songs(client, songs)

    assert len(result.categories) == 1
    assert result.categories[0].name == "Pop"


def test_merge_categories():
    existing = [Category(name="Rock", description="Rock music", song_ids=["t1"])]
    new = [
        Category(name="Rock", description="Rock music", song_ids=["t2"]),
        Category(name="Jazz", description="Jazz music", song_ids=["t3"]),
    ]

    merged = _merge_categories(existing, new)

    assert len(merged) == 2
    rock = next(c for c in merged if c.name == "Rock")
    assert rock.song_ids == ["t1", "t2"]
    jazz = next(c for c in merged if c.name == "Jazz")
    assert jazz.song_ids == ["t3"]


def test_build_user_prompt_first_batch():
    songs = [_make_song("t1", "My Song", "The Band")]
    prompt = _build_user_prompt(songs, existing_categories=None)

    assert "Classify these songs" in prompt
    assert "t1" in prompt
    assert "My Song" in prompt
    assert "The Band" in prompt


def test_format_songs_includes_enriched_data():
    song = Song(
        spotify_id="t1", name="Song", artist="Artist", album="Album",
        genres='["rock"]', enriched=True, mood="chill", bpm=95,
        vibe="late night drive", related_artists='["Artist B", "Artist C"]',
    )
    prompt = _format_songs_for_prompt([song])

    assert "mood: chill" in prompt
    assert "bpm: 95" in prompt
    assert "vibe: late night drive" in prompt
    assert "related:" in prompt
    assert "Artist B" in prompt


def test_format_songs_includes_audio_features():
    song = Song(
        spotify_id="t1", name="Song", artist="Artist", album="Album",
        genres='["rock"]', tempo=120.0, energy=0.8, danceability=0.7, valence=0.6,
    )
    prompt = _format_songs_for_prompt([song])

    assert "tempo: 120" in prompt
    assert "energy: 0.80" in prompt
    assert "dance: 0.70" in prompt
    assert "valence: 0.60" in prompt


def test_format_songs_excludes_enrichment_when_not_enriched():
    song = _make_song("t1")
    prompt = _format_songs_for_prompt([song])

    assert "mood:" not in prompt
    assert "bpm:" not in prompt
    assert "tempo:" not in prompt


def test_build_user_prompt_followup_batch():
    songs = [_make_song("t2")]
    existing = [Category(name="Rock", description="Rock songs", song_ids=["t1"])]
    prompt = _build_user_prompt(songs, existing_categories=existing)

    assert "Existing categories" in prompt
    assert "Rock" in prompt
    assert "t2" in prompt
