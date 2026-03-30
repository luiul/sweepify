import json
from unittest.mock import MagicMock

from sweepify.classifier import (
    ClassificationResult,
    _build_user_prompt,
    _format_songs_for_prompt,
    _merge_categories,
    _refine_categories,
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
    response.stop_reason = "end_turn"
    return response


def test_classify_songs_single_batch():
    """Single batch: no refinement, no threading."""
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
    # Single batch = single API call (no refinement)
    assert client.messages.create.call_count == 1


def test_classify_songs_single_batch_callbacks():
    """Single batch fires on_batch_done and on_refine_done but not on_refine_start."""
    client = MagicMock()
    client.messages.create.return_value = _mock_claude_response([
        {"name": "Pop", "description": "Pop music", "song_ids": ["t1"]},
    ])

    batch_done_calls = []
    refine_start_calls = []
    refine_done_calls = []

    songs = [_make_song("t1")]
    classify_songs(
        client, songs,
        on_batch_done=lambda bn, r: batch_done_calls.append((bn, r)),
        on_refine_start=lambda: refine_start_calls.append(True),
        on_refine_done=lambda r: refine_done_calls.append(r),
    )

    assert len(batch_done_calls) == 1
    assert batch_done_calls[0][0] == 1  # batch_num
    assert len(refine_start_calls) == 0  # no refinement for single batch
    assert len(refine_done_calls) == 1  # but refine_done still fires with the result


def test_classify_songs_with_code_fences():
    """Claude sometimes wraps JSON in markdown code fences."""
    content_block = MagicMock()
    content_block.text = '```json\n{"categories": [{"name": "Pop", "description": "Pop music", "song_ids": ["t1"]}]}\n```'
    response = MagicMock()
    response.content = [content_block]
    response.stop_reason = "end_turn"

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


def test_format_songs_excludes_enrichment_when_not_enriched():
    song = _make_song("t1")
    prompt = _format_songs_for_prompt([song])

    assert "mood:" not in prompt
    assert "bpm:" not in prompt


def test_classify_songs_multi_category():
    """Songs can appear in multiple categories."""
    client = MagicMock()
    client.messages.create.return_value = _mock_claude_response([
        {"name": "Rock Anthems", "description": "Classic rock", "song_ids": ["t1", "t2"]},
        {"name": "Chill Vibes", "description": "Relaxing", "song_ids": ["t1", "t3"]},
    ])

    songs = [_make_song("t1"), _make_song("t2"), _make_song("t3")]
    result = classify_songs(client, songs)

    assert len(result.categories) == 2
    assert "t1" in result.categories[0].song_ids  # t1 in Rock
    assert "t1" in result.categories[1].song_ids  # t1 in Chill too


def test_build_user_prompt_followup_batch():
    songs = [_make_song("t2")]
    existing = [Category(name="Rock", description="Rock songs", song_ids=["t1"])]
    prompt = _build_user_prompt(songs, existing_categories=existing)

    assert "Existing categories" in prompt
    assert "Rock" in prompt
    assert "t2" in prompt


def test_classify_songs_parallel_with_refinement():
    """Multi-batch triggers parallel rough pass + refinement call."""
    # Need 2+ batches: BATCH_SIZE is 100, so 150 songs = 2 batches
    songs = [_make_song(f"t{i}") for i in range(150)]

    # Batch 1 response (first 100 songs)
    batch1_response = _mock_claude_response([
        {"name": "Rock A", "description": "Rock batch 1", "song_ids": [f"t{i}" for i in range(50)]},
        {"name": "Chill A", "description": "Chill batch 1", "song_ids": [f"t{i}" for i in range(50, 100)]},
    ])
    # Batch 2 response (next 50 songs)
    batch2_response = _mock_claude_response([
        {"name": "Rock B", "description": "Rock batch 2", "song_ids": [f"t{i}" for i in range(100, 130)]},
        {"name": "Jazz", "description": "Jazz batch 2", "song_ids": [f"t{i}" for i in range(130, 150)]},
    ])
    # Refinement response (consolidates all)
    refine_response = _mock_claude_response([
        {"name": "Rock", "description": "All rock", "song_ids": [f"t{i}" for i in range(50)] + [f"t{i}" for i in range(100, 130)]},
        {"name": "Chill", "description": "Chill vibes", "song_ids": [f"t{i}" for i in range(50, 100)]},
        {"name": "Jazz", "description": "Jazz tunes", "song_ids": [f"t{i}" for i in range(130, 150)]},
    ])

    client = MagicMock()
    client.messages.create.side_effect = [batch1_response, batch2_response, refine_response]

    batch_start_calls = []
    batch_done_calls = []
    refine_start_called = []
    refine_done_calls = []

    result = classify_songs(
        client, songs,
        on_batch_start=lambda bn, t, s: batch_start_calls.append(bn),
        on_batch_done=lambda bn, r: batch_done_calls.append(bn),
        on_refine_start=lambda: refine_start_called.append(True),
        on_refine_done=lambda r: refine_done_calls.append(r),
    )

    # 2 batch calls + 1 refinement call = 3 API calls
    assert client.messages.create.call_count == 3
    assert len(batch_start_calls) == 2
    assert len(batch_done_calls) == 2
    assert len(refine_start_called) == 1
    assert len(refine_done_calls) == 1
    # Final result is from refinement
    assert len(result.categories) == 3
    assert {c.name for c in result.categories} == {"Rock", "Chill", "Jazz"}


def test_fixed_categories_parallel_no_refinement():
    """Fixed categories mode parallelizes but skips refinement."""
    songs = [_make_song(f"t{i}") for i in range(150)]

    batch_response = _mock_claude_response([
        {"name": "Rock", "description": "Rock", "song_ids": [f"t{i}" for i in range(75)]},
        {"name": "Pop", "description": "Pop", "song_ids": [f"t{i}" for i in range(75, 100)]},
    ])
    batch2_response = _mock_claude_response([
        {"name": "Rock", "description": "Rock", "song_ids": [f"t{i}" for i in range(100, 130)]},
        {"name": "Pop", "description": "Pop", "song_ids": [f"t{i}" for i in range(130, 150)]},
    ])

    client = MagicMock()
    client.messages.create.side_effect = [batch_response, batch2_response]

    refine_start_called = []

    result = classify_songs(
        client, songs,
        fixed_categories=["Rock", "Pop"],
        on_refine_start=lambda: refine_start_called.append(True),
    )

    # 2 batch calls, no refinement call
    assert client.messages.create.call_count == 2
    assert len(refine_start_called) == 0
    # Categories merged by name
    rock = next(c for c in result.categories if c.name == "Rock")
    assert len(rock.song_ids) == 75 + 30  # from both batches


def test_refine_categories():
    """Unit test for the refinement function."""
    refined_response = _mock_claude_response([
        {"name": "Rock Anthems", "description": "All rock", "song_ids": ["t1", "t2", "t3"]},
        {"name": "Chill", "description": "Relaxing", "song_ids": ["t4", "t5"]},
    ])

    client = MagicMock()
    client.messages.create.return_value = refined_response

    rough = [
        Category(name="Rock A", description="Rock batch 1", song_ids=["t1", "t2"]),
        Category(name="Rock B", description="Rock batch 2", song_ids=["t3"]),
        Category(name="Chill Vibes", description="Relaxing", song_ids=["t4", "t5"]),
    ]

    result = _refine_categories(client, rough, max_playlists=5)

    assert len(result.categories) == 2
    assert result.categories[0].name == "Rock Anthems"
    assert result.categories[0].song_ids == ["t1", "t2", "t3"]
