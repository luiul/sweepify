import json
import typing

from pydantic import BaseModel

from sweepify.classifier import BEDROCK_MODEL, DIRECT_MODEL, get_client
from sweepify.config import LLM_PROVIDER
from sweepify.models import Song

BATCH_SIZE = 50

_SYSTEM_PROMPT = """\
You are a music metadata analyst. Given a list of songs with their Spotify metadata, \
analyze each song and provide enriched metadata.

For each song, determine:
- mood: The dominant emotional tone. Must be one of: melancholic, euphoric, chill, aggressive, \
romantic, nostalgic, empowering, dark, playful, serene
- bpm: Your best estimate of the tempo in beats per minute as an integer (e.g. 72, 120, 140)
- vibe: A short evocative phrase (2-4 words) describing the feeling, \
e.g. "late night drive", "sunday morning coffee", "gym pump"
- related_artists: A list of 3-5 artists that fans of this song's artist would also enjoy. \
Use well-known artist names. Do not include the song's own artist(s).

Return a JSON object with this exact structure:
{
  "songs": [
    {
      "spotify_id": "the_original_id",
      "mood": "chill",
      "bpm": 95,
      "vibe": "late night drive",
      "related_artists": ["Artist A", "Artist B", "Artist C"]
    }
  ]
}

Important:
- Return ALL song IDs from the input — do not skip any
- Use ONLY the allowed values listed above for mood
- BPM should be a reasonable integer estimate based on your knowledge of the song
- related_artists must be a JSON array of artist name strings
- Return ONLY the JSON object, no other text"""


class SongEnrichment(BaseModel):
    spotify_id: str
    mood: str
    bpm: int
    vibe: str
    related_artists: list[str]


class EnrichmentResult(BaseModel):
    songs: list[SongEnrichment]


def _format_songs_for_prompt(songs: list[Song]) -> str:
    lines = []
    for s in songs:
        genres = s.genres or "unknown"
        parts = [
            f"id={s.spotify_id}",
            s.name,
            s.artist,
            s.album or "unknown",
            f"genres: {genres}",
        ]
        if s.popularity is not None:
            parts.append(f"popularity: {s.popularity}")
        if s.release_date:
            parts.append(f"released: {s.release_date}")
        lines.append("- " + " | ".join(parts))
    return "\n".join(lines)


MAX_WORKERS = 4


def enrich_songs(
    client: typing.Any,
    songs: list[Song],
    on_progress: typing.Callable[[int, int, int], None] | None = None,
    on_batch_done: typing.Callable[[EnrichmentResult], None] | None = None,
    max_workers: int = MAX_WORKERS,
) -> EnrichmentResult:
    """Enrich songs with AI-generated metadata. Handles batching for large collections.

    Batches are processed concurrently (up to max_workers at a time) for speed.
    on_batch_done is called as each batch completes, allowing incremental persistence.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    batches = [songs[i : i + BATCH_SIZE] for i in range(0, len(songs), BATCH_SIZE)]
    total_batches = len(batches)
    all_enrichments: list[SongEnrichment] = []
    lock = __import__("threading").Lock()

    if on_progress:
        for batch_num, batch in enumerate(batches, 1):
            on_progress(batch_num, total_batches, len(batch))

    def _process_batch(batch: list[Song]) -> EnrichmentResult:
        return _enrich_batch(client, batch)

    with ThreadPoolExecutor(max_workers=min(max_workers, total_batches)) as pool:
        futures = {pool.submit(_process_batch, batch): i for i, batch in enumerate(batches)}
        for future in as_completed(futures):
            result = future.result()
            with lock:
                all_enrichments.extend(result.songs)
            if on_batch_done:
                on_batch_done(result)

    return EnrichmentResult(songs=all_enrichments)


def _enrich_batch(client: typing.Any, songs: list[Song]) -> EnrichmentResult:
    song_list = _format_songs_for_prompt(songs)
    user_prompt = f"Enrich these songs:\n{song_list}"

    response = client.messages.create(
        model=BEDROCK_MODEL if LLM_PROVIDER == "bedrock" else DIRECT_MODEL,
        max_tokens=16384,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )

    if response.stop_reason == "max_tokens":
        raise RuntimeError(
            f"Claude response was truncated (batch of {len(songs)} songs). "
            "Try reducing BATCH_SIZE."
        )

    text = response.content[0].text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        text = text[start : end + 1]

    parsed = json.loads(text)
    return EnrichmentResult.model_validate(parsed)
