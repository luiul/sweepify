import json
import typing

import anthropic
from pydantic import BaseModel

from sweepify.config import ANTHROPIC_API_KEY, AWS_REGION, LLM_PROVIDER
from sweepify.models import Song

BATCH_SIZE = 100

_SYSTEM_PROMPT_TEMPLATE = """You are a music curator. Given a list of songs with metadata (name, artist, album, genres), \
group them into playlists. Create between 5 and {max_playlists} categories based on genre, mood, energy, \
and thematic coherence. Each song can appear in up to 4 categories if it fits naturally.

Return your response as a JSON object with this exact structure:
{{
  "categories": [
    {{
      "name": "Category Name",
      "description": "Brief description of what ties these songs together",
      "song_ids": ["spotify_id_1", "spotify_id_2"]
    }}
  ]
}}

Important:
- Every song_id from the input must appear in at least one category
- A song may appear in up to 4 categories if it genuinely fits multiple playlists
- Category names should be descriptive and evocative (e.g. "Late Night Drive", "Sunday Morning Coffee")
- Do not use generic names like "Category 1" or "Miscellaneous"
- Aim for balanced playlists — each category should have at least 5 songs
- Prefer fewer, well-populated playlists over many small ones. Merge niche categories into broader ones rather than creating categories with only 1-3 songs
- Return ONLY the JSON object, no other text"""

DEFAULT_MAX_PLAYLISTS = 10

FOLLOWUP_PROMPT_PREFIX = """Here are the existing categories from previous batches. \
Assign each song to up to 4 of these categories. Only create a new category if a song truly \
does not fit any existing one.

Existing categories:
"""

_FIXED_CATEGORIES_SYSTEM_PROMPT = """You are a music curator. Given a list of songs with metadata \
(name, artist, album, genres), assign each song to the provided categories. \
Do NOT create any new categories — every song must go into one of the existing ones.

Return your response as a JSON object with this exact structure:
{
  "categories": [
    {
      "name": "Category Name",
      "description": "Brief description of what ties these songs together",
      "song_ids": ["spotify_id_1", "spotify_id_2"]
    }
  ]
}

Important:
- Every song_id from the input must appear in at least one category
- A song may appear in up to 4 categories if it genuinely fits multiple playlists
- Use ONLY the provided category names — do not invent new ones
- Aim for balanced distribution — avoid leaving most songs in a single category
- Return ONLY the JSON object, no other text"""

_FIXED_CATEGORIES_PROMPT_PREFIX = """Use ONLY these categories:

"""


class Category(BaseModel):
    name: str
    description: str
    song_ids: list[str]


class ClassificationResult(BaseModel):
    categories: list[Category]


BEDROCK_MODEL = "eu.anthropic.claude-sonnet-4-20250514-v1:0"
DIRECT_MODEL = "claude-sonnet-4-20250514"


def get_client() -> anthropic.Anthropic | anthropic.AnthropicBedrock:
    if LLM_PROVIDER == "bedrock":
        import boto3

        session = boto3.Session()
        creds = session.get_credentials()
        if creds is None:
            raise RuntimeError(
                "No AWS credentials found. Run 'aws sso login' and ensure AWS_PROFILE is set."
            )
        frozen = creds.get_frozen_credentials()
        return anthropic.AnthropicBedrock(
            aws_region=AWS_REGION,
            aws_access_key=frozen.access_key,
            aws_secret_key=frozen.secret_key,
            aws_session_token=frozen.token,
        )
    return anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


def _format_songs_for_prompt(songs: list[Song]) -> str:
    lines = []
    for s in songs:
        genres = s.genres or "unknown"
        line = f"- id={s.spotify_id} | {s.name} | {s.artist} | {s.album} | genres: {genres}"
        # AI enrichment
        if s.enriched:
            line += f" | mood: {s.mood} | bpm: {s.bpm} | vibe: {s.vibe}"
            if s.related_artists:
                line += f" | related: {s.related_artists}"
        lines.append(line)
    return "\n".join(lines)


def _build_user_prompt(songs: list[Song], existing_categories: list[Category] | None) -> str:
    song_list = _format_songs_for_prompt(songs)

    if existing_categories:
        cat_summary = "\n".join(
            f"- {c.name}: {c.description}" for c in existing_categories
        )
        return f"{FOLLOWUP_PROMPT_PREFIX}{cat_summary}\n\nSongs to classify:\n{song_list}"

    return f"Classify these songs into playlists:\n{song_list}"


def _build_fixed_categories_prompt(songs: list[Song], categories: list[Category]) -> str:
    song_list = _format_songs_for_prompt(songs)
    cat_list = "\n".join(f"- {c.name}" for c in categories)
    return f"{_FIXED_CATEGORIES_PROMPT_PREFIX}{cat_list}\n\nSongs to classify:\n{song_list}"


def classify_songs(
    client: anthropic.Anthropic,
    songs: list[Song],
    on_progress: typing.Callable[[int, int, int], None] | None = None,
    on_batch_done: typing.Callable[[ClassificationResult], None] | None = None,
    max_playlists: int = DEFAULT_MAX_PLAYLISTS,
    fixed_categories: list[str] | None = None,
) -> ClassificationResult:
    """Classify songs into categories using Claude. Handles batching for large collections.

    on_batch_done is called after each batch with that batch's results,
    allowing the caller to persist progress incrementally.

    If fixed_categories is provided, songs are assigned only to those categories
    (no new categories are created).
    """
    all_categories: list[Category] = []
    existing: list[Category] | None = None
    if fixed_categories is not None:
        existing = [Category(name=c, description="", song_ids=[]) for c in fixed_categories]
    total_batches = (len(songs) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_num, i in enumerate(range(0, len(songs), BATCH_SIZE), 1):
        batch = songs[i : i + BATCH_SIZE]
        if on_progress:
            on_progress(batch_num, total_batches, len(batch))
        result = _classify_batch(
            client, batch, existing,
            max_playlists=max_playlists,
            fixed_only=fixed_categories is not None,
        )
        if on_batch_done:
            on_batch_done(result)
        all_categories = _merge_categories(all_categories, result.categories)
        existing = all_categories

    return ClassificationResult(categories=all_categories)


def _classify_batch(
    client: anthropic.Anthropic,
    songs: list[Song],
    existing_categories: list[Category] | None,
    max_playlists: int = DEFAULT_MAX_PLAYLISTS,
    fixed_only: bool = False,
) -> ClassificationResult:
    if fixed_only and existing_categories:
        user_prompt = _build_fixed_categories_prompt(songs, existing_categories)
        system_prompt = _FIXED_CATEGORIES_SYSTEM_PROMPT
    else:
        user_prompt = _build_user_prompt(songs, existing_categories)
        system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(max_playlists=max_playlists)

    response = client.messages.create(
        model=BEDROCK_MODEL if LLM_PROVIDER == "bedrock" else DIRECT_MODEL,
        max_tokens=8192,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )

    if response.stop_reason == "max_tokens":
        raise RuntimeError(
            f"Claude response was truncated (batch of {len(songs)} songs). "
            "Try reducing BATCH_SIZE."
        )

    text = response.content[0].text
    # Extract JSON from response, stripping markdown code fences if present
    text = text.strip()
    if text.startswith("```"):
        # Remove opening fence (```json, ```, etc.) and closing fence
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]

    # Fallback: find the first { ... } JSON object
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        text = text[start : end + 1]

    parsed = json.loads(text)
    return ClassificationResult.model_validate(parsed)


_GENRE_SYSTEM_PROMPT = """You are a music curator. Given a list of songs that match certain genres, \
organize them into playlists. Each song can appear in up to 4 playlists if it fits naturally.

Return your response as a JSON object with this exact structure:
{
  "categories": [
    {
      "name": "Playlist Name",
      "description": "Brief description of what ties these songs together",
      "song_ids": ["spotify_id_1", "spotify_id_2"]
    }
  ]
}

Important:
- Every song_id from the input must appear in at least one category
- A song may appear in up to 4 categories if it genuinely fits multiple playlists
- Playlist names should be descriptive and evocative
- Aim for balanced playlists — each should have at least 5 songs
- Return ONLY the JSON object, no other text"""


def classify_by_genre(
    client: anthropic.Anthropic,
    songs: list[Song],
    genres: list[str],
    max_playlists: int = 1,
    playlist_name: str | None = None,
) -> ClassificationResult:
    """Classify songs matching specific genres into playlists."""
    song_list = _format_songs_for_prompt(songs)
    genre_str = ", ".join(genres)

    if playlist_name and max_playlists == 1:
        user_prompt = (
            f"These songs match the genres: {genre_str}.\n"
            f"Put all of them into a single playlist named \"{playlist_name}\".\n\n"
            f"Songs:\n{song_list}"
        )
    else:
        user_prompt = (
            f"These songs match the genres: {genre_str}.\n"
            f"Organize them into up to {max_playlists} playlist(s) based on sub-genre, "
            f"mood, or thematic coherence.\n\n"
            f"Songs:\n{song_list}"
        )

    response = client.messages.create(
        model=BEDROCK_MODEL if LLM_PROVIDER == "bedrock" else DIRECT_MODEL,
        max_tokens=8192,
        system=_GENRE_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )

    text = response.content[0].text
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        text = text[start : end + 1]

    parsed = json.loads(text)
    return ClassificationResult.model_validate(parsed)


def _merge_categories(
    existing: list[Category], new: list[Category]
) -> list[Category]:
    """Merge new categories into existing ones, combining by name."""
    by_name: dict[str, Category] = {c.name: c for c in existing}

    for cat in new:
        if cat.name in by_name:
            merged_ids = by_name[cat.name].song_ids + cat.song_ids
            by_name[cat.name] = cat.model_copy(update={"song_ids": merged_ids})
        else:
            by_name[cat.name] = cat

    return list(by_name.values())
