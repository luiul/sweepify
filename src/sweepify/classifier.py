import json
import typing

import anthropic
from pydantic import BaseModel

from sweepify.config import ANTHROPIC_API_KEY, AWS_REGION, LLM_PROVIDER
from sweepify.models import Song

BATCH_SIZE = 100

SYSTEM_PROMPT = """You are a music curator. Given a list of songs with metadata (name, artist, album, genres), \
group them into playlists. Create between 5 and 15 categories based on genre, mood, energy, \
and thematic coherence. Each song must belong to exactly one category.

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
- Every song_id from the input must appear in exactly one category
- Category names should be descriptive and evocative (e.g. "Late Night Drive", "Sunday Morning Coffee")
- Do not use generic names like "Category 1" or "Miscellaneous"
- Return ONLY the JSON object, no other text"""

FOLLOWUP_PROMPT_PREFIX = """Here are the existing categories from previous batches. \
Assign each song to one of these categories. Only create a new category if a song truly \
does not fit any existing one.

Existing categories:
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
        lines.append(f"- id={s.spotify_id} | {s.name} | {s.artist} | {s.album} | genres: {genres}")
    return "\n".join(lines)


def _build_user_prompt(songs: list[Song], existing_categories: list[Category] | None) -> str:
    song_list = _format_songs_for_prompt(songs)

    if existing_categories:
        cat_summary = "\n".join(
            f"- {c.name}: {c.description}" for c in existing_categories
        )
        return f"{FOLLOWUP_PROMPT_PREFIX}{cat_summary}\n\nSongs to classify:\n{song_list}"

    return f"Classify these songs into playlists:\n{song_list}"


def classify_songs(
    client: anthropic.Anthropic,
    songs: list[Song],
    on_progress: typing.Callable[[int, int, int], None] | None = None,
    on_batch_done: typing.Callable[[ClassificationResult], None] | None = None,
) -> ClassificationResult:
    """Classify songs into categories using Claude. Handles batching for large collections.

    on_batch_done is called after each batch with that batch's results,
    allowing the caller to persist progress incrementally.
    """
    all_categories: list[Category] = []
    existing: list[Category] | None = None
    total_batches = (len(songs) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_num, i in enumerate(range(0, len(songs), BATCH_SIZE), 1):
        batch = songs[i : i + BATCH_SIZE]
        if on_progress:
            on_progress(batch_num, total_batches, len(batch))
        result = _classify_batch(client, batch, existing)
        if on_batch_done:
            on_batch_done(result)
        all_categories = _merge_categories(all_categories, result.categories)
        existing = all_categories

    return ClassificationResult(categories=all_categories)


def _classify_batch(
    client: anthropic.Anthropic,
    songs: list[Song],
    existing_categories: list[Category] | None,
) -> ClassificationResult:
    user_prompt = _build_user_prompt(songs, existing_categories)

    response = client.messages.create(
        model=BEDROCK_MODEL if LLM_PROVIDER == "bedrock" else DIRECT_MODEL,
        max_tokens=8192,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )

    if response.stop_reason == "max_tokens":
        raise RuntimeError(
            f"Claude response was truncated (batch of {len(songs)} songs). "
            "Try reducing BATCH_SIZE."
        )

    text = response.content[0].text
    # Strip markdown code fences if present
    if text.strip().startswith("```"):
        lines = text.strip().split("\n")
        text = "\n".join(lines[1:-1])

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
