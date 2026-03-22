import click

from sweepify import db


@click.group()
def main():
    """sweepify — Classify your Spotify Liked Songs into playlists using AI."""
    db.init_db()


def _fetch(playlist: str | None = None) -> list[str]:
    """Fetch songs. Returns list of fetched song IDs."""
    from sweepify import spotify

    click.echo("Connecting to Spotify...")
    sp = spotify.get_client()

    source = f"playlist '{playlist}'" if playlist else "Liked Songs"
    click.echo(f"Fetching songs from {source}...")
    songs = spotify.fetch_liked_songs(sp, playlist=playlist)
    click.echo(f"Fetched {len(songs)} song(s) from Spotify.")

    count = db.upsert_songs(songs)
    click.echo(f"Added {count} new song(s) to local database.")
    return [s.spotify_id for s in songs]


def _classify(song_ids: list[str] | None = None) -> int:
    """Classify unclassified songs. Returns number of songs classified.

    If song_ids is provided, only classify those songs (that are also unclassified).
    """
    from sweepify import classifier

    songs = db.get_unclassified_songs()
    if song_ids is not None:
        allowed = set(song_ids)
        songs = [s for s in songs if s.spotify_id in allowed]

    if not songs:
        click.echo("No unclassified songs found.")
        return 0

    click.echo(f"Classifying {len(songs)} song(s) with Claude...")
    client = classifier.get_client()
    classified_count = 0

    def on_progress(batch: int, total: int, size: int) -> None:
        click.echo(f"  Batch {batch}/{total} ({size} songs)...")

    def on_batch_done(result: classifier.ClassificationResult) -> None:
        nonlocal classified_count
        for cat in result.categories:
            db.mark_classified(cat.song_ids, cat.name, playlist_id="")
            classified_count += len(cat.song_ids)

    result = classifier.classify_songs(
        client, songs, on_progress=on_progress, on_batch_done=on_batch_done,
    )

    click.echo("Categories:")
    for cat in result.categories:
        click.echo(f"  {cat.name}: {len(cat.song_ids)} song(s)")

    click.echo(f"Classified {classified_count} song(s) into {len(result.categories)} categories.")
    return classified_count


def _create() -> int:
    """Create playlists. Returns number of playlists processed."""
    from sweepify import spotify
    from sweepify.models import Playlist

    songs_by_cat = db.get_songs_by_category()
    if not songs_by_cat:
        click.echo("No classified songs pending playlist creation.")
        return 0

    click.echo("Connecting to Spotify...")
    sp = spotify.get_client()

    for category, songs in songs_by_cat.items():
        song_ids = [s.spotify_id for s in songs]
        existing = db.get_playlist_by_name(category)

        if existing:
            click.echo(f"  Adding {len(song_ids)} song(s) to existing playlist: {category}")
            spotify.add_to_existing_playlist(sp, existing.spotify_id, song_ids)
            playlist_id = existing.spotify_id
        else:
            click.echo(f"  Creating playlist: {category} ({len(song_ids)} songs)")
            playlist_id = spotify.create_playlist(sp, category, song_ids)
            db.upsert_playlist(Playlist(spotify_id=playlist_id, name=category))

        db.mark_classified(song_ids, category, playlist_id)

    click.echo(f"Processed {len(songs_by_cat)} playlist(s).")
    return len(songs_by_cat)


@main.command()
@click.option("--playlist", "-p", default=None, help="Fetch from a specific playlist (name or ID) instead of Liked Songs.")
def fetch(playlist: str | None):
    """Fetch liked songs from Spotify and store locally."""
    _fetch(playlist=playlist)


@main.command()
@click.option("--playlist", "-p", default=None, help="Only classify songs from this playlist (name or ID).")
def classify(playlist: str | None):
    """Classify unclassified songs using Claude."""
    song_ids = None
    if playlist:
        from sweepify import spotify
        click.echo("Resolving playlist...")
        sp = spotify.get_client()
        songs = spotify.fetch_liked_songs(sp, playlist=playlist)
        song_ids = [s.spotify_id for s in songs]
    _classify(song_ids=song_ids)


@main.command()
def create():
    """Create Spotify playlists and add classified songs."""
    _create()


@main.command()
@click.option("--playlist", "-p", default=None, help="Fetch from a specific playlist instead of Liked Songs.")
def run(playlist: str | None):
    """Run full pipeline: fetch → classify → create."""
    click.echo("=== Step 1/3: Fetch ===")
    try:
        fetched_ids = _fetch(playlist=playlist)
    except Exception as e:
        click.echo(f"Error during fetch: {e}", err=True)
        raise click.Abort()

    click.echo("\n=== Step 2/3: Classify ===")
    try:
        _classify(song_ids=fetched_ids if playlist else None)
    except Exception as e:
        click.echo(f"Error during classify: {e}", err=True)
        raise click.Abort()

    click.echo("\n=== Step 3/3: Create Playlists ===")
    try:
        _create()
    except Exception as e:
        click.echo(f"Error during playlist creation: {e}", err=True)
        raise click.Abort()

    click.echo("\nDone! Run 'sweepify status' to see a summary.")


@main.command()
def status():
    """Show song and classification counts."""
    s = db.get_status()
    click.echo(f"Total songs:   {s['total']}")
    click.echo(f"Classified:    {s['classified']}")
    click.echo(f"Unclassified:  {s['unclassified']}")
    click.echo(f"Categories:    {s['categories']}")
    click.echo(f"Playlists:     {s['playlists']}")


@main.command()
def reset():
    """Clear all classification data (keeps songs)."""
    count = db.reset_classifications()
    click.echo(f"Reset {count} song(s). Playlists cleared.")


@main.command()
def ui():
    """Open the interactive database explorer (Streamlit)."""
    import subprocess
    import sys
    from pathlib import Path

    app_path = Path(__file__).parent / "ui.py"
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path)], check=True)
