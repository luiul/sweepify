import json

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn
from rich.table import Table

from sweepify import db
from sweepify.config import PLAYLIST_PREFIX

console = Console()

def _make_progress() -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    )


@click.group()
def main():
    """sweepify — Classify your Spotify Liked Songs into playlists using AI."""
    db.init_db()


def _fetch(playlist: str | None = None) -> list[str]:
    """Fetch songs. Returns list of fetched song IDs."""
    from sweepify import spotify

    console.print("Connecting to Spotify...")
    sp = spotify.get_client()

    source = f"playlist '{playlist}'" if playlist else "Liked Songs"

    with _make_progress() as progress:
        fetch_task = progress.add_task(f"Fetching songs from {source}", total=None)
        genre_task = progress.add_task("Enriching genres", total=None, visible=False)

        def on_fetch(fetched: int, total: int) -> None:
            progress.update(fetch_task, completed=fetched, total=total)

        def on_genre(processed: int, total: int) -> None:
            progress.update(genre_task, completed=processed, total=total, visible=True)

        songs = spotify.fetch_liked_songs(
            sp, playlist=playlist,
            on_progress=on_fetch, on_genre_progress=on_genre,
        )
        progress.update(fetch_task, completed=progress.tasks[fetch_task].total or len(songs))
        progress.update(genre_task, completed=progress.tasks[genre_task].total or 0, visible=progress.tasks[genre_task].visible)

    console.print(f"Fetched {len(songs)} song(s) from Spotify.")

    count = db.upsert_songs(songs)
    console.print(f"Added {count} new song(s) to local database.")
    return [s.spotify_id for s in songs]


def _enrich(song_ids: list[str] | None = None, force: bool = False) -> int:
    """Enrich songs with AI metadata. Returns number of songs enriched."""
    from sweepify import enricher

    if force:
        reset_count = db.reset_enrichments()
        if reset_count:
            console.print(f"Reset enrichment for {reset_count} song(s).")

    songs = db.get_unenriched_songs()
    if song_ids is not None:
        allowed = set(song_ids)
        songs = [s for s in songs if s.spotify_id in allowed]

    if not songs:
        console.print("No unenriched songs found.")
        return 0

    total_batches = (len(songs) + enricher.BATCH_SIZE - 1) // enricher.BATCH_SIZE
    console.print(f"Enriching {len(songs)} song(s) in {total_batches} batch(es) with {min(enricher.MAX_WORKERS, total_batches)} workers...")
    client = enricher.get_client()
    enriched_count = 0

    with _make_progress() as progress:
        overall = progress.add_task("Overall", total=len(songs))
        batch_tasks: dict[int, int] = {}

        def on_batch_start(batch_num: int, total: int, size: int) -> None:
            batch_tasks[batch_num] = progress.add_task(
                f"  Batch {batch_num}/{total}", total=size, visible=True,
            )

        def on_batch_done(batch_num: int, result: enricher.EnrichmentResult) -> None:
            nonlocal enriched_count
            db.mark_enriched([
                {
                    "spotify_id": e.spotify_id,
                    "mood": e.mood,
                    "bpm": e.bpm,
                    "vibe": e.vibe,
                    "related_artists": json.dumps(e.related_artists),
                }
                for e in result.songs
            ])
            enriched_count += len(result.songs)
            if batch_num in batch_tasks:
                progress.update(batch_tasks[batch_num], completed=len(result.songs))
            progress.advance(overall, len(result.songs))

        enricher.enrich_songs(client, songs, on_batch_start=on_batch_start, on_batch_done=on_batch_done)

    console.print(f"Enriched {enriched_count} song(s).")
    return enriched_count


def _classify(song_ids: list[str] | None = None, max_playlists: int = 10) -> int:
    """Classify unclassified songs. Returns number of songs classified.

    If song_ids is provided, only classify those songs (that are also unclassified).
    If max_playlists is 0, classify only into existing sweepify playlists.
    """
    from sweepify import classifier, spotify

    songs = db.get_unclassified_songs()
    if song_ids is not None:
        allowed = set(song_ids)
        songs = [s for s in songs if s.spotify_id in allowed]

    if not songs:
        console.print("No unclassified songs found.")
        return 0

    fixed_categories = None
    if max_playlists == 0:
        console.print("Fetching existing sweepify playlists from Spotify...")
        sp = spotify.get_client()
        existing = spotify.fetch_sweepify_playlists(sp)
        if not existing:
            console.print("No existing sweepify playlists found. Use -n > 0 to create new ones.")
            return 0
        fixed_categories = list(existing.keys())
        console.print(f"Classifying into {len(fixed_categories)} existing playlist(s).")

    total_batches = (len(songs) + classifier.BATCH_SIZE - 1) // classifier.BATCH_SIZE
    console.print(
        f"Classifying {len(songs)} song(s) in {total_batches} batch(es) "
        f"with {min(classifier.MAX_WORKERS, total_batches)} worker(s)..."
    )
    client = classifier.get_client()
    classified_songs: set[str] = set()

    with _make_progress() as progress:
        overall = progress.add_task("Rough classification", total=len(songs))
        batch_tasks: dict[int, int] = {}
        refine_task: int | None = None

        def on_batch_start(batch_num: int, total: int, size: int) -> None:
            batch_tasks[batch_num] = progress.add_task(
                f"  Batch {batch_num}/{total}", total=size, visible=True,
            )

        def on_batch_done(batch_num: int, result: classifier.ClassificationResult) -> None:
            count = sum(len(c.song_ids) for c in result.categories)
            if batch_num in batch_tasks:
                progress.update(batch_tasks[batch_num], completed=count)
            progress.advance(overall, count)

        def on_refine_start() -> None:
            nonlocal refine_task
            # Hide batch rows
            for tid in batch_tasks.values():
                progress.update(tid, visible=False)
            progress.update(overall, visible=False)
            refine_task = progress.add_task("Refining categories...", total=None)

        def on_refine_done(result: classifier.ClassificationResult) -> None:
            if refine_task is not None:
                progress.update(refine_task, total=1, completed=1)
            for cat in result.categories:
                db.mark_classified(cat.song_ids, cat.name, playlist_id="")
                classified_songs.update(cat.song_ids)

        result = classifier.classify_songs(
            client, songs,
            on_batch_start=on_batch_start,
            on_batch_done=on_batch_done,
            on_refine_start=on_refine_start,
            on_refine_done=on_refine_done,
            max_playlists=max_playlists,
            fixed_categories=fixed_categories,
        )

    console.print("Categories:")
    for cat in result.categories:
        console.print(f"  {cat.name}: {len(cat.song_ids)} song(s)")

    console.print(f"Classified {len(classified_songs)} song(s) into {len(result.categories)} categories.")
    return len(classified_songs)


def _create() -> int:
    """Create playlists. Returns number of playlists processed."""
    from sweepify import spotify
    from sweepify.models import Playlist

    songs_by_cat = db.get_songs_by_category()
    if not songs_by_cat:
        console.print("No classified songs pending playlist creation.")
        return 0

    console.print("Connecting to Spotify...")
    sp = spotify.get_client()

    # Sync existing sweepify playlists from Spotify into local DB
    remote_playlists = spotify.fetch_sweepify_playlists(sp)
    if remote_playlists:
        console.print(f"Found {len(remote_playlists)} existing sweepify playlist(s) on Spotify.")
        for category, pid in remote_playlists.items():
            db.upsert_playlist(Playlist(spotify_id=pid, name=category))

    with _make_progress() as progress:
        task = progress.add_task("Creating playlists", total=len(songs_by_cat))

        for category, songs in songs_by_cat.items():
            song_ids = [s.spotify_id for s in songs]
            existing = db.get_playlist_by_name(category)

            if existing:
                progress.console.print(f"  Adding {len(song_ids)} song(s) to existing playlist: {category}")
                spotify.add_to_existing_playlist(sp, existing.spotify_id, song_ids)
                playlist_id = existing.spotify_id
            else:
                progress.console.print(f"  Creating playlist: {category} ({len(song_ids)} songs)")
                playlist_id = spotify.create_playlist(sp, category, song_ids)
                db.upsert_playlist(Playlist(spotify_id=playlist_id, name=category))

            db.mark_classified(song_ids, category, playlist_id)
            progress.advance(task)

    console.print(f"Processed {len(songs_by_cat)} playlist(s).")
    return len(songs_by_cat)


@main.command()
@click.option("--playlist", "-p", default=None, help="Fetch from a specific playlist (name or ID) instead of Liked Songs.")
def fetch(playlist: str | None):
    """Fetch liked songs from Spotify and store locally."""
    _fetch(playlist=playlist)


@main.command()
@click.option("--playlist", "-p", default=None, help="Only classify songs from this playlist (name or ID).")
@click.option("--max-playlists", "-n", default=10, show_default=True, help="Maximum number of playlists to create. Use 0 to only classify into existing playlists.")
def classify(playlist: str | None, max_playlists: int):
    """Classify unclassified songs using Claude."""
    song_ids = None
    if playlist:
        from sweepify import spotify
        console.print("Resolving playlist...")
        sp = spotify.get_client()
        songs = spotify.fetch_liked_songs(sp, playlist=playlist)
        song_ids = [s.spotify_id for s in songs]
    _classify(song_ids=song_ids, max_playlists=max_playlists)


@main.command()
@click.option("--playlist", "-p", default=None, help="Only enrich songs from this playlist (name or ID).")
@click.option("--force", "-f", is_flag=True, help="Re-enrich already enriched songs.")
def enrich(playlist: str | None, force: bool):
    """Enrich songs with AI-generated metadata (mood, BPM, vibe, related artists)."""
    song_ids = None
    if playlist:
        from sweepify import spotify
        console.print("Resolving playlist...")
        sp = spotify.get_client()
        songs = spotify.fetch_liked_songs(sp, playlist=playlist)
        song_ids = [s.spotify_id for s in songs]
    _enrich(song_ids=song_ids, force=force)


@main.command()
def create():
    """Create Spotify playlists and add classified songs."""
    _create()


@main.command()
@click.option("--playlist", "-p", default=None, help="Fetch from a specific playlist instead of Liked Songs.")
@click.option("--max-playlists", "-n", default=10, show_default=True, help="Maximum number of playlists to create. Use 0 to only classify into existing playlists.")
def run(playlist: str | None, max_playlists: int):
    """Run full pipeline: fetch → enrich → classify → create."""
    console.rule("[bold]Step 1/4: Fetch")
    try:
        fetched_ids = _fetch(playlist=playlist)
    except Exception as e:
        console.print(f"[red]Error during fetch: {e}[/red]")
        raise click.Abort()

    console.rule("[bold]Step 2/4: Enrich")
    try:
        _enrich(song_ids=fetched_ids if playlist else None)
    except Exception as e:
        console.print(f"[red]Error during enrichment: {e}[/red]")
        raise click.Abort()

    console.rule("[bold]Step 3/4: Classify")
    try:
        _classify(song_ids=fetched_ids if playlist else None, max_playlists=max_playlists)
    except Exception as e:
        console.print(f"[red]Error during classify: {e}[/red]")
        raise click.Abort()

    console.rule("[bold]Step 4/4: Create Playlists")
    try:
        _create()
    except Exception as e:
        console.print(f"[red]Error during playlist creation: {e}[/red]")
        raise click.Abort()

    console.print("\n[green]Done![/green] Run 'sweepify status' to see a summary.")


@main.command()
def status():
    """Show song and classification counts."""
    s = db.get_status()
    table = Table(title="Sweepify Status", show_header=False)
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    table.add_row("Total songs", str(s['total']))
    table.add_row("Enriched", str(s['enriched']))
    table.add_row("Classified", str(s['classified']))
    table.add_row("Unclassified", str(s['unclassified']))
    table.add_row("Categories", str(s['categories']))
    table.add_row("Playlists", str(s['playlists']))
    console.print(table)


@main.command()
@click.option("--enrichment", is_flag=True, help="Also reset enrichment data (mood, BPM, vibe, related artists).")
def reset(enrichment: bool):
    """Clear all classification data (keeps songs)."""
    count = db.reset_classifications()
    console.print(f"Reset {count} song(s). Playlists cleared.")
    if enrichment:
        enrich_count = db.reset_enrichments()
        console.print(f"Reset enrichment for {enrich_count} song(s).")


@main.command()
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt.")
def clear(yes: bool):
    """Remove all sweepify playlists from Spotify and reset classifications."""
    from sweepify import spotify

    console.print("Connecting to Spotify...")
    sp = spotify.get_client()
    playlists = spotify.fetch_sweepify_playlists(sp)

    if not playlists:
        console.print("No sweepify playlists found on Spotify.")
        count = db.reset_classifications()
        if count:
            console.print(f"Reset {count} local classification(s).")
        return

    console.print(f"This will delete {len(playlists)} playlist(s) from Spotify:")
    for name in playlists:
        console.print(f"  - {PLAYLIST_PREFIX} {name}")

    if not yes:
        click.confirm("\nAre you sure?", abort=True)

    with _make_progress() as progress:
        task = progress.add_task("Deleting playlists", total=len(playlists))

        def on_delete_progress(deleted: int, total: int) -> None:
            progress.update(task, completed=deleted)

        deleted = spotify.delete_sweepify_playlists(sp, on_progress=on_delete_progress)

    console.print(f"Deleted {len(deleted)} playlist(s) from Spotify.")

    count = db.reset_classifications()
    console.print(f"Reset {count} local classification(s).")


@main.command()
@click.option("--genres", "-g", required=True, help="Comma-separated list of genres to match.")
@click.option("--name", default=None, help="Playlist name. If omitted, Claude picks one.")
@click.option("--max-playlists", "-n", default=1, show_default=True, help="Number of playlists to split into.")
def playlist(genres: str, name: str | None, max_playlists: int):
    """Create a playlist from songs matching specific genres."""
    from sweepify import classifier, spotify
    from sweepify.models import Playlist

    genre_list = [g.strip() for g in genres.split(",") if g.strip()]
    if not genre_list:
        console.print("[red]No genres provided.[/red]")
        raise click.Abort()

    console.print(f"Searching for songs matching: {', '.join(genre_list)}")
    songs = db.get_songs_by_genres(genre_list)

    if not songs:
        console.print("No songs found matching those genres.")
        return

    console.print(f"Found {len(songs)} song(s). Sending to Claude...")
    client = classifier.get_client()
    result = classifier.classify_by_genre(
        client, songs, genre_list,
        max_playlists=max_playlists, playlist_name=name,
    )

    console.print("Connecting to Spotify...")
    sp = spotify.get_client()

    # Sync existing playlists
    remote_playlists = spotify.fetch_sweepify_playlists(sp)
    for cat, pid in remote_playlists.items():
        db.upsert_playlist(Playlist(spotify_id=pid, name=cat))

    with _make_progress() as progress:
        task = progress.add_task("Creating playlists", total=len(result.categories))

        for cat in result.categories:
            song_ids = cat.song_ids
            existing = db.get_playlist_by_name(cat.name)

            if existing:
                progress.console.print(f"  Adding {len(song_ids)} song(s) to existing playlist: {cat.name}")
                spotify.add_to_existing_playlist(sp, existing.spotify_id, song_ids)
                playlist_id = existing.spotify_id
            else:
                progress.console.print(f"  Creating playlist: {cat.name} ({len(song_ids)} songs)")
                playlist_id = spotify.create_playlist(sp, cat.name, song_ids)
                db.upsert_playlist(Playlist(spotify_id=playlist_id, name=cat.name))

            db.mark_classified(song_ids, cat.name, playlist_id)
            progress.advance(task)

    total = sum(len(c.song_ids) for c in result.categories)
    console.print(f"[green]Done![/green] Added {total} song(s) to {len(result.categories)} playlist(s).")


@main.command()
def ui():
    """Open the interactive database explorer (Streamlit)."""
    import subprocess
    import sys
    from pathlib import Path

    app_path = Path(__file__).parent / "ui.py"
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path)], check=True)
