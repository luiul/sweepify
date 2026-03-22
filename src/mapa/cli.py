import click

from mapa import db


@click.group()
def main():
    """mapa — Classify your Spotify Liked Songs into playlists using AI."""
    db.init_db()


def _fetch() -> int:
    """Fetch liked songs. Returns number of new songs added."""
    from mapa import spotify

    click.echo("Connecting to Spotify...")
    sp = spotify.get_client()

    click.echo("Fetching liked songs...")
    songs = spotify.fetch_liked_songs(sp)
    click.echo(f"Fetched {len(songs)} song(s) from Spotify.")

    count = db.upsert_songs(songs)
    click.echo(f"Added {count} new song(s) to local database.")
    return count


def _classify() -> int:
    """Classify unclassified songs. Returns number of songs classified."""
    from mapa import classifier

    songs = db.get_unclassified_songs()
    if not songs:
        click.echo("No unclassified songs found.")
        return 0

    click.echo(f"Classifying {len(songs)} song(s) with Claude...")
    client = classifier.get_client()
    result = classifier.classify_songs(client, songs)

    for cat in result.categories:
        click.echo(f"  {cat.name}: {len(cat.song_ids)} song(s)")
        db.mark_classified(cat.song_ids, cat.name, playlist_id="")

    total = sum(len(c.song_ids) for c in result.categories)
    click.echo(f"Classified {total} song(s) into {len(result.categories)} categories.")
    return total


def _create() -> int:
    """Create playlists. Returns number of playlists processed."""
    from mapa import spotify
    from mapa.models import Playlist

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
def fetch():
    """Fetch liked songs from Spotify and store locally."""
    _fetch()


@main.command()
def classify():
    """Classify unclassified songs using Claude."""
    _classify()


@main.command()
def create():
    """Create Spotify playlists and add classified songs."""
    _create()


@main.command()
def run():
    """Run full pipeline: fetch → classify → create."""
    click.echo("=== Step 1/3: Fetch ===")
    try:
        _fetch()
    except Exception as e:
        click.echo(f"Error during fetch: {e}", err=True)
        raise click.Abort()

    click.echo("\n=== Step 2/3: Classify ===")
    try:
        _classify()
    except Exception as e:
        click.echo(f"Error during classify: {e}", err=True)
        raise click.Abort()

    click.echo("\n=== Step 3/3: Create Playlists ===")
    try:
        _create()
    except Exception as e:
        click.echo(f"Error during playlist creation: {e}", err=True)
        raise click.Abort()

    click.echo("\nDone! Run 'mapa status' to see a summary.")


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
