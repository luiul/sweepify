import click

from mapa import db


@click.group()
def main():
    """mapa — Classify your Spotify Liked Songs into playlists using AI."""
    db.init_db()


@main.command()
def fetch():
    """Fetch liked songs from Spotify and store locally."""
    from mapa import spotify

    click.echo("Connecting to Spotify...")
    sp = spotify.get_client()

    click.echo("Fetching liked songs...")
    songs = spotify.fetch_liked_songs(sp)
    click.echo(f"Fetched {len(songs)} songs from Spotify.")

    count = db.upsert_songs(songs)
    click.echo(f"Added {count} new song(s) to local database.")


@main.command()
def classify():
    """Classify unclassified songs using Claude."""
    from mapa import classifier

    songs = db.get_unclassified_songs()
    if not songs:
        click.echo("No unclassified songs found. Run 'mapa fetch' first.")
        return

    click.echo(f"Classifying {len(songs)} song(s) with Claude...")
    client = classifier.get_client()
    result = classifier.classify_songs(client, songs)

    for cat in result.categories:
        click.echo(f"  {cat.name}: {len(cat.song_ids)} song(s)")
        # Mark songs as classified with a placeholder playlist_id (created later)
        db.mark_classified(cat.song_ids, cat.name, playlist_id="")

    total = sum(len(c.song_ids) for c in result.categories)
    click.echo(f"Classified {total} song(s) into {len(result.categories)} categories.")


@main.command()
def create():
    """Create Spotify playlists and add classified songs."""
    from mapa import spotify
    from mapa.models import Playlist

    songs_by_cat = db.get_songs_by_category()
    if not songs_by_cat:
        click.echo("No classified songs pending playlist creation. Run 'mapa classify' first.")
        return

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

    click.echo(f"Done. Processed {len(songs_by_cat)} playlist(s).")


@main.command()
def run():
    """Run full pipeline: fetch → classify → create."""
    click.echo("Not implemented yet.")


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
