import click

from mapa import db


@click.group()
def main():
    """mapa — Classify your Spotify Liked Songs into playlists using AI."""
    db.init_db()


@main.command()
def fetch():
    """Fetch liked songs from Spotify and store locally."""
    click.echo("Not implemented yet.")


@main.command()
def classify():
    """Classify unclassified songs using Claude."""
    click.echo("Not implemented yet.")


@main.command()
def create():
    """Create Spotify playlists and add classified songs."""
    click.echo("Not implemented yet.")


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
