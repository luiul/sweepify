import click


@click.group()
def main():
    """mapa — Classify your Spotify Liked Songs into playlists using AI."""
    pass


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
    click.echo("Not implemented yet.")


@main.command()
def reset():
    """Clear all classification data (keeps songs)."""
    click.echo("Not implemented yet.")
