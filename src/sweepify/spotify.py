import json

import click
import spotipy
from spotipy.oauth2 import SpotifyOAuth

from sweepify.config import PLAYLIST_PREFIX, SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET, SPOTIFY_REDIRECT_URI
from sweepify.models import Song

SCOPES = "user-library-read playlist-read-private playlist-modify-private playlist-modify-public"


def get_client() -> spotipy.Spotify:
    auth_manager = SpotifyOAuth(
        client_id=SPOTIFY_CLIENT_ID,
        client_secret=SPOTIFY_CLIENT_SECRET,
        redirect_uri=SPOTIFY_REDIRECT_URI,
        scope=SCOPES,
    )
    return spotipy.Spotify(auth_manager=auth_manager)


def fetch_liked_songs(sp: spotipy.Spotify, playlist: str | None = None) -> list[Song]:
    """Fetch songs from Liked Songs or a specific playlist by name/ID."""
    if playlist:
        playlist_id = _resolve_playlist(sp, playlist)
        return _fetch_playlist_tracks(sp, playlist_id)
    return _fetch_saved_tracks(sp)


def _resolve_playlist(sp: spotipy.Spotify, playlist: str) -> str:
    """Resolve a playlist name or ID to a playlist ID."""
    # If it looks like a Spotify ID (22 chars, alphanumeric), use directly
    if len(playlist) == 22 and playlist.isalnum():
        return playlist

    # Search user playlists by name
    offset = 0
    while True:
        results = sp.current_user_playlists(limit=50, offset=offset)
        items = results.get("items", [])
        if not items:
            break
        for item in items:
            if item["name"].lower() == playlist.lower():
                return item["id"]
        offset += len(items)
        if not results.get("next"):
            break

    raise click.ClickException(f"Playlist '{playlist}' not found.")


def _fetch_saved_tracks(sp: spotipy.Spotify) -> list[Song]:
    """Fetch all liked songs with pagination."""
    songs: list[Song] = []
    song_artist_ids: dict[str, list[str]] = {}
    offset = 0

    while True:
        results = sp.current_user_saved_tracks(limit=50, offset=offset)
        items = results.get("items", [])
        if not items:
            break

        for item in items:
            track = item["track"]
            song, artist_ids = _parse_track(track, added_at=item.get("added_at"))
            songs.append(song)
            song_artist_ids[song.spotify_id] = artist_ids

        offset += len(items)
        if not results.get("next"):
            break

    return _enrich_with_genres(sp, songs, song_artist_ids)


def _fetch_playlist_tracks(sp: spotipy.Spotify, playlist_id: str) -> list[Song]:
    """Fetch all tracks from a playlist."""
    songs: list[Song] = []
    song_artist_ids: dict[str, list[str]] = {}
    offset = 0

    while True:
        results = sp.playlist_items(playlist_id, limit=100, offset=offset)
        items = results.get("items", [])
        if not items:
            break

        for item in items:
            track = item.get("track")
            if not track or not track.get("id"):
                continue
            song, artist_ids = _parse_track(track, added_at=item.get("added_at"))
            songs.append(song)
            song_artist_ids[song.spotify_id] = artist_ids

        offset += len(items)
        if not results.get("next"):
            break

    return _enrich_with_genres(sp, songs, song_artist_ids)


def _parse_track(track: dict, added_at: str | None = None) -> tuple[Song, list[str]]:
    """Parse a Spotify track dict into a Song and list of artist IDs."""
    artist_names = ", ".join(a["name"] for a in track["artists"])
    artist_ids = [a["id"] for a in track["artists"] if a.get("id")]
    song = Song(
        spotify_id=track["id"],
        name=track["name"],
        artist=artist_names,
        album=track["album"]["name"],
        added_at=added_at,
    )
    return song, artist_ids


def _enrich_with_genres(
    sp: spotipy.Spotify,
    songs: list[Song],
    song_artist_ids: dict[str, list[str]],
) -> list[Song]:
    """Fetch artist genres in batches and attach them to songs."""
    # Collect all unique artist IDs
    all_artist_ids = list({
        aid
        for ids in song_artist_ids.values()
        for aid in ids
    })

    # Fetch genres in batches of 50 (Spotify API limit)
    artist_genres: dict[str, list[str]] = {}
    for i in range(0, len(all_artist_ids), 50):
        batch = all_artist_ids[i : i + 50]
        try:
            results = sp.artists(batch)
            for artist in results.get("artists", []):
                if artist:
                    artist_genres[artist["id"]] = artist.get("genres", [])
        except Exception:
            pass

    # Attach genres to songs
    enriched = []
    for song in songs:
        genres: set[str] = set()
        for aid in song_artist_ids.get(song.spotify_id, []):
            genres.update(artist_genres.get(aid, []))
        genre_json = json.dumps(sorted(genres)) if genres else None
        enriched.append(song.model_copy(update={"genres": genre_json}))

    return enriched


def create_playlist(
    sp: spotipy.Spotify, category_name: str, song_ids: list[str]
) -> str:
    """Create a Spotify playlist and add songs. Returns the playlist ID."""
    user_id = sp.current_user()["id"]
    playlist_name = f"{PLAYLIST_PREFIX} {category_name}"

    playlist = sp.user_playlist_create(
        user=user_id,
        name=playlist_name,
        public=False,
        description=f"Auto-generated by sweepify",
    )
    playlist_id = playlist["id"]

    _add_tracks_to_playlist(sp, playlist_id, song_ids)
    return playlist_id


def add_to_existing_playlist(
    sp: spotipy.Spotify, playlist_id: str, song_ids: list[str]
) -> None:
    """Add songs to an existing playlist."""
    _add_tracks_to_playlist(sp, playlist_id, song_ids)


def _add_tracks_to_playlist(
    sp: spotipy.Spotify, playlist_id: str, song_ids: list[str]
) -> None:
    """Add tracks in batches of 100 (Spotify API limit)."""
    uris = [f"spotify:track:{sid}" for sid in song_ids]
    for i in range(0, len(uris), 100):
        sp.playlist_add_items(playlist_id, uris[i : i + 100])
