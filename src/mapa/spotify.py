import json

import spotipy
from spotipy.oauth2 import SpotifyOAuth

from mapa.config import SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET, SPOTIFY_REDIRECT_URI
from mapa.models import Song

SCOPES = "user-library-read playlist-modify-private playlist-modify-public"


def get_client() -> spotipy.Spotify:
    auth_manager = SpotifyOAuth(
        client_id=SPOTIFY_CLIENT_ID,
        client_secret=SPOTIFY_CLIENT_SECRET,
        redirect_uri=SPOTIFY_REDIRECT_URI,
        scope=SCOPES,
    )
    return spotipy.Spotify(auth_manager=auth_manager)


def fetch_liked_songs(sp: spotipy.Spotify) -> list[Song]:
    """Fetch all liked songs with pagination. Returns songs with artist IDs stored in genres field temporarily."""
    songs: list[Song] = []
    # Track artist IDs per song for genre enrichment
    song_artist_ids: dict[str, list[str]] = {}
    offset = 0

    while True:
        results = sp.current_user_saved_tracks(limit=50, offset=offset)
        items = results.get("items", [])
        if not items:
            break

        for item in items:
            track = item["track"]
            artist_names = ", ".join(a["name"] for a in track["artists"])
            artist_ids = [a["id"] for a in track["artists"] if a.get("id")]
            song_id = track["id"]

            songs.append(
                Song(
                    spotify_id=song_id,
                    name=track["name"],
                    artist=artist_names,
                    album=track["album"]["name"],
                    added_at=item.get("added_at"),
                )
            )
            song_artist_ids[song_id] = artist_ids

        offset += len(items)
        if not results.get("next"):
            break

    return _enrich_with_genres(sp, songs, song_artist_ids)


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
