from dataclasses import dataclass


@dataclass
class Song:
    spotify_id: str
    name: str
    artist: str
    album: str | None = None
    genres: str | None = None
    added_at: str | None = None
    energy: float | None = None
    valence: float | None = None
    tempo: float | None = None
    danceability: float | None = None
    classified: bool = False
    playlist_id: str | None = None
    category: str | None = None


@dataclass
class Playlist:
    spotify_id: str
    name: str
    created_at: str | None = None
