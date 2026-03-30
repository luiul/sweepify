from dataclasses import dataclass
from typing import Annotated, get_args, get_origin

from pydantic import BaseModel


@dataclass(frozen=True)
class ColumnMeta:
    primary_key: bool = False
    sql_default: str | None = None
    db_managed: bool = False


PrimaryKey = Annotated[str, ColumnMeta(primary_key=True)]
FetchedTimestamp = Annotated[str | None, ColumnMeta(sql_default="CURRENT_TIMESTAMP", db_managed=True)]

PYTHON_TO_SQLITE = {
    str: "TEXT",
    int: "INTEGER",
    float: "REAL",
    bool: "INTEGER",
}


def _get_column_meta(field_info) -> ColumnMeta:
    for item in field_info.metadata:
        if isinstance(item, ColumnMeta):
            return item
    return ColumnMeta()


def _resolve_sqlite_type(annotation) -> str:
    if get_origin(annotation) is Annotated:
        annotation = get_args(annotation)[0]

    origin = get_origin(annotation)
    if origin is type(str | None):
        args = [a for a in get_args(annotation) if a is not type(None)]
        if args:
            return PYTHON_TO_SQLITE.get(args[0], "TEXT")
    return PYTHON_TO_SQLITE.get(annotation, "TEXT")


def generate_create_table(model: type[BaseModel], table_name: str) -> str:
    columns = []
    for name, info in model.model_fields.items():
        col_type = _resolve_sqlite_type(info.annotation)
        meta = _get_column_meta(info)

        parts = [name, col_type]
        if meta.primary_key:
            parts.append("PRIMARY KEY")
        if info.is_required() and not meta.primary_key:
            parts.append("NOT NULL")
        if meta.sql_default:
            parts.append(f"DEFAULT {meta.sql_default}")
        elif info.default is not None and not info.is_required() and not meta.primary_key:
            if isinstance(info.default, bool):
                parts.append(f"DEFAULT {int(info.default)}")
            elif isinstance(info.default, (int, float)):
                parts.append(f"DEFAULT {info.default}")

        columns.append("    " + " ".join(parts))

    cols = ",\n".join(columns)
    return f"CREATE TABLE IF NOT EXISTS {table_name} (\n{cols}\n);"


def get_insert_columns(model: type[BaseModel]) -> list[str]:
    return [
        name
        for name, info in model.model_fields.items()
        if not _get_column_meta(info).db_managed
    ]


class Song(BaseModel):
    spotify_id: PrimaryKey
    name: str
    artist: str
    album: str | None = None
    genres: str | None = None
    added_at: str | None = None
    duration_ms: int | None = None
    explicit: bool | None = None
    release_date: str | None = None
    mood: str | None = None
    bpm: int | None = None
    vibe: str | None = None
    related_artists: str | None = None
    enriched: bool = False
    classified: bool = False
    refined: bool = False
    categories: str | None = None
    playlist_ids: str | None = None
    fetched_at: FetchedTimestamp = None


class Playlist(BaseModel):
    spotify_id: PrimaryKey
    name: str
    created_at: FetchedTimestamp = None
