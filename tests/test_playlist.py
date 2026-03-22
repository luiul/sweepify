from unittest.mock import MagicMock, call

from mapa.spotify import create_playlist, add_to_existing_playlist


def test_create_playlist():
    sp = MagicMock()
    sp.current_user.return_value = {"id": "user123"}
    sp.user_playlist_create.return_value = {"id": "pl_new"}

    playlist_id = create_playlist(sp, "Chill Vibes", ["t1", "t2"])

    assert playlist_id == "pl_new"
    sp.user_playlist_create.assert_called_once()
    create_args = sp.user_playlist_create.call_args
    assert "Chill Vibes" in create_args.kwargs["name"]
    assert create_args.kwargs["public"] is False
    sp.playlist_add_items.assert_called_once_with(
        "pl_new", ["spotify:track:t1", "spotify:track:t2"]
    )


def test_add_to_existing_playlist():
    sp = MagicMock()

    add_to_existing_playlist(sp, "pl_existing", ["t1", "t2"])

    sp.playlist_add_items.assert_called_once_with(
        "pl_existing", ["spotify:track:t1", "spotify:track:t2"]
    )


def test_add_tracks_batches_over_100():
    sp = MagicMock()
    sp.current_user.return_value = {"id": "user123"}
    sp.user_playlist_create.return_value = {"id": "pl_big"}

    song_ids = [f"t{i}" for i in range(150)]
    create_playlist(sp, "Big Playlist", song_ids)

    assert sp.playlist_add_items.call_count == 2
    first_call_uris = sp.playlist_add_items.call_args_list[0][0][1]
    second_call_uris = sp.playlist_add_items.call_args_list[1][0][1]
    assert len(first_call_uris) == 100
    assert len(second_call_uris) == 50
