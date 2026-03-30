import json
import threading
import time
from datetime import timedelta

import pandas as pd
import plotly.express as px
import streamlit as st

from sweepify.config import DB_PATH
from sweepify.db import get_connection, get_songs_by_genres, get_status, init_db

st.set_page_config(page_title="sweepify", page_icon="🧹", layout="wide")

init_db()


_ALLOWED_TABLES = {"songs", "playlists"}


def load_table(table: str) -> pd.DataFrame:
    if table not in _ALLOWED_TABLES:
        raise ValueError(f"Unknown table: {table}")
    with get_connection() as conn:
        return pd.read_sql_query(f"SELECT * FROM {table}", conn)  # noqa: S608


# --- Sidebar ---

st.sidebar.title("sweepify")
st.sidebar.caption(f"DB: `{DB_PATH}`")

status = get_status()
st.sidebar.metric("Total songs", status["total"])
cols = st.sidebar.columns(3)
cols[0].metric("Enriched", status["enriched"])
cols[1].metric("Classified", status["classified"])
cols[2].metric("Unclassified", status["unclassified"])
st.sidebar.metric("Categories", status["categories"])
st.sidebar.metric("Playlists", status["playlists"])

view = st.sidebar.radio("View", ["Actions", "Songs", "Categories", "Genres", "Enrichment", "Playlists", "Playlist Builder", "SQL"])


# --- Action state (cached at server level so it survives page refreshes) ---
# st.session_state resets on refresh; cache_resource persists for the server lifetime.


@st.cache_resource
def _get_action_state() -> dict:
    return {
        "running": None,  # action name or None
        "cancel": False,  # set True to request stop
        "progress": "",  # text description of current step
        "pct": 0.0,  # 0.0 - 1.0
        "result": None,  # ("success"|"warning"|"error", message)
        "thread": None,  # threading.Thread reference
        "gen": 0,  # generation counter -- prevents abandoned threads from overwriting state
        "start_time": None,  # time.monotonic() when action started
    }


_action = _get_action_state()

# --- Sticky sidebar notification (auto-refreshing fragment, visible on all tabs) ---


def _format_eta() -> str:
    """Return a string like '1m 23s elapsed · ~2m 10s remaining' based on progress."""
    start = _action["start_time"]
    pct = _action["pct"]
    if start is None:
        return ""
    elapsed = time.monotonic() - start
    parts = []
    em, es = divmod(int(elapsed), 60)
    parts.append(f"{em}m {es:02d}s elapsed" if em else f"{es}s elapsed")
    if pct and pct > 0.02:
        remaining = elapsed / pct * (1 - pct)
        rm, rs = divmod(int(remaining), 60)
        parts.append(f"~{rm}m {rs:02d}s remaining" if rm else f"~{rs}s remaining")
    return " · ".join(parts)


@st.fragment(run_every=timedelta(seconds=2))
def _action_monitor():
    """Poll background thread progress every 2s without blocking the main script."""
    if _action["running"]:
        thread = _action["thread"]
        if thread is not None and not thread.is_alive():
            _action["running"] = None
            _action["thread"] = None
            st.rerun(scope="app")
            return
        progress = _action["progress"] or "Starting..."
        eta = _format_eta()
        st.info(f"**{_action['running']}** — {progress}")
        if eta:
            st.caption(eta)
        st.progress(_action["pct"])
        if st.button("Stop", key="monitor_stop", type="primary", use_container_width=True):
            _action["cancel"] = True
            _action["running"] = None
            _action["progress"] = ""
            _action["pct"] = 0.0
            _action["start_time"] = None
            _action["result"] = ("warning", "Cancelled. Progress saved to database.")
            st.rerun(scope="app")
    elif _action["result"]:
        level, msg = _action["result"]
        if level == "success":
            st.success(msg)
        elif level == "warning":
            st.warning(msg)
        else:
            st.error(msg)
        if st.button("Dismiss", key="monitor_dismiss"):
            _action["result"] = None
            st.rerun(scope="app")


with st.sidebar:
    _action_monitor()

_action_is_live = bool(_action["running"])


# --- Background action helpers ---


class CancelledError(Exception):
    pass


def _check_cancel():
    if _action["cancel"]:
        _action["cancel"] = False
        raise CancelledError


def _update_progress(text: str, pct: float | None = None) -> None:
    _action["progress"] = text
    if pct is not None:
        _action["pct"] = min(max(pct, 0.0), 1.0)


def _do_fetch(playlist: str | None) -> str:
    from sweepify import db, spotify

    sp = spotify.get_client()
    source = f"playlist '{playlist}'" if playlist else "Liked Songs"
    _update_progress(f"Connecting to Spotify ({source})...", 0.0)

    def on_fetch(fetched: int, total: int) -> None:
        _check_cancel()
        pct = fetched / total if total else 0
        _update_progress(f"Fetching: {fetched}/{total} songs", pct * 0.6)

    def on_genre(processed: int, total: int) -> None:
        _check_cancel()
        pct = processed / total if total else 0
        _update_progress(f"Genres: {processed}/{total} artists", 0.6 + pct * 0.35)

    songs = spotify.fetch_liked_songs(
        sp,
        playlist=playlist,
        on_progress=on_fetch,
        on_genre_progress=on_genre,
    )

    _update_progress("Saving to database...", 0.95)
    count = db.upsert_songs(songs)
    return f"Fetched {len(songs)} song(s). Added {count} new to database."


def _do_enrich(force: bool) -> str:
    from sweepify import db, enricher
    from sweepify.classifier import get_client

    if force:
        db.reset_enrichments()

    songs = db.get_unenriched_songs()
    if not songs:
        return "No unenriched songs found."

    total = len(songs)
    total_batches = (total + enricher.BATCH_SIZE - 1) // enricher.BATCH_SIZE
    _update_progress(f"Enriching: 0/{total} songs", 0.0)
    client = get_client()
    enriched_count = 0
    batch_status: dict[int, str] = {}  # batch_num -> "processing" | "done"

    def _batch_text() -> str:
        if total_batches <= 1:
            return ""
        parts = []
        for b in sorted(batch_status):
            label = f"Batch {b}/{total_batches}"
            parts.append(f"{label} \u2713" if batch_status[b] == "done" else label)
        return "\n" + " \u00b7 ".join(parts)

    def on_batch_start(batch_num: int, total: int, size: int) -> None:
        batch_status[batch_num] = "processing"
        _update_progress(f"Enriching: {enriched_count}/{total_batches * enricher.BATCH_SIZE} songs{_batch_text()}", enriched_count / total if total else 0)

    def on_batch_done(batch_num: int, result: enricher.EnrichmentResult) -> None:
        nonlocal enriched_count
        _check_cancel()
        db.mark_enriched(
            [
                {
                    "spotify_id": e.spotify_id,
                    "mood": e.mood,
                    "bpm": e.bpm,
                    "vibe": e.vibe,
                    "related_artists": json.dumps(e.related_artists),
                }
                for e in result.songs
            ],
        )
        enriched_count += len(result.songs)
        batch_status[batch_num] = "done"
        _update_progress(f"Enriching: {enriched_count}/{total} songs{_batch_text()}", enriched_count / total)

    enricher.enrich_songs(client, songs, on_batch_start=on_batch_start, on_batch_done=on_batch_done)
    return f"Enriched {enriched_count} song(s)."


def _do_classify(max_playlists: int) -> str:
    from sweepify import classifier, db, spotify

    songs = db.get_unclassified_songs()
    if not songs:
        return "No unclassified songs found."

    fixed_categories = None
    if max_playlists == 0:
        sp = spotify.get_client()
        existing = spotify.fetch_sweepify_playlists(sp)
        if not existing:
            return "No existing sweepify playlists found. Use max > 0 to create new ones."
        fixed_categories = list(existing.keys())

    total = len(songs)
    total_batches = (total + classifier.BATCH_SIZE - 1) // classifier.BATCH_SIZE
    _update_progress(f"Classifying: 0/{total_batches} batches", 0.0)
    client = classifier.get_client()
    batches_done = 0
    classified_songs: set[str] = set()
    batch_status: dict[int, str] = {}

    def _batch_text() -> str:
        if total_batches <= 1:
            return ""
        parts = []
        for b in sorted(batch_status):
            label = f"Batch {b}/{total_batches}"
            parts.append(f"{label} \u2713" if batch_status[b] == "done" else label)
        return "\n" + " \u00b7 ".join(parts)

    def on_batch_start(batch_num: int, total: int, size: int) -> None:
        _check_cancel()
        batch_status[batch_num] = "processing"
        _update_progress(
            f"Classifying: {batches_done}/{total_batches} batches{_batch_text()}",
            batches_done / total_batches if total_batches else 0,
        )

    def on_batch_done(batch_num: int, result: classifier.ClassificationResult) -> None:
        nonlocal batches_done
        _check_cancel()
        batches_done += 1
        batch_status[batch_num] = "done"
        for cat in result.categories:
            db.mark_classified(cat.song_ids, cat.name, playlist_id="")
            classified_songs.update(cat.song_ids)
        _update_progress(
            f"Classifying: {batches_done}/{total_batches} batches{_batch_text()}",
            batches_done / total_batches if total_batches else 0,
        )

    classifier.classify_songs(
        client,
        songs,
        on_batch_start=on_batch_start,
        on_batch_done=on_batch_done,
        max_playlists=max_playlists,
        fixed_categories=fixed_categories,
    )
    return f"Classified {len(classified_songs)} song(s) into {len(set(batch_status))} batch(es). Run Refine to consolidate."


def _do_refine(max_playlists: int) -> str:
    from sweepify import classifier, db

    songs_by_cat = db.get_songs_by_category()
    if not songs_by_cat:
        return "No unrefined classifications found. Run Classify first."

    rough_categories = [
        classifier.Category(
            name=name,
            description=f"{len(cat_songs)} songs",
            song_ids=[s.spotify_id for s in cat_songs],
        )
        for name, cat_songs in songs_by_cat.items()
    ]

    all_song_ids = []
    for cat in rough_categories:
        all_song_ids.extend(cat.song_ids)
    unique_ids = list(set(all_song_ids))

    _update_progress(f"Refining {len(rough_categories)} categories...", 0.1)
    client = classifier.get_client()

    _check_cancel()
    result = classifier.refine_categories(client, rough_categories, max_playlists=max_playlists)

    db.reset_classifications_for_songs(unique_ids)
    classified_songs: set[str] = set()
    for cat in result.categories:
        db.mark_classified(cat.song_ids, cat.name, playlist_id="")
        classified_songs.update(cat.song_ids)

    _update_progress("Refinement complete", 1.0)
    summary = ", ".join(f"{c.name} ({len(c.song_ids)})" for c in result.categories)
    return f"Refined into {len(result.categories)} categories ({len(classified_songs)} songs): {summary}"


def _do_create() -> str:
    from sweepify import db, spotify
    from sweepify.models import Playlist

    songs_by_cat = db.get_songs_by_category()
    if not songs_by_cat:
        return "No classified songs pending playlist creation."

    sp = spotify.get_client()
    remote_playlists = spotify.fetch_sweepify_playlists(sp)
    for category, pid in remote_playlists.items():
        db.upsert_playlist(Playlist(spotify_id=pid, name=category))

    total = len(songs_by_cat)
    _update_progress(f"Creating playlists: 0/{total}", 0.0)
    done = 0

    for category, songs in songs_by_cat.items():
        song_ids = [s.spotify_id for s in songs]
        existing = db.get_playlist_by_name(category)

        if existing:
            spotify.add_to_existing_playlist(sp, existing.spotify_id, song_ids)
            playlist_id = existing.spotify_id
        else:
            playlist_id = spotify.create_playlist(sp, category, song_ids)
            db.upsert_playlist(Playlist(spotify_id=playlist_id, name=category))

        db.mark_classified(song_ids, category, playlist_id)
        done += 1
        _update_progress(f"Creating playlists: {done}/{total}", done / total)
        _check_cancel()

    return f"Processed {total} playlist(s)."


def _do_clear() -> str:
    from sweepify import db, spotify

    sp = spotify.get_client()
    playlists = spotify.fetch_sweepify_playlists(sp)
    if not playlists:
        count = db.reset_classifications()
        return f"No sweepify playlists on Spotify. Reset {count} local classification(s)."

    total = len(playlists)
    _update_progress(f"Deleting playlists: 0/{total}", 0.0)

    def on_progress(deleted: int, total: int) -> None:
        _check_cancel()
        _update_progress(f"Deleting playlists: {deleted}/{total}", deleted / total)

    deleted = spotify.delete_sweepify_playlists(sp, on_progress=on_progress)
    count = db.reset_classifications()
    return f"Deleted {len(deleted)} playlist(s). Reset {count} classification(s)."


def _do_full_pipeline(playlist: str | None, max_playlists: int) -> str:
    steps = [
        ("1/5 Fetch", _do_fetch, [playlist]),
        ("2/5 Enrich", _do_enrich, [False]),
        ("3/5 Classify", _do_classify, [max_playlists]),
        ("4/5 Refine", _do_refine, [max_playlists]),
        ("5/5 Create", _do_create, []),
    ]
    results = []
    for label, fn, fn_args in steps:
        _action["progress"] = f"Step {label}..."
        _action["pct"] = 0.0
        msg = fn(*fn_args)
        results.append(f"{label}: {msg}")
    return " | ".join(results)


def _run_action(name: str, action, *args):
    """Start an action in a background thread."""
    if _action["running"]:
        st.warning(f"**{_action['running']}** is already running. Stop it first.")
        return

    _action["gen"] += 1
    my_gen = _action["gen"]
    _action["running"] = name
    _action["cancel"] = False
    _action["progress"] = "Starting..."
    _action["pct"] = 0.0
    _action["result"] = None
    _action["start_time"] = time.monotonic()

    def _worker():
        try:
            msg = action(*args)
            if _action["gen"] == my_gen:
                _action["result"] = ("success", msg)
        except CancelledError:
            if _action["gen"] == my_gen:
                _action["result"] = ("warning", "Cancelled. Progress saved to database.")
        except Exception as e:
            if _action["gen"] == my_gen:
                _action["result"] = ("error", f"Error: {e}")
        finally:
            if _action["gen"] == my_gen:
                _action["running"] = None
                _action["progress"] = ""
                _action["pct"] = 0.0
                _action["start_time"] = None

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    _action["thread"] = thread
    st.rerun()


# --- Main area ---

if view == "Actions":
    st.header("Actions")
    st.caption("Run pipeline steps directly from the UI.")

    # Show inline progress when on this tab (auto-refreshes via fragment)
    @st.fragment(run_every=timedelta(seconds=2) if _action_is_live else None)
    def _inline_progress():
        if _action["running"]:
            eta = _format_eta()
            progress_text = _action["progress"] or "Working..."
            if eta:
                progress_text += f" — {eta}"
            st.progress(_action["pct"], text=progress_text)

    _inline_progress()

    # --- Pipeline steps ---
    st.subheader("Pipeline")

    pipeline_cols = st.columns(5)
    _disabled = _action_is_live  # Disable buttons while action is running

    with pipeline_cols[0]:
        st.markdown("**1. Fetch**")
        st.caption("Pull songs from Spotify")
        fetch_playlist = st.text_input("Playlist (optional)", key="fetch_playlist", placeholder="Liked Songs")
        if st.button("Fetch", use_container_width=True, type="primary", disabled=_disabled):
            _run_action("Fetch", _do_fetch, fetch_playlist or None)

    with pipeline_cols[1]:
        st.markdown("**2. Enrich**")
        st.caption("Add AI metadata (mood, BPM, vibe)")
        enrich_force = st.checkbox("Re-enrich all", key="enrich_force")
        if st.button("Enrich", use_container_width=True, type="primary", disabled=_disabled):
            _run_action("Enrich", _do_enrich, enrich_force)

    with pipeline_cols[2]:
        st.markdown("**3. Classify**")
        st.caption("Rough-classify songs in parallel")
        classify_max = st.number_input("Max playlists", min_value=0, max_value=30, value=10, key="classify_max")
        if st.button("Classify", use_container_width=True, type="primary", disabled=_disabled):
            _run_action("Classify", _do_classify, int(classify_max))

    with pipeline_cols[3]:
        st.markdown("**4. Refine**")
        st.caption("Consolidate rough categories")
        refine_max = st.number_input("Max playlists", min_value=1, max_value=30, value=10, key="refine_max")
        if st.button("Refine", use_container_width=True, type="primary", disabled=_disabled):
            _run_action("Refine", _do_refine, int(refine_max))

    with pipeline_cols[4]:
        st.markdown("**5. Create**")
        st.caption("Push playlists to Spotify")
        if st.button("Create Playlists", use_container_width=True, type="primary", disabled=_disabled):
            _run_action("Create", _do_create)

    st.divider()

    # --- Full pipeline ---
    st.subheader("Full Pipeline")
    st.caption("Run all steps in sequence: fetch, enrich, classify, refine, create.")
    run_cols = st.columns([2, 1, 1])
    with run_cols[0]:
        run_playlist = st.text_input("Playlist (optional)", key="run_playlist", placeholder="Liked Songs")
    with run_cols[1]:
        run_max = st.number_input("Max playlists", min_value=1, max_value=30, value=10, key="run_max")

    with run_cols[2]:
        st.write("")  # spacer
        st.write("")  # spacer
        if st.button("Run Full Pipeline", use_container_width=True, type="primary", disabled=_disabled):
            _run_action("Pipeline", _do_full_pipeline, run_playlist or None, int(run_max))

    st.divider()

    # --- Maintenance ---
    st.subheader("Maintenance")
    maint_cols = st.columns(3)

    with maint_cols[0]:
        st.markdown("**Reset**")
        st.caption("Clear classifications (keeps songs)")
        reset_enrichment = st.checkbox("Also reset enrichment", key="reset_enrichment")
        if st.button("Reset", use_container_width=True, disabled=_disabled):
            from sweepify import db as _db

            count = _db.reset_classifications()
            msg = f"Reset {count} song(s). Playlists cleared."
            if reset_enrichment:
                enrich_count = _db.reset_enrichments()
                msg += f" Reset enrichment for {enrich_count} song(s)."
            _action["result"] = ("success", msg)
            st.rerun()

    with maint_cols[1]:
        st.markdown("**Clear**")
        st.caption("Delete playlists from Spotify and reset")
        if st.button("Clear All", use_container_width=True, disabled=_disabled):
            _run_action("Clear", _do_clear)

    with maint_cols[2]:
        st.markdown("**Status**")
        st.caption("Refresh sidebar metrics")
        if st.button("Refresh", use_container_width=True):
            st.rerun()

elif view == "Songs":
    st.header("Songs")
    df = load_table("songs")

    if df.empty:
        st.info("No songs yet. Run `sweepify fetch` to get started.")
    else:
        # Parse categories JSON for filtering
        def _parse_categories(val):
            if not val:
                return []
            try:
                return json.loads(val)
            except (json.JSONDecodeError, TypeError):
                return []

        all_cats: set[str] = set()
        for c in df["categories"].dropna():
            all_cats.update(_parse_categories(c))

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            search = st.text_input("Search (name or artist)")
        with col2:
            categories = ["All"] + sorted(all_cats)
            cat_filter = st.selectbox("Category", categories)
        with col3:
            artists = ["All"] + sorted(df["artist"].dropna().unique().tolist())
            artist_filter = st.selectbox("Artist", artists)
        with col4:
            moods = ["All"] + sorted(df["mood"].dropna().unique().tolist())
            mood_filter = st.selectbox("Mood", moods)

        filtered = df.copy()
        if search:
            mask = filtered["name"].str.contains(search, case=False, na=False) | filtered["artist"].str.contains(
                search,
                case=False,
                na=False,
            )
            filtered = filtered[mask]
        if cat_filter != "All":
            filtered = filtered[filtered["categories"].apply(lambda v: cat_filter in _parse_categories(v))]
        if artist_filter != "All":
            filtered = filtered[filtered["artist"] == artist_filter]
        if mood_filter != "All":
            filtered = filtered[filtered["mood"] == mood_filter]

        st.dataframe(filtered, use_container_width=True, hide_index=True)
        st.caption(f"Showing {len(filtered)} of {len(df)} songs")

elif view == "Categories":
    st.header("Category Breakdown")
    df = load_table("songs")

    if df.empty:
        st.info("No songs yet. Run `sweepify fetch` to get started.")
    else:
        classified_df = df[df["classified"] == 1]
        if classified_df.empty:
            st.info("No classified songs yet. Run `sweepify classify` to get started.")
        else:
            # Parse categories JSON into rows
            def _parse_cat_list(val):
                if not val:
                    return []
                try:
                    return json.loads(val)
                except (json.JSONDecodeError, TypeError):
                    return []

            classified_df = classified_df.copy()
            classified_df["cat_list"] = classified_df["categories"].apply(_parse_cat_list)
            exploded = classified_df.explode("cat_list").dropna(subset=["cat_list"])
            cat_counts = exploded["cat_list"].value_counts().reset_index()
            cat_counts.columns = ["category", "songs"]

            # Summary metrics
            m_cols = st.columns(4)
            m_cols[0].metric("Categories", len(cat_counts))
            m_cols[1].metric("Classified Songs", len(classified_df))
            m_cols[2].metric("Avg Songs/Category", round(cat_counts["songs"].mean(), 1) if not cat_counts.empty else 0)
            multi = classified_df[classified_df["cat_list"].apply(len) > 1]
            m_cols[3].metric("Multi-Category Songs", len(multi))

            col1, col2 = st.columns(2)

            with col1:
                # Bar chart: songs per category
                st.subheader("Songs per Category")
                fig_bar = px.bar(
                    cat_counts,
                    x="songs",
                    y="category",
                    orientation="h",
                    color="songs",
                    color_continuous_scale=["#1A1E2E", "#1DB954"],
                )
                fig_bar.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    yaxis={"categoryorder": "total ascending"},
                    coloraxis_showscale=False,
                    margin={"l": 0, "r": 0, "t": 10, "b": 0},
                    height=max(400, len(cat_counts) * 35),
                )
                st.plotly_chart(fig_bar, use_container_width=True, theme="streamlit")

            with col2:
                # Pie chart: category distribution
                st.subheader("Category Distribution")
                fig_pie = px.pie(
                    cat_counts,
                    names="category",
                    values="songs",
                    color_discrete_sequence=px.colors.sequential.Emrld,
                )
                fig_pie.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    margin={"l": 0, "r": 0, "t": 10, "b": 0},
                    height=max(400, len(cat_counts) * 35),
                )
                st.plotly_chart(fig_pie, use_container_width=True, theme="streamlit")

            # Mood breakdown per category (if enriched)
            enriched_exploded = exploded[exploded["enriched"] == 1]
            if not enriched_exploded.empty and "mood" in enriched_exploded.columns:
                st.subheader("Mood by Category")
                mood_cat = enriched_exploded.groupby(["cat_list", "mood"]).size().reset_index(name="count")
                mood_cat.columns = ["category", "mood", "count"]
                fig_mood = px.bar(
                    mood_cat,
                    x="count",
                    y="category",
                    color="mood",
                    orientation="h",
                    barmode="stack",
                )
                fig_mood.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    yaxis={"categoryorder": "total ascending"},
                    margin={"l": 0, "r": 0, "t": 10, "b": 0},
                    height=max(400, len(cat_counts) * 35),
                    legend={"orientation": "h", "y": -0.15},
                )
                st.plotly_chart(fig_mood, use_container_width=True, theme="streamlit")

            # Browse by category
            st.subheader("Browse by Category")
            selected_cat = st.selectbox(
                "Select a category",
                cat_counts["category"].tolist(),
            )
            if selected_cat:
                cat_songs = exploded[exploded["cat_list"] == selected_cat][
                    ["name", "artist", "album", "mood", "vibe", "bpm"]
                ]
                st.dataframe(cat_songs, use_container_width=True, hide_index=True)
                st.caption(f"{len(cat_songs)} songs in '{selected_cat}'")

elif view == "Genres":
    st.header("Genre Breakdown")
    df = load_table("songs")

    if df.empty:
        st.info("No songs yet. Run `sweepify fetch` to get started.")
    else:
        # Explode genres JSON into one row per song-genre pair
        def parse_genres(genres_str):
            if not genres_str:
                return []
            try:
                return json.loads(genres_str)
            except (json.JSONDecodeError, TypeError):
                return []

        df["genre_list"] = df["genres"].apply(parse_genres)
        exploded = df.explode("genre_list").dropna(subset=["genre_list"])
        exploded = exploded[exploded["genre_list"] != ""]

        if exploded.empty:
            st.info("No genre data available.")
        else:
            genre_counts = exploded["genre_list"].value_counts()

            # Top genres bar chart
            max_genres = min(50, len(genre_counts))
            top_n = st.slider(
                "Top N genres",
                min_value=min(5, max_genres),
                max_value=max_genres,
                value=min(20, max_genres),
            )
            top_genres = genre_counts.head(top_n).reset_index()
            top_genres.columns = ["genre", "songs"]

            st.subheader(f"Top {top_n} Genres")
            fig_bar = px.bar(
                top_genres,
                x="songs",
                y="genre",
                orientation="h",
                color="songs",
                color_continuous_scale=["#1A1E2E", "#1DB954"],
            )
            fig_bar.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                yaxis={"categoryorder": "total ascending"},
                coloraxis_showscale=False,
                margin={"l": 0, "r": 0, "t": 10, "b": 0},
                height=max(400, top_n * 28),
            )
            st.plotly_chart(fig_bar, use_container_width=True, theme="streamlit")

            # Genre treemap
            st.subheader("Genre Map")
            top_tree = genre_counts.head(30).reset_index()
            top_tree.columns = ["genre", "songs"]
            fig_tree = px.treemap(
                top_tree,
                path=["genre"],
                values="songs",
                color="songs",
                color_continuous_scale=["#1A1E2E", "#1DB954", "#A0E8AF"],
            )
            fig_tree.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                coloraxis_showscale=False,
                margin={"l": 0, "r": 0, "t": 10, "b": 0},
                height=500,
            )
            st.plotly_chart(fig_tree, use_container_width=True, theme="streamlit")

            # Genre table with song counts
            st.subheader("All Genres")
            genre_df = genre_counts.reset_index()
            genre_df.columns = ["genre", "songs"]
            st.dataframe(genre_df, use_container_width=True, hide_index=True)
            st.caption(f"{len(genre_df)} unique genres across {len(df)} songs")

            # Songs by selected genre
            st.subheader("Browse by Genre")
            selected_genre = st.selectbox(
                "Select a genre",
                genre_counts.index.tolist(),
            )
            if selected_genre:
                genre_songs = exploded[exploded["genre_list"] == selected_genre][
                    ["name", "artist", "album", "release_date"]
                ]
                st.dataframe(
                    genre_songs,
                    use_container_width=True,
                    hide_index=True,
                )
                st.caption(
                    f"{len(genre_songs)} songs in '{selected_genre}'",
                )

elif view == "Enrichment":
    st.header("Enrichment Overview")
    df = load_table("songs")

    if df.empty:
        st.info("No songs yet. Run `sweepify fetch` to get started.")
    else:
        enriched_df = df[df["enriched"] == 1]
        if enriched_df.empty:
            st.info("No enriched songs yet. Run `sweepify enrich` to get started.")
        else:
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Mood Distribution")
                mood_counts = enriched_df["mood"].value_counts().reset_index()
                mood_counts.columns = ["mood", "songs"]
                fig_mood = px.bar(
                    mood_counts,
                    x="songs",
                    y="mood",
                    orientation="h",
                    color="songs",
                    color_continuous_scale=["#1A1E2E", "#1DB954"],
                )
                fig_mood.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    yaxis={"categoryorder": "total ascending"},
                    coloraxis_showscale=False,
                    margin={"l": 0, "r": 0, "t": 10, "b": 0},
                    height=max(300, len(mood_counts) * 30),
                )
                st.plotly_chart(fig_mood, use_container_width=True, theme="streamlit")

            with col2:
                st.subheader("BPM Distribution")
                bpm_data = enriched_df["bpm"].dropna()
                if not bpm_data.empty:
                    fig_bpm = px.histogram(
                        enriched_df,
                        x="bpm",
                        nbins=30,
                        color_discrete_sequence=["#1DB954"],
                    )
                    fig_bpm.update_layout(
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        margin={"l": 0, "r": 0, "t": 10, "b": 0},
                        height=300,
                    )
                    st.plotly_chart(fig_bpm, use_container_width=True, theme="streamlit")

            # Vibe word cloud / table
            st.subheader("Top Vibes")
            vibe_counts = enriched_df["vibe"].value_counts().head(20).reset_index()
            vibe_counts.columns = ["vibe", "songs"]
            st.dataframe(vibe_counts, use_container_width=True, hide_index=True)

            st.caption(f"{len(enriched_df)} of {len(df)} songs enriched")

elif view == "Playlists":
    st.header("Playlists")
    df = load_table("playlists")

    if df.empty:
        st.info("No playlists yet. Run `sweepify create` to generate them.")
    else:
        st.dataframe(df, use_container_width=True, hide_index=True)

elif view == "Playlist Builder":
    st.header("Playlist Builder")
    df = load_table("songs")

    if df.empty:
        st.info("No songs yet. Run `sweepify fetch` to get started.")
    else:
        # Parse all available genres
        all_genres: set[str] = set()
        for g in df["genres"].dropna():
            try:
                all_genres.update(json.loads(g))
            except (json.JSONDecodeError, TypeError):
                pass
        sorted_genres = sorted(all_genres)

        # Existing sweepify playlists
        playlists_df = load_table("playlists")
        if not playlists_df.empty:
            st.subheader("Existing Sweepify Playlists")
            # Explode categories JSON to count songs per category
            cat_counts: dict[str, int] = {}
            for cats_str in df["categories"].dropna():
                try:
                    for c in json.loads(cats_str):
                        cat_counts[c] = cat_counts.get(c, 0) + 1
                except (json.JSONDecodeError, TypeError):
                    pass
            songs_per_playlist = pd.DataFrame(
                [{"category": k, "songs": v} for k, v in cat_counts.items()]
            ) if cat_counts else pd.DataFrame(columns=["category", "songs"])
            playlist_info = playlists_df.merge(
                songs_per_playlist,
                left_on="name",
                right_on="category",
                how="left",
            )[["name", "spotify_id", "songs"]].fillna(0)
            playlist_info["songs"] = playlist_info["songs"].astype(int)
            st.dataframe(playlist_info, use_container_width=True, hide_index=True)

        st.subheader("Build Command")

        selected_genres = st.multiselect(
            "Select genres",
            sorted_genres,
            help="Songs matching any of these genres will be included.",
        )

        col1, col2 = st.columns(2)
        with col1:
            playlist_name = st.text_input(
                "Playlist name (optional)",
                help="Leave empty to let Claude pick a name.",
            )
        with col2:
            max_playlists = st.number_input(
                "Max playlists",
                min_value=1,
                max_value=20,
                value=1,
            )

        # Preview matching songs
        if selected_genres:
            matching = get_songs_by_genres(selected_genres)
            st.caption(f"{len(matching)} song(s) match the selected genres")

            if matching:
                match_df = pd.DataFrame([{"name": s.name, "artist": s.artist, "genres": s.genres} for s in matching])
                with st.expander(f"Preview ({len(matching)} songs)"):
                    st.dataframe(match_df, use_container_width=True, hide_index=True)

            # Build the command
            genre_arg = ", ".join(selected_genres)
            cmd = f'uv run sweepify playlist -g "{genre_arg}"'
            if playlist_name:
                cmd += f' --name "{playlist_name}"'
            if max_playlists > 1:
                cmd += f" -n {max_playlists}"

            st.subheader("Command")
            st.code(cmd, language="bash")
        else:
            st.caption("Select at least one genre to preview matching songs.")

elif view == "SQL":
    st.header("SQL Playground")
    query = st.text_area(
        "Query",
        value=(
            "SELECT j.value as category, COUNT(*) as count\n"
            "FROM songs, json_each(categories) as j\n"
            "WHERE classified = 1\n"
            "GROUP BY j.value\nORDER BY count DESC;"
        ),
        height=120,
    )
    if st.button("Run"):
        try:
            with get_connection() as conn:
                conn.execute("BEGIN")
                result = pd.read_sql_query(query, conn)
                conn.execute("ROLLBACK")
            st.dataframe(result, use_container_width=True, hide_index=True)
            st.caption(f"{len(result)} row(s)")
        except Exception as e:
            st.error(str(e))
