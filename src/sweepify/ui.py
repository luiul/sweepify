import json

import pandas as pd
import plotly.express as px
import streamlit as st

from sweepify.config import DB_PATH
from sweepify.db import get_connection, get_status, init_db

COLORS = ["#1DB954", "#1ED760", "#2EBD59", "#57B660", "#A0E8AF",
          "#B497D6", "#E8A0BF", "#F2C57C", "#7EC8E3", "#FF6B6B"]

st.set_page_config(page_title="sweepify", page_icon="🧹", layout="wide")

init_db()


def load_table(table: str) -> pd.DataFrame:
    with get_connection() as conn:
        return pd.read_sql_query(f"SELECT * FROM {table}", conn)  # noqa: S608


# --- Sidebar ---

st.sidebar.title("sweepify")
st.sidebar.caption(f"DB: `{DB_PATH}`")

status = get_status()
st.sidebar.metric("Total songs", status["total"])
cols = st.sidebar.columns(2)
cols[0].metric("Classified", status["classified"])
cols[1].metric("Unclassified", status["unclassified"])
st.sidebar.metric("Categories", status["categories"])
st.sidebar.metric("Playlists", status["playlists"])

view = st.sidebar.radio("View", ["Songs", "Genres", "Playlists", "SQL"])

# --- Main area ---

if view == "Songs":
    st.header("Songs")
    df = load_table("songs")

    if df.empty:
        st.info("No songs yet. Run `sweepify fetch` to get started.")
    else:
        col1, col2, col3 = st.columns(3)
        with col1:
            search = st.text_input("Search (name or artist)")
        with col2:
            categories = ["All"] + sorted(df["category"].dropna().unique().tolist())
            cat_filter = st.selectbox("Category", categories)
        with col3:
            artists = ["All"] + sorted(df["artist"].dropna().unique().tolist())
            artist_filter = st.selectbox("Artist", artists)

        filtered = df.copy()
        if search:
            mask = (
                filtered["name"].str.contains(search, case=False, na=False)
                | filtered["artist"].str.contains(search, case=False, na=False)
            )
            filtered = filtered[mask]
        if cat_filter != "All":
            filtered = filtered[filtered["category"] == cat_filter]
        if artist_filter != "All":
            filtered = filtered[filtered["artist"] == artist_filter]

        st.dataframe(filtered, use_container_width=True, hide_index=True)
        st.caption(f"Showing {len(filtered)} of {len(df)} songs")

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
            top_n = st.slider(
                "Top N genres",
                min_value=5,
                max_value=min(50, len(genre_counts)),
                value=min(20, len(genre_counts)),
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
                "Select a genre", genre_counts.index.tolist()
            )
            if selected_genre:
                genre_songs = exploded[
                    exploded["genre_list"] == selected_genre
                ][["name", "artist", "album", "popularity", "release_date"]]
                st.dataframe(
                    genre_songs, use_container_width=True, hide_index=True
                )
                st.caption(
                    f"{len(genre_songs)} songs in '{selected_genre}'"
                )

elif view == "Playlists":
    st.header("Playlists")
    df = load_table("playlists")

    if df.empty:
        st.info("No playlists yet. Run `sweepify create` to generate them.")
    else:
        st.dataframe(df, use_container_width=True, hide_index=True)

elif view == "SQL":
    st.header("SQL Playground")
    query = st.text_area(
        "Query",
        value="SELECT category, COUNT(*) as count FROM songs\nWHERE classified = 1\nGROUP BY category\nORDER BY count DESC;",
        height=120,
    )
    if st.button("Run"):
        try:
            with get_connection() as conn:
                result = pd.read_sql_query(query, conn)
            st.dataframe(result, use_container_width=True, hide_index=True)
            st.caption(f"{len(result)} row(s)")
        except Exception as e:
            st.error(str(e))
