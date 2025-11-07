# agents/music_agent.py
from langchain.tools import tool
from utils.database import db

def _escape_sql_string(value: str) -> str:
    return value.replace("'", "''")

@tool 
def get_info_about_track(track_name: str):
    """Get all information about a track/song by its name."""
    escaped = _escape_sql_string(track_name)
    #example: "Let There Be Rock"
    return db.run(
        f"SELECT Track.UnitPrice, Track.Composer, Album.Title, Genre.Name as GenreName, MediaType.Name as MediaTypeName FROM Track LEFT JOIN Album ON Track.AlbumId = Album.AlbumId LEFT JOIN Genre ON Track.GenreId = Genre.GenreId LEFT JOIN MediaType ON Track.MediaTypeId = MediaType.MediaTypeId WHERE Track.Name LIKE '%{escaped}%';",
        include_columns=True
    )


@tool
def get_albums_by_artist(artist: str):
    """Get albums by an artist."""


    #example: exception 
    raise Exception("This is a test exception")
    escaped = _escape_sql_string(artist)
    return db.run(
        f"""
        SELECT Album.Title, Artist.Name 
        FROM Album 
        JOIN Artist ON Album.ArtistId = Artist.ArtistId 
        WHERE Artist.Name LIKE '%{escaped}%';
        """,
        include_columns=True
    )

@tool
def get_tracks_by_artist(artist: str):
    """Get songs by an artist (or similar artists)."""
    escaped = _escape_sql_string(artist)
    return db.run(
        f"""
        SELECT Track.Name as SongName, Artist.Name as ArtistName 
        FROM Album 
        LEFT JOIN Artist ON Album.ArtistId = Artist.ArtistId 
        LEFT JOIN Track ON Track.AlbumId = Album.AlbumId 
        WHERE Artist.Name LIKE '%{escaped}%';
        """,
        include_columns=True
    )

# @tool
# def check_for_songs(song_title: str):
#     """Check if a song exists by its name."""
#     escaped = _escape_sql_string(song_title)
#     return db.run(
#         f"SELECT * FROM Track WHERE Name LIKE '%{escaped}%';",
#         include_columns=True
#     )

music_system_prompt = """You help customers find songs and albums.
Use tools to search. If a lookup returns no exact matches, suggest similar artists/tracks."""
