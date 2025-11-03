from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from utils.database import db
from utils.model import model

#PART 2 - DEFINITION FOR MUSIC AGENT

def _escape_sql_string(value: str) -> str:
    """Escape single quotes in SQL strings by doubling them."""
    return value.replace("'", "''")

#TOOL: Get albums by artist
@tool
def get_albums_by_artist(artist: str):
    """Get albums by an artist."""
    escaped_artist = _escape_sql_string(artist)
    return db.run(
        f"""
        SELECT Album.Title, Artist.Name 
        FROM Album 
        JOIN Artist ON Album.ArtistId = Artist.ArtistId 
        WHERE Artist.Name LIKE '%{escaped_artist}%';
        """,
        include_columns=True
    )


#TOOL: getting tracks by an artist
@tool
def get_tracks_by_artist(artist: str):
    """Get songs by an artist (or similar artists)."""
    escaped_artist = _escape_sql_string(artist)
    return db.run(
        f"""
        SELECT Track.Name as SongName, Artist.Name as ArtistName 
        FROM Album 
        LEFT JOIN Artist ON Album.ArtistId = Artist.ArtistId 
        LEFT JOIN Track ON Track.AlbumId = Album.AlbumId 
        WHERE Artist.Name LIKE '%{escaped_artist}%';
        """,
        include_columns=True
    )

#TOOL: looking up songs by their name
@tool
def check_for_songs(song_title):
    """Check if a song exists by its name."""
    escaped_title = _escape_sql_string(song_title)
    return db.run(
        f"""
        SELECT * FROM Track WHERE Name LIKE '%{escaped_title}%';
        """,
        include_columns=True
    )

song_system_message = """Your job is to help a customer find any songs they are looking for. 

You only have certain tools you can use. If a customer asks you to look something up that you don't know how, politely tell them what you can help with.

When looking up artists and songs, sometimes the artist/song will not be found. In that case, the tools will return information \
on simliar songs and artists. This is intentional, it is not the tool messing up."""

def get_song_messages(messages):
    return [SystemMessage(content=song_system_message)] + messages

song_recc_chain = get_song_messages | model.bind_tools([get_albums_by_artist, get_tracks_by_artist, check_for_songs])
