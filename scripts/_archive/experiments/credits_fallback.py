"""
Fallback credits data for top MovieLens movies.

This provides a curated list of actors/directors for the most popular movies
when TMDB API is not available.

For full credits, set TMDB_API_KEY and run fetch_tmdb_credits.py
"""

# Top actors appearing in popular movies (manually curated for top ~100 movies)
TOP_ACTORS = [
    "Tom Hanks", "Morgan Freeman", "Leonardo DiCaprio", "Brad Pitt",
    "Robert De Niro", "Al Pacino", "Denzel Washington", "Matt Damon",
    "Harrison Ford", "Samuel L. Jackson", "Liam Neeson", "Christian Bale",
    "Tom Cruise", "Johnny Depp", "Will Smith", "Mark Wahlberg",
    "Keanu Reeves", "Bruce Willis", "Arnold Schwarzenegger", "Sylvester Stallone",
    "Robin Williams", "Jim Carrey", "Adam Sandler", "Ben Stiller",
    "Owen Wilson", "Vince Vaughn", "Bradley Cooper", "Ryan Gosling",
    "Joaquin Phoenix", "Jake Gyllenhaal", "Edward Norton", "Kevin Spacey",
    "John Travolta", "Nicolas Cage", "Russell Crowe", "Hugh Jackman",
    "Michael Caine", "Anthony Hopkins", "Ian McKellen", "Patrick Stewart",
    "Meryl Streep", "Julia Roberts", "Sandra Bullock", "Angelina Jolie",
    "Scarlett Johansson", "Natalie Portman", "Anne Hathaway", "Emma Stone",
    "Jennifer Lawrence", "Cate Blanchett"
]

# Top directors of popular movies
TOP_DIRECTORS = [
    "Steven Spielberg", "Christopher Nolan", "Martin Scorsese", "Quentin Tarantino",
    "James Cameron", "Ridley Scott", "David Fincher", "Peter Jackson",
    "Clint Eastwood", "Ron Howard", "Robert Zemeckis", "Tim Burton",
    "Joel Coen", "Ethan Coen", "Wes Anderson", "Denis Villeneuve",
    "Francis Ford Coppola", "Stanley Kubrick", "Alfred Hitchcock", "Woody Allen",
    "Guy Ritchie", "Michael Bay", "M. Night Shyamalan", "Darren Aronofsky"
]

# Movie -> actors mapping for top movies (partial - will be expanded with TMDB)
MOVIE_ACTORS = {
    "Forrest Gump (1994)": ["Tom Hanks", "Robin Wright", "Gary Sinise"],
    "Shawshank Redemption, The (1994)": ["Tim Robbins", "Morgan Freeman"],
    "Pulp Fiction (1994)": ["John Travolta", "Samuel L. Jackson", "Uma Thurman"],
    "Silence of the Lambs, The (1991)": ["Jodie Foster", "Anthony Hopkins"],
    "Matrix, The (1999)": ["Keanu Reeves", "Laurence Fishburne", "Carrie-Anne Moss"],
    "Schindler's List (1993)": ["Liam Neeson", "Ralph Fiennes", "Ben Kingsley"],
    "Star Wars: Episode IV - A New Hope (1977)": ["Mark Hamill", "Harrison Ford", "Carrie Fisher"],
    "Jurassic Park (1993)": ["Sam Neill", "Laura Dern", "Jeff Goldblum"],
    "Lord of the Rings: The Fellowship of the Ring, The (2001)": ["Elijah Wood", "Ian McKellen", "Viggo Mortensen"],
    "Fight Club (1999)": ["Brad Pitt", "Edward Norton", "Helena Bonham Carter"],
    "Inception (2010)": ["Leonardo DiCaprio", "Joseph Gordon-Levitt", "Ellen Page"],
    "Dark Knight, The (2008)": ["Christian Bale", "Heath Ledger", "Aaron Eckhart"],
    "Goodfellas (1990)": ["Robert De Niro", "Ray Liotta", "Joe Pesci"],
    "Usual Suspects, The (1995)": ["Kevin Spacey", "Gabriel Byrne", "Benicio del Toro"],
    "Se7en (1995)": ["Brad Pitt", "Morgan Freeman", "Kevin Spacey"],
    "Saving Private Ryan (1998)": ["Tom Hanks", "Matt Damon", "Tom Sizemore"],
    "Terminator 2: Judgment Day (1991)": ["Arnold Schwarzenegger", "Linda Hamilton", "Edward Furlong"],
    "Braveheart (1995)": ["Mel Gibson", "Sophie Marceau", "Patrick McGoohan"],
    "American Beauty (1999)": ["Kevin Spacey", "Annette Bening", "Thora Birch"],
    "Toy Story (1995)": ["Tom Hanks", "Tim Allen", "Don Rickles"],
}

# Movie -> directors mapping
MOVIE_DIRECTORS = {
    "Forrest Gump (1994)": ["Robert Zemeckis"],
    "Shawshank Redemption, The (1994)": ["Frank Darabont"],
    "Pulp Fiction (1994)": ["Quentin Tarantino"],
    "Silence of the Lambs, The (1991)": ["Jonathan Demme"],
    "Matrix, The (1999)": ["Lana Wachowski", "Lilly Wachowski"],
    "Schindler's List (1993)": ["Steven Spielberg"],
    "Star Wars: Episode IV - A New Hope (1977)": ["George Lucas"],
    "Jurassic Park (1993)": ["Steven Spielberg"],
    "Lord of the Rings: The Fellowship of the Ring, The (2001)": ["Peter Jackson"],
    "Fight Club (1999)": ["David Fincher"],
    "Inception (2010)": ["Christopher Nolan"],
    "Dark Knight, The (2008)": ["Christopher Nolan"],
    "Goodfellas (1990)": ["Martin Scorsese"],
    "Usual Suspects, The (1995)": ["Bryan Singer"],
    "Se7en (1995)": ["David Fincher"],
    "Saving Private Ryan (1998)": ["Steven Spielberg"],
    "Terminator 2: Judgment Day (1991)": ["James Cameron"],
    "Braveheart (1995)": ["Mel Gibson"],
    "American Beauty (1999)": ["Sam Mendes"],
    "Toy Story (1995)": ["John Lasseter"],
}


def get_fallback_actors() -> list:
    """Get list of top actors."""
    return TOP_ACTORS.copy()


def get_fallback_directors() -> list:
    """Get list of top directors."""
    return TOP_DIRECTORS.copy()


def get_movie_actors(movie_title: str) -> list:
    """Get actors for a specific movie."""
    return MOVIE_ACTORS.get(movie_title, [])


def get_movie_director(movie_title: str) -> list:
    """Get director(s) for a specific movie."""
    return MOVIE_DIRECTORS.get(movie_title, [])
