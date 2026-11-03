from pathlib import Path

import pandas as pd


# Locate the project and dataset
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# Load movie data
movies = pd.read_csv(DATA_DIR / "movies.csv")

# Display sample movies
print("\n--- SAMPLE MOVIES ---")
print(movies[["title", "genres"]].head(10).to_string(index=False))

# Split each movie's genres into a list
movies["genre_list"] = movies["genres"].apply(
    lambda genres: genres.split("|")
    if isinstance(genres, str) else []
)

# Count how many movies belong to each genre
genre_counts = (


    
    movies.explode("genre_list")["genre_list"]
    .value_counts()
)

print("\n--- MOVIE COUNT BY GENRE ---")
print(genre_counts.to_string())

# Count movies with no genres
no_genre_count = movies["genre_list"].apply(
    lambda genres: len(genres) == 0
).sum()

print("\nMovies without genres:", no_genre_count)

# Show movies belonging to a selected genre
selected_genre = "Animation"

animation_movies = movies[
    movies["genre_list"].apply(
        lambda genres: selected_genre in genres
    )
]

print(f"\n--- SAMPLE {selected_genre.upper()} MOVIES ---")
print(animation_movies[["title", "genres"]].head(10).to_string(index=False))
