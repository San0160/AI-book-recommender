import pandas as pd
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from Book_recommender.pipeline.prediction import RecommenderPredictor, _canon_lang

# title, authors, genre, rating, ratings_count, language, description
BOOKS = [
    ("Dune", "Frank Herbert", "Science Fiction", 4.2, 5000, "eng", "desert planet spice empire prophecy sandworm"),
    ("Dune Messiah", "Frank Herbert", "Science Fiction", 3.9, 3000, "eng", "desert planet emperor prophecy spice"),
    ("Dune  Messiah", "Frank Herbert", "Science Fiction", 3.9, 200, "eng", "desert planet emperor prophecy spice"),
    ("Children of Dune", "Frank Herbert", "Science Fiction", 4.0, 2500, "en-US", "desert planet children prophecy spice"),
    ("Foundation", "Isaac Asimov", "Science Fiction", 4.1, 4000, "eng", "galactic empire psychohistory future decline"),
    ("Neuromancer", "William Gibson", "Science Fiction", 3.9, 2000, "eng", "cyberspace hacker future corporation"),
    ("Pride and Prejudice", "Jane Austen", "Romance", 4.3, 6000, "eng", "marriage society england courtship manners"),
    ("Emma", "Jane Austen", "Romance", 3.7, 1500, "eng", "matchmaking society england manners marriage"),
    ("Sense and Sensibility", "Jane Austen", "Romance", 4.1, 1800, "eng", "sisters marriage society england inheritance"),
    ("Gone Girl", "Gillian Flynn", "Thriller", 4.0, 5500, "eng", "marriage disappearance detective secrets"),
    ("Le Petit Prince", "Saint-Exupery", "Fiction", 4.4, 4500, "fre", "prince planet fox rose child"),
    ("The Hobbit", "J.R.R. Tolkien", "Fantasy", 4.3, 7000, "eng", "dragon quest hobbit mountain ring"),
]


@pytest.fixture(scope="session")
def toy_predictor():
    data = pd.DataFrame(
        BOOKS,
        columns=["title", "authors", "genre_clean", "average_rating",
                 "ratings_count", "language_code", "description"],
    )
    data["_lang"] = data["language_code"].map(_canon_lang)
    data["isbn13"] = [9780000000000 + i for i in range(len(data))]
    data["combined_features"] = (
        data["title"] + " " + data["authors"] + " " + data["description"]
    ).str.lower()
    data["_norm_title"] = data["title"].str.lower().str.split().str.join(" ")

    v, r = data["ratings_count"], data["average_rating"]
    C, m = r.mean(), v.quantile(0.75)
    data["score"] = (v / (v + m)) * r + (m / (v + m)) * C

    features = TfidfVectorizer().fit_transform(data["combined_features"])
    model = NearestNeighbors(metric="cosine", algorithm="brute").fit(features)

    # Build the predictor without touching artifacts on disk
    predictor = RecommenderPredictor.__new__(RecommenderPredictor)
    predictor.model, predictor.features, predictor.data = model, features, data
    return predictor