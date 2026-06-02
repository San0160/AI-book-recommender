# pipeline/prediction.py
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from Book_recommender.logging.log import logger


class RecommenderPredictor:

    def __init__(self):
        logger.info("Loading artifacts...")

        self.knn = joblib.load("artifacts/model_trainer/knn_model.pkl")
        self.tfidf_matrix = joblib.load("artifacts/data_transformation/tfidf_matrix.pkl")
        self.vectorizer = joblib.load("artifacts/data_transformation/tfidf.pkl")
        self.data = pd.read_csv("artifacts/data_preprocessing/processed_data.csv")

        logger.info(f"Loaded {len(self.data)} books ✅")

    def recommend(self, book_title=None, author=None, genre=None,
              min_rating=0.0, language="eng", n=5):

        if not book_title and not author and not genre:
            raise ValueError("Provide at least one of: book_title, author, genre")

        # Step 1 — filter
        filtered = self.data.copy()
        if language:
            filtered = filtered[filtered["language_code"].str.strip() == language]
        if min_rating:
            filtered = filtered[filtered["average_rating"] >= min_rating]
        if genre:
            filtered = filtered[
                filtered["genre_clean"].str.contains(genre, case=False, na=False, regex=False)
            ]

        if filtered.empty:
            raise ValueError("No books found with those filters")

        # Step 2 — find seed
        if book_title:
            seed = filtered[filtered["title"].str.contains(
                book_title, case=False, na=False, regex=False
            )]
        elif author:
            seed = filtered[filtered["authors"].str.contains(
                author, case=False, na=False, regex=False
            )]
        else:
            # only genre/rating given — return top by bayesian score
            return self.top_by_genre(genre=genre, min_rating=min_rating, n=n)

        if seed.empty:
            raise ValueError(f"Book not found: {book_title or author}")

        idx = seed.index[0]
        logger.info(f"Seed book: {self.data.loc[idx, 'title']}")

        # Step 3 — KNN
        distances, indices = self.knn.kneighbors(
            self.tfidf_matrix[idx],
            n_neighbors=n + 1
        )
        book_indices = [i for i in indices.flatten() if i != idx][:n]

        return self._format_results(book_indices)

    # Step 4 — format
    def _format_results(self, book_indices):
        results = []
        for i in book_indices:
            row = self.data.iloc[i]
            results.append({
                "title":          str(row["title"]),
                "authors":        str(row["authors"]),
                "average_rating": float(row["average_rating"]),
                "genre":          str(row["genre_clean"]) if pd.notna(row["genre_clean"]) else "",
                "cover_url":      f"https://covers.openlibrary.org/b/isbn/{row['isbn13']}-M.jpg",
                "isbn13":         str(row["isbn13"])
            })
        return results

    def top_by_genre(self, genre: str, min_rating: float = 0.0, n: int = 5):
        filtered = self.data.copy()

        if genre:
            filtered = filtered[
                filtered["genre_clean"].str.contains(genre, case=False, na=False, regex=False)
            ]
        if min_rating:
            filtered = filtered[filtered["average_rating"] >= min_rating]

        if filtered.empty:
            raise ValueError(f"No books found for genre: {genre}")

        # bayesian score
        C = self.data["average_rating"].mean()
        m = self.data["ratings_count"].quantile(0.75)

        filtered = filtered.copy()
        filtered["score"] = filtered.apply(
            lambda row: (row["ratings_count"] / (row["ratings_count"] + m)) * row["average_rating"] +
                        (m / (row["ratings_count"] + m)) * C,
            axis=1
        )

        top = filtered.sort_values("score", ascending=False).head(n)

        return [
            {
                "title":          str(row["title"]),
                "authors":        str(row["authors"]),
                "average_rating": float(row["average_rating"]),
                "genre":          str(row["genre_clean"]) if pd.notna(row["genre_clean"]) else "",
                "cover_url":      f"https://covers.openlibrary.org/b/isbn/{row['isbn13']}-M.jpg",
                "isbn13":         str(row["isbn13"])
            }
            for _, row in top.iterrows()
        ]