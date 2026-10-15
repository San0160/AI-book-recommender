import joblib
import pandas as pd

from Book_recommender.config.configuration import configurationManager
from Book_recommender.logging.log import logger


class RecommenderPredictor:
    OVERSAMPLE = 20  # fetch extra neighbours so filters still leave n results

    def __init__(self):
        logger.info("Loading artifacts...")
        cfg = configurationManager()
        trainer_cfg = cfg.get_model_trainer_config()
        eval_cfg = cfg.get_model_evaluation_config()

        self.model = joblib.load(trainer_cfg.model_path)
        self.features = joblib.load(eval_cfg.feature_matrix_path)
        self.data = pd.read_csv(trainer_cfg.data_path).reset_index(drop=True)
        self.data["_norm_title"] = (
            self.data["title"].astype(str).str.lower().str.split().str.join(" ")
        )

        # Bayesian score computed once, not on every request
        C = self.data["average_rating"].mean()
        m = self.data["ratings_count"].quantile(0.75)
        v = self.data["ratings_count"]
        self.data["score"] = (v / (v + m)) * self.data["average_rating"] + (m / (v + m)) * C

        logger.info(f"Loaded {len(self.data)} books with {type(self.model).__name__}")

    # ---------- helpers ----------
    def _filter_mask(self, language=None, min_rating=0.0, genre=None):
        mask = pd.Series(True, index=self.data.index)
        if language:
            mask &= self.data["language_code"].str.strip().eq(language)
        if min_rating:
            mask &= self.data["average_rating"] >= min_rating
        if genre:
            mask &= self.data["genre_clean"].str.contains(
                genre, case=False, na=False, regex=False
            )
        return mask.to_numpy()

    def _find_seed(self, book_title=None, author=None):
        col, query = ("title", book_title) if book_title else ("authors", author)
        matches = self.data[
            self.data[col].str.contains(query, case=False, na=False, regex=False)
        ]
        if matches.empty:
            raise ValueError(f"Book not found: {query}")
        exact = matches[matches[col].str.lower() == query.lower()]
        pool = exact if not exact.empty else matches
        return pool["ratings_count"].idxmax()  # most-rated match, not first row

    def _format(self, indices):
        rows = self.data.iloc[list(indices)]
        return [
            {
                "title": str(r["title"]),
                "authors": str(r["authors"]),
                "average_rating": float(r["average_rating"]),
                "genre": str(r["genre_clean"]) if pd.notna(r["genre_clean"]) else "",
                "cover_url": f"https://covers.openlibrary.org/b/isbn/{r['isbn13']}-M.jpg",
                "isbn13": str(r["isbn13"]),
            }
            for _, r in rows.iterrows()
        ]

    # ---------- public API ----------
    def recommend(self, book_title=None, author=None, genre=None,
                  min_rating=0.0, language="eng", n=5):
        if not book_title and not author and not genre:
            raise ValueError("Provide at least one of: book_title, author, genre")

        if not book_title and not author:
            return self.top_by_genre(genre=genre, min_rating=min_rating,
                                     language=language, n=n)

        seed = self._find_seed(book_title, author)
        seed_title = self.data.at[seed, "_norm_title"]
        logger.info(
            f"Seed book: {self.data.at[seed, 'title']} "
            f"(language={self.data.at[seed, 'language_code']})"
        )

        mask = self._filter_mask(language, min_rating, genre)

        n_candidates = min(len(self.data), n * self.OVERSAMPLE + 1)
        _, indices = self.model.kneighbors(
            self.features[[seed]], n_neighbors=n_candidates
        )

        picked, seen_titles = [], {seed_title}
        for i in indices.ravel():
            title = self.data.at[i, "_norm_title"]
            if i == seed or not mask[i] or title in seen_titles:
                continue  # skip self, filtered-out books, duplicate editions
            seen_titles.add(title)
            picked.append(i)
            if len(picked) == n:
                break

        return self._format(picked)

    def top_by_genre(self, genre, min_rating=0.0, language=None, n=5):
        mask = self._filter_mask(language, min_rating, genre)
        if not mask.any():
            raise ValueError(f"No books found for genre: {genre}")
        top = self.data[mask].nlargest(n, "score")
        return self._format(top.index)

    def popular_random(self, n=15, pool=500, language="eng"):
        """n random books drawn from the best-rated `pool` books (changes every call)."""
        mask = self._filter_mask(language=language)
        candidates = self.data[mask].nlargest(pool, "score")
        picks = candidates.sample(min(n, len(candidates)))
        return self._format(picks.index)

    def model_info(self):
        return {
            "algorithm": type(self.model).__name__,
            "books": len(self.data),
        }