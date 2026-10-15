import pytest


def titles(recs):
    return [r["title"] for r in recs]


def test_seed_is_never_recommended(toy_predictor):
    recs = toy_predictor.recommend(book_title="Pride and Prejudice", n=5)
    assert "Pride and Prejudice" not in titles(recs)


def test_most_similar_books_come_first(toy_predictor):
    recs = toy_predictor.recommend(book_title="Pride and Prejudice", n=2)
    assert {r["authors"] for r in recs} == {"Jane Austen"}


def test_duplicate_editions_are_collapsed(toy_predictor):
    recs = toy_predictor.recommend(book_title="Dune", n=5)
    normalised = [" ".join(t.lower().split()) for t in titles(recs)]
    assert len(normalised) == len(set(normalised))


def test_min_rating_applies_to_recommendations_not_just_seed(toy_predictor):
    recs = toy_predictor.recommend(book_title="Dune", min_rating=4.0, n=5)
    assert recs and all(r["average_rating"] >= 4.0 for r in recs)


def test_language_filter_applies_to_recommendations(toy_predictor):
    recs = toy_predictor.recommend(book_title="Dune", language="fre", n=5)
    assert titles(recs) == ["Le Petit Prince"]


def test_unknown_book_raises(toy_predictor):
    with pytest.raises(ValueError):
        toy_predictor.recommend(book_title="zzzzqq")


def test_no_arguments_raises(toy_predictor):
    with pytest.raises(ValueError):
        toy_predictor.recommend()


def test_top_by_genre_filters_and_limits(toy_predictor):
    recs = toy_predictor.top_by_genre(genre="Romance", n=2)
    assert len(recs) == 2 and all(r["genre"] == "Romance" for r in recs)