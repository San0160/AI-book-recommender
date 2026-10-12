from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors


class SvdNeighbors:
    """TruncatedSVD (LSA) followed by nearest-neighbour search.

    Exposes the same kneighbors() interface as NearestNeighbors, so
    evaluation and prediction work with either model unchanged.
    """

    def __init__(self, n_components=150, random_state=42, **nn_params):
        self.svd = TruncatedSVD(n_components=n_components, random_state=random_state)
        self.nn = NearestNeighbors(**nn_params)

    def fit(self, tfidf_matrix):
        reduced = self.svd.fit_transform(tfidf_matrix)
        self.nn.fit(reduced)
        return self

    def kneighbors(self, X, n_neighbors=None, return_distance=True):
        return self.nn.kneighbors(
            self.svd.transform(X),
            n_neighbors=n_neighbors,
            return_distance=return_distance,
        )