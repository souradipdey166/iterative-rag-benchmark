"""
TF-IDF embedding function -- no model download required. Chosen because of
unreliable/slow internet on this machine; trades semantic understanding
(TF-IDF only matches word overlap, not meaning) for being fully offline
and instant.

IMPORTANT: must be fit on the corpus (via indexing) BEFORE any queries are
made -- TF-IDF vectors only make sense within a fitted vocabulary.
"""

from sklearn.feature_extraction.text import TfidfVectorizer


class TfidfEmbedder:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self._fitted = False

    def __call__(self, texts: list[str]) -> list[list[float]]:
        if not self._fitted:
            vectors = self.vectorizer.fit_transform(texts)
            self._fitted = True
        else:
            vectors = self.vectorizer.transform(texts)
        return vectors.toarray().tolist()