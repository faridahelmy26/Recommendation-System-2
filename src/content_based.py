import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List


class ContentBased:
    """Simple content-based recommender wrapper.

    Usage:
        cb = ContentBased(df)
        cb.recommend("Some title", k=5)
    """

    def __init__(self, df: pd.DataFrame, text_cols: List[str] = ["title", "description"]):
        self.df = df.copy()
        for c in text_cols:
            if c not in self.df.columns:
                self.df[c] = ""
            else:
                self.df[c] = self.df[c].fillna("")

        self.df["combined"] = self.df[text_cols].agg(" ".join, axis=1)
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=10000)
        self.tfidf = self.vectorizer.fit_transform(self.df["combined"]) 

    def recommend(self, title: str, k: int = 5) -> List[str]:
        text = str(title).lower().strip()
        if not text:
            return []

        input_vec = self.vectorizer.transform([text])
        sims_to_input = cosine_similarity(input_vec, self.tfidf).flatten()
        if sims_to_input.max() < 0.05:
            return []

        best_idx = int(sims_to_input.argmax())
        item_sims = cosine_similarity(self.tfidf[best_idx], self.tfidf).flatten()
        order = item_sims.argsort()[::-1]

        recs = []
        for idx in order:
            if len(recs) >= k:
                break
            if int(idx) == best_idx:
                continue
            recs.append(self.df.iloc[int(idx)]["title"])

        return recs
