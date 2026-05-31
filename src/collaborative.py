import pandas as pd
from typing import List


class Collaborative:
    """Simple collaborative placeholder recommender.

    This module contains a very small implementation for popularity-based
    fallback recommendations. Replace with matrix-factorization or
    neighborhood methods as needed.
    """

    def __init__(self, interactions: pd.DataFrame | None = None):
        self.interactions = interactions

    def fit(self, interactions: pd.DataFrame):
        self.interactions = interactions

    def recommend_for_user(self, user_id: int, k: int = 5) -> List[int]:
        if self.interactions is None or self.interactions.empty:
            return []
        top = self.interactions["content_id"].value_counts().head(k).index.tolist()
        return top
