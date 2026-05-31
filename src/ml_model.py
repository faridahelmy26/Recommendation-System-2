from typing import Any


class MLModel:
    """Thin wrapper for ML models (placeholder).

    Implement `fit` and `predict` for a supervised model used in ranking or
    re-ranking pipelines.
    """

    def __init__(self):
        self.model: Any = None

    def fit(self, X, y):
        # placeholder: set a flag or train a scikit-learn/lightgbm model
        self.model = True

    def predict(self, X):
        if self.model is None:
            raise RuntimeError("Model not fitted")
        # placeholder: return zero scores
        return [0.0 for _ in range(len(X))]
