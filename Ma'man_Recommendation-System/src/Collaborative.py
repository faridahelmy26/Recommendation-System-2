import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import coo_matrix
from typing import List, Optional
import pickle
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CollaborativeRecommender:
    """
    Collaborative filtering recommender using Truncated SVD with sparse matrices.
    """

    def __init__(
        self,
        interactions: Optional[pd.DataFrame] = None,
        n_components: int = 50,
        random_state: int = 42,
        max_users: int = 5000,
        max_items: int = 2000
    ):
        self.interactions = interactions
        self.n_components = n_components
        self.random_state = random_state
        self.max_users = max_users
        self.max_items = max_items

        self.user_item_matrix = None
        self.user_factors = None
        self.item_factors = None
        self.svd = None
        self.popularity_scores = None
        self.user_map = {}
        self.item_map = {}
        self.reverse_user_map = {}
        self.reverse_item_map = {}

        if interactions is not None:
            self.fit(interactions)

    def fit(self, interactions: Optional[pd.DataFrame] = None):
        if interactions is not None:
            self.interactions = interactions

        if self.interactions is None or self.interactions.empty:
            logger.warning("⚠️ No interactions data to fit")
            return

        logger.info(f"🔄 Building collaborative model with {len(self.interactions)} interactions...")

        if len(self.interactions) > 500000:
            logger.info(f"📊 Sampling 500k interactions from {len(self.interactions)}")
            self.interactions = self.interactions.sample(n=500000, random_state=self.random_state)

        user_counts = self.interactions['user_id'].value_counts()
        item_counts = self.interactions['content_id'].value_counts()

        top_users = user_counts.head(self.max_users).index.tolist()
        top_items = item_counts.head(self.max_items).index.tolist()

        filtered_interactions = self.interactions[
            self.interactions['user_id'].isin(top_users) &
            self.interactions['content_id'].isin(top_items)
        ]

        logger.info(f"📊 Filtered to {len(filtered_interactions)} interactions")
        logger.info(f"   Users: {len(top_users)}, Items: {len(top_items)}")

        self.user_map = {user: i for i, user in enumerate(top_users)}
        self.item_map = {item: i for i, item in enumerate(top_items)}
        self.reverse_user_map = {i: user for user, i in self.user_map.items()}
        self.reverse_item_map = {i: item for item, i in self.item_map.items()}

        rows = filtered_interactions['user_id'].map(self.user_map).values
        cols = filtered_interactions['content_id'].map(self.item_map).values
        data = filtered_interactions['rating'].values

        self.user_item_matrix = coo_matrix(
            (data, (rows, cols)),
            shape=(len(top_users), len(top_items))
        )

        self.popularity_scores = filtered_interactions.groupby('content_id')['rating'].mean()

        n_users, n_items = self.user_item_matrix.shape
        n_components = min(self.n_components, n_items - 1, n_users - 1, 50)

        if n_components < 1:
            logger.warning(f"⚠️ Not enough data for SVD (users={n_users}, items={n_items})")
            return

        logger.info(f"🔄 Training SVD with {n_components} components...")
        logger.info(f"   Users: {n_users}, Items: {n_items}")

        self.user_item_csr = self.user_item_matrix.tocsr()

        self.svd = TruncatedSVD(n_components=n_components, random_state=self.random_state)

        self.user_factors = self.svd.fit_transform(self.user_item_csr)
        self.item_factors = self.svd.components_

        logger.info(f"✅ Collaborative model trained successfully!")
        logger.info(f"   Users: {len(self.user_factors)}")
        logger.info(f"   Items: {self.item_factors.shape[1]}")
        logger.info(f"   Components: {self.svd.n_components}")
        print(len(self.item_map))
        print(len(self.reverse_item_map))
        print(max(self.item_map.values()))

    def recommend_for_user(self, user_id: int, k: int = 5, exclude_seen: bool = True) -> List[int]:
        if user_id not in self.user_map or self.item_factors is None:
            logger.info(f"📚 User {user_id} not in training data, using fallback")
            return self._fallback_recommend(user_id, k)

        user_idx = self.user_map[user_id]
        user_vector = self.user_factors[user_idx]

        scores = np.dot(user_vector, self.item_factors)
        top_indices = scores.argsort()[::-1]

        seen_items = []
        if exclude_seen and self.interactions is not None:
            seen_items = self.interactions[self.interactions['user_id'] == user_id]['content_id'].tolist()

        recommendations = []
        for idx in top_indices:
            if len(recommendations) >= k:
                break

            content_id = self.reverse_item_map.get(idx)
            if content_id is None:
                continue

            if exclude_seen and content_id in seen_items:
                continue

            recommendations.append(content_id)

        logger.info(f"📚 Returning {len(recommendations)} recommendations for user {user_id}")
        return recommendations

    def predict_rating(self, user_id: int, content_id: int) -> float:

        print("=" * 50)
        print("User:", user_id)
        print("Item:", content_id)
        print("User exists:", user_id in self.user_map)
        print("Item exists:", content_id in self.item_map)

        if user_id not in self.user_map:
                print("❌ user not found")
                return 0.0

        if content_id not in self.item_map:
                print("❌ item not found")
                return 0.0

        user_idx = self.user_map[user_id]
        item_idx = self.item_map[content_id]

        print("user_idx =", user_idx)
        print("item_idx =", item_idx)

        prediction = np.dot(
                self.user_factors[user_idx],
                self.item_factors[:, item_idx]
        )

        print("raw prediction =", prediction)

        prediction = float(prediction)

        print("final prediction =", prediction)

        return prediction

    def _fallback_recommend(self, user_id: int, k: int = 5) -> List[int]:
        if self.popularity_scores is None or self.popularity_scores.empty:
            return []

        top_items = self.popularity_scores.sort_values(ascending=False).head(k * 2).index.tolist()

        if self.interactions is not None:
            seen_items = self.interactions[self.interactions['user_id'] == user_id]['content_id'].tolist()

            if seen_items:
                unseen_items = [item for item in top_items if item not in seen_items]
                return unseen_items[:k]

        return top_items[:k]

    def save(self, path: str):
        """Save the model to a directory."""
        os.makedirs(path, exist_ok=True)

        model_data = {
            'user_factors': self.user_factors,
            'item_factors': self.item_factors,
            'popularity_scores': self.popularity_scores,
            'svd': self.svd,
            'n_components': self.n_components,
            'random_state': self.random_state,
            'user_map': self.user_map,
            'item_map': self.item_map,
            'reverse_user_map': self.reverse_user_map,
            'reverse_item_map': self.reverse_item_map,
            'max_users': self.max_users,
            'max_items': self.max_items,
            'interactions': self.interactions
        }

        with open(os.path.join(path, 'collaborative_model.pkl'), 'wb') as f:
            pickle.dump(model_data, f)

        logger.info(f"✅ Collaborative model saved to {path}")

    def load(self, path: str):
        """Load a model from a directory (must match what `save()` produced)."""
        if path.endswith('.pkl'):
            path = os.path.dirname(path) or '.'

        with open(os.path.join(path, 'collaborative_model.pkl'), 'rb') as f:
            model_data = pickle.load(f)

        self.user_factors = model_data['user_factors']
        self.item_factors = model_data['item_factors']
        self.popularity_scores = model_data['popularity_scores']
        self.svd = model_data['svd']
        self.n_components = model_data.get('n_components', 50)
        self.random_state = model_data.get('random_state', 42)
        self.user_map = model_data.get('user_map', {})
        self.item_map = model_data.get('item_map', {})
        self.reverse_user_map = model_data.get('reverse_user_map', {})
        self.reverse_item_map = model_data.get('reverse_item_map', {})
        self.max_users = model_data.get('max_users', 5000)
        self.max_items = model_data.get('max_items', 2000)
        self.interactions = model_data.get('interactions')

        logger.info(f"✅ Collaborative model loaded from {path}")
        return self