import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


def _norm(value: Any) -> str:
    """Normalize a string for case/whitespace-insensitive comparison."""
    return str(value).strip().lower()


class HybridRecommender:
    """
    Hybrid recommender combining content-based and collaborative filtering.

    Fixes vs. the original version:
    - `collab_recs` used to be built but never populated, so collaborative
      items never made it into the merged candidate pool. Now the items
      returned by `collaborative_recommender.recommend_for_user` are used
      directly as both candidates and score source.
    - Cold-start users (no interaction history) now get recommendations
      based on their stated interest/level/age from `users_df`, instead of
      generic popularity.
    - Weights are adaptive: users with little/no history lean on
      content + profile similarity; users with more history lean on
      collaborative filtering.
    """

    def __init__(
        self,
        content_recommender,
        collaborative_recommender,
        users_df: Optional[pd.DataFrame] = None,
        interactions_df: Optional[pd.DataFrame] = None,
        content_weight: float = 0.5,
        collab_weight: float = 0.5,
        adaptive_weights: bool = True
    ):
        self.content_recommender = content_recommender
        self.collaborative_recommender = collaborative_recommender
        self.users_df = users_df
        # Prefer the live interactions_df passed in directly; fall back to
        # whatever's attached to collaborative_recommender only if the
        # caller didn't provide one (e.g. older call sites, tests).
        self.interactions_df = interactions_df
        self.content_weight = content_weight
        self.collab_weight = collab_weight
        self.adaptive_weights = adaptive_weights

        logger.info(
            f"✅ HybridRecommender initialized with weights: "
            f"content={content_weight}, collab={collab_weight}, adaptive={adaptive_weights}"
        )

    # =========================
    # Weighting
    # =========================

    def _get_user_history(self, user_id: int) -> List[int]:
        """Get user's interaction history (content_ids)."""
        # Bug fix: this used to read exclusively from
        # `collaborative_recommender.interactions`, a snapshot frozen at
        # whatever moment that model was last trained/loaded. The
        # collaborative *model* (SVD factors) is allowed to lag behind by
        # design — retraining is expensive — but "what has this user
        # already seen" is a simple fact that should always reflect the
        # live interactions.csv, otherwise `total_interactions` reported
        # by the API and this method's own internal count silently
        # diverge for the same user (observed: 38 vs 44).
        interactions = self.interactions_df
        if interactions is None and self.collaborative_recommender is not None:
            interactions = getattr(self.collaborative_recommender, 'interactions', None)

        if interactions is not None and not interactions.empty and 'content_id' in interactions.columns:
            return interactions[interactions['user_id'] == user_id]['content_id'].tolist()
        return []

    def _compute_weights(self, n_interactions: int) -> Dict[str, float]:
        """
        Adaptive Hybrid: users with little/no history get more weight on
        content-based scoring (and profile-based cold start), users with
        richer history get more weight on collaborative filtering.
        """
        if not self.adaptive_weights:
            return {'content': self.content_weight, 'collab': self.collab_weight}

        if n_interactions == 0:
            return {'content': 0.9, 'collab': 0.1}
        elif n_interactions < 5:
            return {'content': 0.7, 'collab': 0.3}
        else:
            return {'content': 0.3, 'collab': 0.7}

    # =========================
    # Cold start
    # =========================

    def _get_user_profile(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Look up a user's interest/level/age from users_df, if available."""
        if self.users_df is None or self.users_df.empty:
            return None
        if 'user_id' not in self.users_df.columns:
            return None

        row = self.users_df[self.users_df['user_id'] == user_id]
        if row.empty:
            return None

        row = row.iloc[0]
        profile = {}
        for col in ('interest', 'level', 'age', 'learning_style'):
            if col in row.index:
                profile[col] = row[col]
        return profile or None

    def _cold_start_recommend(
        self,
        user_id: int,
        top_n: int,
        filter_category: Optional[str] = None,
        filter_level: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Real cold-start handling: match the user's stated interest/level
        against content categories/levels rather than falling back to
        plain global popularity.
        """
        profile = self._get_user_profile(user_id)
        content_df = getattr(self.content_recommender, 'df', None)

        if profile is None or content_df is None or content_df.empty:
            df = content_df if content_df is not None else pd.DataFrame()
            if not df.empty:
                if filter_category and 'category' in df.columns:
                    df = df[df['category'].map(_norm) == _norm(filter_category)]
                if filter_level and 'level' in df.columns:
                    df = df[df['level'].map(_norm) == _norm(filter_level)]

            if df.empty:
                # A filter was requested and genuinely matches nothing —
                # returning unrelated items here would be worse than
                # returning none; the caller (app.py) still has its own
                # ranked fallback to fall back to if it wants one.
                return []

            if 'rating' in df.columns:
                df = df.sort_values('rating', ascending=False)

            recommendations = []
            for _, row in df.head(top_n).iterrows():
                recommendations.append({
                    'content_id': int(row['content_id']),
                    'title': row.get('title', f"Item {row['content_id']}"),
                    'category': row.get('category', ''),
                    'level': row.get('level', ''),
                    'score': float(row.get('rating', 0.5)) / 5.0 if 'rating' in row else 0.5,
                    'content_score': 0.5,
                    'collab_score': 0.0,
                    'source': 'cold_start_no_profile'
                })
            return recommendations

        candidates = content_df.copy()

        interest = profile.get('interest')
        level = profile.get('level')

        # Prefer exact interest + level match, then relax to interest only,
        # then fall back to the full catalog.
        matched = candidates
        if interest and 'category' in candidates.columns:
            by_interest = candidates[candidates['category'].map(_norm) == _norm(interest)]
            if not by_interest.empty:
                matched = by_interest
        if level and 'level' in matched.columns:
            by_level = matched[matched['level'].map(_norm) == _norm(level)]
            if not by_level.empty:
                matched = by_level

        if filter_category and 'category' in matched.columns:
            matched = matched[matched['category'].map(_norm) == _norm(filter_category)]
        if filter_level and 'level' in matched.columns:
            matched = matched[matched['level'].map(_norm) == _norm(filter_level)]

        if matched.empty:
            matched = candidates

        # Rank by rating if present, else keep catalog order
        if 'rating' in matched.columns:
            matched = matched.sort_values('rating', ascending=False)

        recommendations = []
        for _, row in matched.head(top_n).iterrows():
            recommendations.append({
                'content_id': int(row['content_id']),
                'title': row.get('title', f"Item {row['content_id']}"),
                'category': row.get('category', ''),
                'level': row.get('level', ''),
                'score': float(row.get('rating', 0.6)) / 5.0 if 'rating' in row else 0.6,
                'content_score': 0.6,
                'collab_score': 0.0,
                'source': 'cold_start_profile'
            })

        logger.info(f"🧊 Cold-start recommendations for user {user_id} using profile {profile}")
        return recommendations

    # =========================
    # Main recommend
    # =========================

    def recommend(
        self,
        user_id: int,
        top_n: int = 5,
        exclude_seen: bool = True,
        filter_category: Optional[str] = None,
        filter_level: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get hybrid recommendations for a user.
        """
        # Exposed after every call so callers (e.g. the /recommend/hybrid
        # endpoint) can surface exactly what happened internally, instead
        # of silently falling back with no visibility into why.
        self.last_debug: Dict[str, Any] = {}

        try:
            user_history = self._get_user_history(user_id)
            weights = self._compute_weights(len(user_history))
            self.last_debug['user_history_count'] = len(user_history)
            self.last_debug['weights'] = weights

            # Cold start: no history at all -> use profile-based matching
            # as the primary signal instead of generic popularity.
            if not user_history:
                recs = self._cold_start_recommend(user_id, top_n, filter_category, filter_level)
                self.last_debug['path'] = 'cold_start'
                self.last_debug['cold_start_count'] = len(recs)
                if recs:
                    return recs
                # if no profile/content available either, fall through to
                # the regular popularity-based path below.

            # 1. Collaborative candidates + scores
            collab_recs: List[int] = []
            collab_scores: Dict[int, float] = {}

            if self.collaborative_recommender:
                try:
                    # When a category/level filter is active, a chunk of
                    # whatever top-k collab returns will get rejected later
                    # (rejected_category_filter/rejected_level_filter) since
                    # recommend_for_user() has no notion of content
                    # metadata. Widening the pool up front means a narrow
                    # filter is less likely to reject everything and fall
                    # back to the generic ranked list.
                    filter_active = bool(filter_category or filter_level)
                    collab_k = max(top_n * 4, 20) * (3 if filter_active else 1)

                    collab_items = self.collaborative_recommender.recommend_for_user(
                        user_id=user_id,
                        k=collab_k,
                        exclude_seen=exclude_seen
                    )
                    # Bug fix: previously collab_items were only used to
                    # populate collab_scores, never collab_recs, so they
                    # never entered the merged candidate pool below.
                    collab_recs = list(collab_items)

                    for item_id in collab_items:
                        score = self.collaborative_recommender.predict_rating(user_id, item_id)
                        collab_scores[item_id] = float(score) / 5.0 if score else 0.0

                    logger.info(f"✅ Got {len(collab_recs)} collaborative recommendations for user {user_id}")
                    self.last_debug['collab_recs_count'] = len(collab_recs)
                except Exception as e:
                    logger.warning(f"⚠️ Collaborative recommendation failed: {e}")
                    self.last_debug['collab_error'] = f"{type(e).__name__}: {e}"

            # 2. Content-based candidates + scores (seeded from most recent history item)
            content_recs: List[Dict[str, Any]] = []
            content_scores: Dict[int, float] = {}

            if self.content_recommender and user_history:
                try:
                    seed_item = user_history[-1]
                    self.last_debug['content_seed_item'] = seed_item
                    seen_set = set(user_history)

                    # content_recommender.recommend() only excludes the
                    # seed item itself (exclude_self), not the rest of the
                    # user's history. For an active user with many
                    # interactions in a template-heavy catalog, requesting
                    # only top_n*2 candidates meant every single one could
                    # already be seen — leaving zero survivors after the
                    # exclude_seen filter below, and silently degrading to
                    # the generic ranked fallback. Requesting a much wider
                    # pool and filtering seen items out here (rather than
                    # only at the very end) fixes that.
                    candidate_k = max(top_n * 10, len(seen_set) + top_n)
                    if filter_category or filter_level:
                        candidate_k *= 3

                    content_filter_condition = {}
                    if filter_category:
                        content_filter_condition['category'] = filter_category
                    if filter_level:
                        content_filter_condition['level'] = filter_level

                    recs = self.content_recommender.recommend(
                        query=seed_item,
                        k=candidate_k,
                        return_scores=True,
                        exclude_self=True,
                        filter_condition=content_filter_condition or None
                    )
                    self.last_debug['content_recs_raw_count'] = len(recs) if recs else 0

                    # If filtering by category/level at the seed item's own
                    # neighborhood found nothing (e.g. the seed item itself
                    # sits in a different category than what's requested),
                    # fall back to a filtered semantic search over the
                    # whole catalog instead of giving up on content-based
                    # candidates entirely.
                    if not recs and content_filter_condition and hasattr(self.content_recommender, 'search_by_text'):
                        try:
                            item_details = self._get_item_details(seed_item)
                            seed_text = item_details.get('title', '') if item_details else ''
                            recs = self.content_recommender.search_by_text(
                                text=seed_text,
                                k=candidate_k,
                                return_scores=True,
                                filter_condition=content_filter_condition
                            )
                            self.last_debug['content_recs_filtered_search_count'] = len(recs) if recs else 0
                        except Exception as e:
                            self.last_debug['content_filtered_search_error'] = f"{type(e).__name__}: {e}"

                    if recs:
                        if exclude_seen:
                            recs = [
                                r for r in recs
                                if isinstance(r, dict) and r.get('content_id') not in seen_set
                            ]
                        content_recs = recs[:top_n * 2]
                        for rec in content_recs:
                            if isinstance(rec, dict) and 'content_id' in rec:
                                content_scores[rec['content_id']] = rec.get('similarity_score', 0.5)
                    logger.info(f"✅ Got {len(content_recs)} content recommendations")
                    self.last_debug['content_recs_count'] = len(content_recs)
                except Exception as e:
                    logger.warning(f"⚠️ Content recommendation failed: {e}")
                    self.last_debug['content_error'] = f"{type(e).__name__}: {e}"

            # 3. Merge candidate pool
            all_item_ids = set(collab_recs)
            for rec in content_recs:
                if isinstance(rec, dict) and 'content_id' in rec:
                    all_item_ids.add(rec['content_id'])
            self.last_debug['merged_candidate_count'] = len(all_item_ids)

            if not all_item_ids:
                logger.warning(f"⚠️ No recommendations found for user {user_id}, using popular items")
                all_item_ids = set(self._get_popular_items(top_n * 2))

            # 4. Build final recommendations with adaptive weighting
            recommendations = []
            rejected_no_details = 0
            rejected_category = 0
            rejected_level = 0

            self.last_debug['filter_category'] = filter_category
            self.last_debug['filter_level'] = filter_level

            for item_id in all_item_ids:
                content_score = content_scores.get(item_id, 0.0)
                collab_score = collab_scores.get(item_id, 0.0)

                if content_score == 0 and collab_score == 0:
                    item_row = self._get_item_details(item_id)
                    if item_row:
                        content_score = 0.3

                hybrid_score = weights['content'] * content_score + weights['collab'] * collab_score

                item_details = self._get_item_details(item_id)
                if not item_details:
                    rejected_no_details += 1
                    continue

                if filter_category and _norm(item_details.get('category', '')) != _norm(filter_category):
                    rejected_category += 1
                    continue
                if filter_level and _norm(item_details.get('level', '')) != _norm(filter_level):
                    rejected_level += 1
                    continue

                recommendations.append({
                    'content_id': int(item_id),
                    'title': item_details.get('title', f'Item {item_id}'),
                    'category': item_details.get('category', ''),
                    'level': item_details.get('level', ''),
                    'score': float(hybrid_score),
                    'content_score': float(content_score),
                    'collab_score': float(collab_score),
                    'source': 'hybrid'
                })

            self.last_debug['rejected_no_details'] = rejected_no_details
            self.last_debug['rejected_category_filter'] = rejected_category
            self.last_debug['rejected_level_filter'] = rejected_level
            self.last_debug['recs_before_exclude_seen'] = len(recommendations)

            recommendations = sorted(recommendations, key=lambda x: x['score'], reverse=True)

            if exclude_seen:
                before = len(recommendations)
                recommendations = [r for r in recommendations if r['content_id'] not in user_history]
                self.last_debug['excluded_as_already_seen'] = before - len(recommendations)
                logger.info(f"📍 Excluded {len(user_history)} seen items")

            final_recs = recommendations[:top_n]
            self.last_debug['final_recs_count'] = len(final_recs)
            logger.info(f"✅ Returning {len(final_recs)} recommendations for user {user_id}")

            return final_recs

        except Exception as e:
            logger.error(f"❌ Error in hybrid recommendation: {e}")
            self.last_debug['outer_error'] = f"{type(e).__name__}: {e}"
            return []

    def _get_item_details(self, item_id: int) -> Optional[Dict[str, Any]]:
        if self.content_recommender and hasattr(self.content_recommender, 'df'):
            df = self.content_recommender.df
            if 'content_id' in df.columns:
                # str() comparison avoids silent zero-match issues when
                # content_id dtypes differ slightly between content.csv
                # and interactions.csv (e.g. Int64 vs int64 vs object).
                item = df[df['content_id'].astype(str) == str(item_id)]
                if not item.empty:
                    row = item.iloc[0]
                    return {
                        'content_id': int(row['content_id']),
                        'title': row.get('title', f'Item {item_id}'),
                        'category': row.get('category', ''),
                        'level': row.get('level', '')
                    }
        return None

    def _get_popular_items(self, n: int = 10) -> List[int]:
        if self.collaborative_recommender and hasattr(self.collaborative_recommender, 'interactions'):
            interactions = self.collaborative_recommender.interactions
            if interactions is not None and not interactions.empty:
                popular = interactions.groupby('content_id').size().sort_values(ascending=False)
                return popular.head(n).index.tolist()

        if self.content_recommender and hasattr(self.content_recommender, 'df'):
            df = self.content_recommender.df
            if 'content_id' in df.columns:
                return df['content_id'].head(n).tolist()

        return []

    def _get_popular_item_details(self, n: int = 10) -> List[Dict[str, Any]]:
        ids = self._get_popular_items(n)
        details = []
        for item_id in ids:
            d = self._get_item_details(item_id)
            if d:
                details.append(d)
        return details