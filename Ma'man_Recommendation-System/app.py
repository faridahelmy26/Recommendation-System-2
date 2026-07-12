import sys
import os
import traceback
from pathlib import Path

# Add the 'src' folder to Python path (this is where your code is)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
import numpy as np
from fastapi import FastAPI, Query, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Union
import pickle
import logging
from datetime import datetime
import uvicorn
import json
import hashlib

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Import from the 'src' folder
try:
    from DataLoader import DataLoader
    from DataCleaner import DataCleaner
    from ContentRecommender import ContentBased
    from Collaborative import CollaborativeRecommender
    from MLModel import MLModel
    from recommender import HybridRecommender
    logger.info("✅ All modules imported successfully from src/")
except ImportError as e:
    logger.error(f"❌ Import error: {e}")

    class DataLoader:
        def __init__(self, data_dir="data"):
            self.data_dir = data_dir

        def load_content(self, path):
            return pd.read_csv(path)

        def load_interactions(self, path):
            return pd.read_csv(path)

        def load_users(self, path):
            return pd.read_excel(path) if os.path.exists(path) else pd.DataFrame()

    class DataCleaner:
        def clean(self, df):
            return df

    class ContentBased:
        def __init__(self, df=None, text_cols=None, use_embeddings=False):
            self.df = df

        def recommend(self, query, k=5, **kwargs):
            return []

        def recommend_by_title(self, title, k=5, **kwargs):
            return []

        def save(self, path):
            pass

        def load(self, path):
            return self

    class CollaborativeRecommender:
        def __init__(self, interactions=None, n_components=50):
            self.interactions = interactions

        def fit(self, interactions=None):
            pass

        def recommend_for_user(self, user_id, k=5, **kwargs):
            return []

        def predict_rating(self, user_id, content_id):
            return 0.0

        def save(self, path):
            pass

        def load(self, path):
            return self

    class MLModel:
        def __init__(self, **kwargs):
            pass

    class HybridRecommender:
        def __init__(self, **kwargs):
            pass

        def recommend(self, user_id, top_n=5, **kwargs):
            return []

# Create necessary directories
os.makedirs('logs', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('cache', exist_ok=True)

def _matches(value: Any, target: Optional[str]) -> bool:
    """Case/whitespace-insensitive equality check for category/level filters."""
    if target is None:
        return True
    return str(value).strip().lower() == target.strip().lower()


# Directories used for persisted models (directories, not filenames -
# ContentBased.save/load and CollaborativeRecommender.save/load both
# expect a directory path, not a .pkl filename)
CONTENT_MODEL_DIR = "models/content_based"
COLLAB_MODEL_DIR = "models/collaborative"

# =========================
# FastAPI App
# =========================
app = FastAPI(
    title="Hybrid Recommendation System API",
    description="Advanced recommendation system with content-based and collaborative filtering",
    version="2.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# Pydantic Models
# =========================
class HybridRequest(BaseModel):
    # Union[int, str]: your platform's real student IDs are MongoDB
    # ObjectId strings (see handoff note), not plain integers. Pydantic
    # tries int first — so numeric test IDs like 239 still work exactly
    # as before — and falls back to str for real ObjectIds. Everything
    # downstream (dict lookups, DataFrame equality) works with either
    # since nothing requires user_id to be numeric internally.
    user_id: Union[int, str] = Field(..., description="User ID (integer for test data, ObjectId string in production)")
    top_n: int = Field(5, description="Number of recommendations", ge=1, le=50)
    include_scores: bool = Field(True, description="Include scores")
    exclude_seen: bool = Field(True, description="Exclude seen items")
    filter_category: Optional[str] = Field(None, description="Filter by category")
    filter_level: Optional[str] = Field(None, description="Filter by level")


class ContentRequest(BaseModel):
    title: str = Field(..., description="Course title", min_length=1)
    top_n: int = Field(5, description="Number of recommendations", ge=1, le=50)
    filter_category: Optional[str] = Field(None, description="Filter by category")
    filter_level: Optional[str] = Field(None, description="Filter by level")


class RetrainRequest(BaseModel):
    interactions_path: Optional[str] = Field(None, description="Path to new interactions file")
    force_retrain: bool = Field(False, description="Force retrain")


# =========================
# Global State
# =========================
class AppState:
    def __init__(self):
        self.content_df = None
        self.interactions_df = None
        self.users_df = None
        self.content_recommender = None
        self.collaborative_recommender = None
        self.hybrid_recommender = None
        self.ml_model = None
        self.is_loaded = False
        self.last_update = None
        self.cache = {}
        self.cache_timestamps = {}


state = AppState()


# =========================
# Cache Functions
# =========================
def get_cache_key(user_id: int, top_n: int, **kwargs) -> str:
    key_dict = {'user_id': user_id, 'top_n': top_n, **kwargs}
    key_str = json.dumps(key_dict, sort_keys=True)
    return hashlib.md5(key_str.encode()).hexdigest()


def get_cached(user_id: int, top_n: int, **kwargs):
    key = get_cache_key(user_id, top_n, **kwargs)
    if key in state.cache:
        if (datetime.now() - state.cache_timestamps[key]).seconds < 3600:
            logger.info(f"✅ Cache hit for user {user_id}")
            return state.cache[key]
        else:
            del state.cache[key]
            del state.cache_timestamps[key]
    return None


def cache_result(user_id: int, top_n: int, result: Dict, **kwargs):
    key = get_cache_key(user_id, top_n, **kwargs)
    state.cache[key] = result
    state.cache_timestamps[key] = datetime.now()
    logger.info(f"💾 Cached for user {user_id}")


# =========================
# Load Models
# =========================
def load_models(force_retrain: bool = False):
    """
    Load all models and data.

    Args:
        force_retrain: If True, ignore any cached content/collaborative
                        model pickles and rebuild from the current
                        content.csv/interactions.csv. Without this, once
                        a model pickle exists on disk it gets reused
                        forever — including across `/reload` calls — even
                        after the underlying CSV/xlsx files change. That
                        previously made `/reload` a no-op for anything
                        except in-memory cache clearing whenever a stale
                        pickle from an earlier run (e.g. smaller demo
                        data) was still sitting in models/.
    """
    global state

    logger.info("🔄 Loading data and models...")
    logger.info(f"📁 Current directory: {os.getcwd()}")

    try:
        base_dir = os.path.dirname(__file__)
        data_dir = os.path.join(base_dir, 'data')

        content_paths = [os.path.join(data_dir, "content.csv"), "data/content.csv", "content.csv"]
        interactions_paths = [os.path.join(data_dir, "interactions.csv"), "data/interactions.csv", "interactions.csv"]
        users_paths = [os.path.join(data_dir, "users.xlsx"), "data/users.xlsx", "users.xlsx"]

        # Load content
        content_path = next((p for p in content_paths if os.path.exists(p)), None)
        if content_path:
            state.content_df = pd.read_csv(content_path)
            logger.info(f"✅ Loaded content from {content_path} ({len(state.content_df)} items)")
        else:
            logger.warning("⚠️ No content file found")
            state.content_df = pd.DataFrame()

        # Load interactions
        interactions_path = next((p for p in interactions_paths if os.path.exists(p)), None)
        if interactions_path:
            state.interactions_df = pd.read_csv(interactions_path)
            logger.info(f"✅ Loaded interactions from {interactions_path} ({len(state.interactions_df)} interactions)")
        else:
            logger.warning("⚠️ No interactions file found")
            state.interactions_df = pd.DataFrame()

        # Load users
        users_path = next((p for p in users_paths if os.path.exists(p)), None)
        if users_path:
            try:
                state.users_df = pd.read_excel(users_path)
                logger.info(f"✅ Loaded users from {users_path} ({len(state.users_df)} users)")
            except Exception as e:
                logger.warning(f"⚠️ Could not load users: {e}")
                state.users_df = pd.DataFrame()
        else:
            state.users_df = pd.DataFrame()

        if state.content_df.empty:
            logger.warning("⚠️ Content data is empty!")
        if state.interactions_df.empty:
            logger.warning("⚠️ Interactions data is empty!")

        # Clean content data
        if not state.content_df.empty:
            try:
                # IMPORTANT: encode_categorical=False here.
                # content_df is used directly downstream for title search
                # (.str.contains), category/level filtering, and as the
                # text source for ContentBased's TF-IDF/embeddings. If
                # DataCleaner label-encodes 'category'/'level'/'title'
                # into integers (its default behavior for any text column
                # with <50 unique values — which easily happens with
                # repetitive course titles/categories), those columns
                # stop being usable strings: title search crashes with
                # "Can only use .str accessor with string values!",
                # category filters silently match nothing, and the
                # content recommender is built on meaningless integer
                # text. Label encoding is only appropriate for a
                # dedicated ML feature matrix (see MLModel.create_features),
                # never for this source-of-truth dataframe.
                cleaner = DataCleaner(encode_categorical=False)
                state.content_df = cleaner.clean(state.content_df)

                # Defensive belt-and-braces: guarantee these columns are
                # always plain strings regardless of what upstream data
                # (or a future DataCleaner change) provides.
                for col in ('title', 'category', 'level', 'description'):
                    if col in state.content_df.columns:
                        state.content_df[col] = (
                            state.content_df[col].fillna('').astype(str)
                        )

                logger.info("✅ Data cleaned")
            except Exception as e:
                logger.warning(f"⚠️ Data cleaning failed: {e}")
                logger.warning(traceback.format_exc())

        # Initialize content recommender
        if not state.content_df.empty:
            try:
                content_model_path = os.path.join(CONTENT_MODEL_DIR, 'content_model.pkl')
                use_cached = (not force_retrain) and os.path.exists(content_model_path)

                if use_cached:
                    logger.info("📂 Loading content recommender from disk...")
                    state.content_recommender = ContentBased(df=state.content_df)
                    state.content_recommender.load(CONTENT_MODEL_DIR)

                    # Staleness guard: a cached model trained on an older
                    # content.csv is worse than useless — it'll look up
                    # content_ids that don't mean the same thing anymore
                    # and silently return wrong categories/titles. Row
                    # COUNT matching alone isn't enough to prove the data
                    # is the same: content.csv can get regenerated with
                    # the exact same row count but completely different
                    # content per content_id (observed in practice — same
                    # 2000 rows, different title/category mapping). Hash
                    # the actual (content_id, title, category) content
                    # instead, so any real change is caught regardless of
                    # whether the row count happens to match.
                    def _content_fingerprint(df: pd.DataFrame) -> str:
                        cols = [c for c in ('content_id', 'title', 'category', 'level') if c in df.columns]
                        sample = df[cols].astype(str).sort_values(by=cols[0])
                        payload = sample.to_csv(index=False).encode('utf-8')
                        return hashlib.md5(payload).hexdigest()

                    cached_count = state.content_recommender.get_item_count()
                    current_count = len(state.content_df)
                    cached_fingerprint = _content_fingerprint(state.content_recommender.df)
                    current_fingerprint = _content_fingerprint(state.content_df)

                    if cached_count != current_count or cached_fingerprint != current_fingerprint:
                        logger.warning(
                            f"⚠️ Cached content model doesn't match current content.csv "
                            f"(rows: {cached_count} vs {current_count}, "
                            f"content changed: {cached_fingerprint != current_fingerprint}) — "
                            f"retraining instead of using the stale cache."
                        )
                        use_cached = False
                    else:
                        logger.info("✅ Content recommender loaded")

                if not use_cached:
                    logger.info("🔄 Creating new content recommender...")
                    state.content_recommender = ContentBased(
                        df=state.content_df,
                        text_cols=['title', 'category', 'level', 'description'],
                        use_embeddings=True
                    )
                    state.content_recommender.save(CONTENT_MODEL_DIR)
                    logger.info("✅ Content recommender created and saved")
            except Exception as e:
                logger.warning(f"⚠️ Content recommender failed: {e}")
                logger.warning(traceback.format_exc())

        # Initialize collaborative recommender
        if not state.interactions_df.empty:
            try:
                collab_model_path = os.path.join(COLLAB_MODEL_DIR, 'collaborative_model.pkl')
                use_cached = (not force_retrain) and os.path.exists(collab_model_path)

                if use_cached:
                    logger.info("📂 Loading collaborative recommender from disk...")
                    state.collaborative_recommender = CollaborativeRecommender()
                    state.collaborative_recommender.load(COLLAB_MODEL_DIR)

                    cached_interactions = state.collaborative_recommender.interactions
                    cached_len = len(cached_interactions) if cached_interactions is not None else 0
                    current_len = len(state.interactions_df)
                    # Allow some drift (new interactions trickling in) but
                    # a large mismatch means this is a different dataset
                    # entirely (e.g. old demo data vs. real export).
                    if cached_len == 0 or abs(cached_len - current_len) / max(current_len, 1) > 0.2:
                        logger.warning(
                            f"⚠️ Cached collaborative model was trained on {cached_len} "
                            f"interactions but interactions.csv now has {current_len} — "
                            f"data changed, retraining instead of using the stale cache."
                        )
                        use_cached = False
                    else:
                        logger.info("✅ Collaborative recommender loaded")

                if not use_cached:
                    logger.info("🔄 Creating new collaborative recommender...")
                    state.collaborative_recommender = CollaborativeRecommender(
                        interactions=state.interactions_df,
                        n_components=50
                    )
                    state.collaborative_recommender.save(COLLAB_MODEL_DIR)
                    logger.info("✅ Collaborative recommender created and saved")
            except Exception as e:
                logger.warning(f"⚠️ Collaborative recommender failed: {e}")
                logger.warning(traceback.format_exc())

        # Create hybrid recommender (actually used in the /recommend/hybrid
        # endpoint now, with the users_df wired in for cold-start handling)
        if state.content_recommender and state.collaborative_recommender:
            try:
                state.hybrid_recommender = HybridRecommender(
                    content_recommender=state.content_recommender,
                    collaborative_recommender=state.collaborative_recommender,
                    users_df=state.users_df,
                    interactions_df=state.interactions_df,
                    content_weight=0.5,
                    collab_weight=0.5,
                    adaptive_weights=True
                )
                logger.info("✅ Hybrid recommender ready")
            except Exception as e:
                logger.warning(f"⚠️ Hybrid recommender failed: {e}")

        state.is_loaded = True
        state.last_update = datetime.now()
        logger.info("✅ All models loaded successfully!")

    except Exception as e:
        logger.error(f"❌ Error loading models: {e}")
        logger.error(traceback.format_exc())
        state.is_loaded = True  # Still allow API to run in fallback mode


# =========================
# Fallback helpers (used only when the hybrid recommender itself
# could not be built — e.g. missing data on a fresh deploy)
# =========================
def _same_category_fallback(
    exclude_ids: set,
    top_n: int,
    filter_category: Optional[str] = None,
    filter_level: Optional[str] = None,
    preferred_category: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Replaces the old `sample()` (pure random) last-resort fallback with a
    ranked one: prefer the user's/seed item's category, then top-rated,
    then simple popularity by interaction count.
    """
    if state.content_df is None or state.content_df.empty:
        return []

    def _norm(series: pd.Series) -> pd.Series:
        return series.astype(str).str.strip().str.lower()

    df = state.content_df.copy()

    if filter_category and 'category' in df.columns:
        df = df[_norm(df['category']) == filter_category.strip().lower()]
    if filter_level and 'level' in df.columns:
        df = df[_norm(df['level']) == filter_level.strip().lower()]

    # Cast both sides to str before isin() — content_id dtype mismatches
    # between content.csv (e.g. Int64) and interactions.csv (e.g. plain
    # int64, or strings if the exporter quoted them) otherwise make isin()
    # silently match nothing, which used to make this fallback return an
    # empty list even though unseen items existed.
    exclude_ids_str = {str(i) for i in exclude_ids}
    df = df[~df['content_id'].astype(str).isin(exclude_ids_str)]

    if df.empty:
        # Exclusion emptied the whole catalog (e.g. a very small demo
        # dataset) — better to show already-seen items again than to
        # return nothing at all.
        df = state.content_df.copy()
        if filter_category and 'category' in df.columns:
            df = df[_norm(df['category']) == filter_category.strip().lower()]
        if filter_level and 'level' in df.columns:
            df = df[_norm(df['level']) == filter_level.strip().lower()]
        if df.empty:
            return []

    def _rank(sub_df: pd.DataFrame) -> pd.DataFrame:
        if 'rating' in sub_df.columns:
            return sub_df.sort_values('rating', ascending=False)
        elif state.interactions_df is not None and not state.interactions_df.empty:
            popularity = state.interactions_df.groupby('content_id').size()
            sub_df = sub_df.assign(_pop=sub_df['content_id'].map(popularity).fillna(0))
            return sub_df.sort_values('_pop', ascending=False)
        return sub_df

    if preferred_category and 'category' in df.columns:
        cat_norm = preferred_category.strip().lower()
        preferred = _rank(df[_norm(df['category']) == cat_norm])
        rest = _rank(df[_norm(df['category']) != cat_norm])
        # Rank *within* each group, then stack — ranking the whole catalog
        # afterwards would undo the preferred-category ordering entirely.
        df = pd.concat([preferred, rest])
    else:
        df = _rank(df)

    recs = []
    for _, row in df.head(top_n).iterrows():
        recs.append({
            'content_id': int(row['content_id']),
            'title': row.get('title', f"Item {row['content_id']}"),
            'category': row.get('category', ''),
            'level': row.get('level', ''),
            'score': float(row.get('rating', 0.4)) / 5.0 if 'rating' in row else 0.4
        })
    return recs


# =========================
# Startup / Shutdown
# =========================
@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Starting API server...")
    try:
        load_models()
        logger.info("✅ API started successfully!")
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("🛑 Shutting down API...")


# =========================
# API Endpoints
# =========================
@app.get("/")
async def root():
    return {
        "message": "Hybrid Recommendation System API",
        "status": "ready" if state.is_loaded else "loading",
        "version": "2.1.0",
        "endpoints": {
            "POST /recommend/hybrid": "Get hybrid recommendations for a user",
            "POST /recommend/content": "Get content-based recommendations by title",
            "GET /recommend/popular": "Get popular items",
            "POST /retrain": "Retrain collaborative model",
            "POST /reload": "Reload all models",
            "GET /health": "Health check",
            "GET /stats": "System statistics",
            "POST /cache/clear": "Clear cache"
        }
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy" if state.is_loaded else "loading",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": state.is_loaded,
        "hybrid_ready": state.hybrid_recommender is not None,
        "cache_size": len(state.cache)
    }


@app.get("/stats")
async def stats():
    if not state.is_loaded:
        raise HTTPException(status_code=503, detail="Service not ready")

    content_recommender_items = (
        state.content_recommender.get_item_count()
        if state.content_recommender is not None else None
    )
    collab_recommender_interactions = (
        len(state.collaborative_recommender.interactions)
        if state.collaborative_recommender is not None and state.collaborative_recommender.interactions is not None
        else None
    )

    return {
        "content_items": len(state.content_df) if state.content_df is not None else 0,
        "interactions": len(state.interactions_df) if state.interactions_df is not None else 0,
        "users": state.interactions_df['user_id'].nunique() if state.interactions_df is not None and 'user_id' in state.interactions_df.columns else 0,
        "categories": state.content_df['category'].nunique() if state.content_df is not None and 'category' in state.content_df.columns else 0,
        # Actual filterable values — use one of these verbatim in
        # ?category=... / ?filter_category=..., not a course title.
        "available_categories": sorted(state.content_df['category'].dropna().unique().tolist())
            if state.content_df is not None and 'category' in state.content_df.columns else [],
        "available_levels": sorted(state.content_df['level'].dropna().unique().tolist())
            if state.content_df is not None and 'level' in state.content_df.columns else [],
        "cache_size": len(state.cache),
        "last_update": state.last_update.isoformat() if state.last_update else None,
        "model_sync": {
            "content_recommender_items": content_recommender_items,
            "content_df_items": len(state.content_df) if state.content_df is not None else 0,
            "content_model_in_sync": (
                content_recommender_items == len(state.content_df)
                if state.content_df is not None and content_recommender_items is not None else None
            ),
            "collab_recommender_interactions": collab_recommender_interactions,
            "interactions_df_rows": len(state.interactions_df) if state.interactions_df is not None else 0,
        }
    }


@app.post("/recommend/hybrid")
async def recommend_hybrid(request: HybridRequest):
    """
    Get hybrid recommendations for a user.

    This now actually delegates to `HybridRecommender.recommend`, which
    combines content-based and collaborative scores (adaptively weighted)
    and handles cold-start users via their profile (interest/level/age),
    instead of the previous Content -> Popular -> Random chain that never
    touched the collaborative model.
    """
    if not state.is_loaded:
        raise HTTPException(status_code=503, detail="Service not ready")

    cached = get_cached(
        request.user_id,
        request.top_n,
        exclude_seen=request.exclude_seen,
        filter_category=request.filter_category,
        filter_level=request.filter_level
    )
    if cached:
        return cached

    try:
        user_history = []
        if state.interactions_df is not None and not state.interactions_df.empty:
            user_history = state.interactions_df[
                state.interactions_df['user_id'] == request.user_id
            ]['content_id'].tolist()

        recommendations = []
        user_type = "existing" if user_history else "cold_start"
        hybrid_debug: Dict[str, Any] = {}

        if state.hybrid_recommender is not None:
            recommendations = state.hybrid_recommender.recommend(
                user_id=request.user_id,
                top_n=request.top_n,
                exclude_seen=request.exclude_seen,
                filter_category=request.filter_category,
                filter_level=request.filter_level
            )
            hybrid_debug = getattr(state.hybrid_recommender, 'last_debug', {})
        else:
            logger.warning("⚠️ Hybrid recommender unavailable, using ranked fallback")
            hybrid_debug['hybrid_recommender'] = 'not_initialized'

        # Ranked fallback (category/rating/popularity — no more random sample())
        if len(recommendations) < request.top_n:
            preferred_category = None
            if user_history and state.content_df is not None:
                last_item = state.content_df[state.content_df['content_id'] == user_history[-1]]
                if not last_item.empty:
                    preferred_category = last_item.iloc[0].get('category')

            existing_ids = {r['content_id'] for r in recommendations} | set(user_history)
            filler = _same_category_fallback(
                exclude_ids=existing_ids,
                top_n=request.top_n - len(recommendations),
                filter_category=request.filter_category,
                filter_level=request.filter_level,
                preferred_category=preferred_category
            )
            recommendations.extend(filler)

        result = {
            "user_id": request.user_id,
            "recommendations": recommendations[:request.top_n],
            "timestamp": datetime.now().isoformat(),
            "user_type": user_type,
            "total_interactions": len(user_history),
            "_debug": hybrid_debug
        }

        cache_result(
            request.user_id,
            request.top_n,
            result,
            exclude_seen=request.exclude_seen,
            filter_category=request.filter_category,
            filter_level=request.filter_level
        )

        return result

    except Exception as e:
        logger.error(f"❌ Error recommending for user {request.user_id}: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/recommend/content")
async def recommend_content(request: ContentRequest):
    """Get content-based recommendations."""
    if not state.is_loaded:
        raise HTTPException(status_code=503, detail="Service not ready")

    try:
        if state.content_recommender is None:
            raise HTTPException(status_code=503, detail="Content recommender not available")

        search_term = request.title.lower().strip()

        # .astype(str) makes this crash-proof regardless of the column's
        # actual dtype (e.g. if it ever ends up numeric/mixed again) —
        # this line was previously the source of
        # "Can only use .str accessor with string values!".
        titles_as_str = state.content_df['title'].astype(str)
        # regex=False: without this, .str.contains() treats search_term as
        # a REGEX pattern. Real course titles often contain regex special
        # characters (|, ., (, ), +, *) — e.g. "Communication Skills -
        # Advanced | Professional Certificate" — and the "|" was being
        # interpreted as regex alternation, silently matching completely
        # unrelated courses that merely contained "professional
        # certificate" anywhere in the title.
        matches = state.content_df[
            titles_as_str.str.lower().str.contains(search_term, na=False, regex=False)
        ]

        recommendations = []
        debug_info: Dict[str, Any] = {}

        if not matches.empty:
            first_match_id = int(matches.iloc[0]['content_id'])

            # Temporary diagnostic wrapper: recommend() has been observed
            # to come back empty for every query with no visible error.
            # Capturing the exception explicitly here (instead of letting
            # the outer handler swallow it into a generic 500, or letting
            # a silent empty return hide it) tells us definitively whether
            # this is throwing internally or genuinely returning [].
            recs = []
            try:
                recs = state.content_recommender.recommend(
                    query=first_match_id,
                    k=request.top_n * 2,
                    return_scores=True,
                    exclude_self=True
                )
                debug_info['content_recommend_call'] = 'ok'
                debug_info['content_recommend_raw_count'] = len(recs) if recs else 0
                debug_info['content_recommender_model_type'] = getattr(state.content_recommender, 'model_type', 'unknown')
                debug_info['content_recommender_df_rows'] = len(getattr(state.content_recommender, 'df', []))
            except Exception as e:
                logger.error(f"❌ content_recommender.recommend() raised: {e}")
                logger.error(traceback.format_exc())
                debug_info['content_recommend_call'] = 'exception'
                debug_info['content_recommend_error'] = f"{type(e).__name__}: {e}"

            if recs:
                for rec in recs:
                    if isinstance(rec, dict):
                        if request.filter_category and not _matches(rec.get('category'), request.filter_category):
                            continue
                        if request.filter_level and not _matches(rec.get('level'), request.filter_level):
                            continue

                        recommendations.append({
                            'content_id': rec.get('content_id'),
                            'title': rec.get('title'),
                            'category': rec.get('category', ''),
                            'level': rec.get('level', ''),
                            'similarity_score': rec.get('similarity_score', 0.0)
                        })
        else:
            # No direct title match: use the content recommender's own
            # semantic text search instead of jumping straight to popular.
            try:
                recs = state.content_recommender.search_by_text(
                    text=search_term,
                    k=request.top_n * 2,
                    return_scores=True,
                    filter_condition={
                        k: v for k, v in {
                            'category': request.filter_category,
                            'level': request.filter_level
                        }.items() if v
                    } or None
                )
                debug_info['search_by_text_call'] = 'ok'
                debug_info['search_by_text_raw_count'] = len(recs) if recs else 0
            except Exception as e:
                logger.error(f"❌ content_recommender.search_by_text() raised: {e}")
                logger.error(traceback.format_exc())
                debug_info['search_by_text_call'] = 'exception'
                debug_info['search_by_text_error'] = f"{type(e).__name__}: {e}"
                recs = []
            for rec in recs:
                recommendations.append({
                    'content_id': rec.get('content_id'),
                    'title': rec.get('title'),
                    'category': rec.get('category', ''),
                    'level': rec.get('level', ''),
                    'similarity_score': rec.get('similarity_score', 0.0)
                })

        # Ranked fallback if still nothing (no random sample())
        if not recommendations:
            logger.info(f"📚 No content/semantic matches for '{request.title}', using ranked fallback")
            filler = _same_category_fallback(
                exclude_ids=set(),
                top_n=request.top_n,
                filter_category=request.filter_category,
                filter_level=request.filter_level
            )
            for item in filler:
                recommendations.append({
                    'content_id': item['content_id'],
                    'title': item['title'],
                    'category': item['category'],
                    'level': item['level'],
                    'similarity_score': item['score']
                })

        return {
            "input": request.title,
            "matched_course": matches.iloc[0]['title'] if not matches.empty else request.title,
            "recommendations": recommendations[:request.top_n],
            "timestamp": datetime.now().isoformat(),
            "total_found": len(recommendations),
            "_debug": debug_info
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error recommending for title '{request.title}': {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/recommend/popular")
async def get_popular(
    top_n: int = Query(10, ge=1, le=50),
    category: Optional[str] = Query(None, description="Filter by category")
):
    """Get most popular items."""
    if not state.is_loaded:
        raise HTTPException(status_code=503, detail="Service not ready")

    try:
        if state.interactions_df is None or state.interactions_df.empty:
            return {"popular_items": [], "timestamp": datetime.now().isoformat()}

        popular = state.interactions_df.groupby('content_id')['rating'].agg(['count', 'mean'])
        popular = popular.sort_values(['count', 'mean'], ascending=False).reset_index()

        if state.content_df is not None and 'content_id' in state.content_df.columns:
            popular = popular.merge(
                state.content_df[['content_id', 'title', 'category', 'level']],
                on='content_id',
                how='left'
            )

        if category and 'category' in popular.columns:
            # Case/whitespace-insensitive match — real course category
            # strings coming from another system are rarely byte-for-byte
            # identical to what's typed in a query param.
            cat_norm = category.strip().lower()
            popular = popular[popular['category'].astype(str).str.strip().str.lower() == cat_norm]

            if popular.empty:
                available = sorted(state.content_df['category'].dropna().unique().tolist()) \
                    if state.content_df is not None and 'category' in state.content_df.columns else []
                return {
                    "popular_items": [],
                    "timestamp": datetime.now().isoformat(),
                    "hint": f"No items found for category '{category}'. "
                            f"'category' expects one of the values below, not a course title.",
                    "available_categories": available
                }

        # Bug fix: category filtering must happen BEFORE truncating to
        # top_n, not after — otherwise a perfectly valid category could
        # come back empty just because none of the globally-top-N items
        # (picked with no category awareness) happened to belong to it.
        result = popular.head(top_n)

        return {
            "popular_items": result.to_dict('records'),
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"❌ Error getting popular items: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/retrain")
async def retrain(request: RetrainRequest, background_tasks: BackgroundTasks):
    """Retrain the collaborative model."""
    global state

    if not state.is_loaded:
        raise HTTPException(status_code=503, detail="Service not ready")

    try:
        if request.interactions_path and os.path.exists(request.interactions_path):
            new_data = pd.read_csv(request.interactions_path)
            updated_interactions = pd.concat([state.interactions_df, new_data])
        else:
            if not request.force_retrain:
                raise HTTPException(
                    status_code=400,
                    detail="No new data provided. Use force_retrain=True to retrain with existing data"
                )
            updated_interactions = state.interactions_df

        if state.collaborative_recommender is None:
            raise HTTPException(status_code=503, detail="Collaborative recommender not available")

        if len(updated_interactions) > 100000:
            background_tasks.add_task(retrain_background, updated_interactions)
            return {"message": "🔄 Retraining started in background", "status": "processing"}

        state.collaborative_recommender.fit(updated_interactions)
        state.interactions_df = updated_interactions
        state.last_update = datetime.now()

        state.collaborative_recommender.save(COLLAB_MODEL_DIR)

        # Rebuild so it picks up the new interactions_df reference —
        # otherwise hybrid_recommender.interactions_df keeps pointing at
        # the pre-retrain DataFrame object even though state.interactions_df
        # has moved on, silently desyncing user-history lookups again.
        if state.content_recommender is not None:
            state.hybrid_recommender = HybridRecommender(
                content_recommender=state.content_recommender,
                collaborative_recommender=state.collaborative_recommender,
                users_df=state.users_df,
                interactions_df=state.interactions_df,
                content_weight=0.5,
                collab_weight=0.5,
                adaptive_weights=True
            )

        state.cache.clear()
        state.cache_timestamps.clear()

        return {
            "message": "✅ Model retrained successfully!",
            "interactions": len(updated_interactions),
            "timestamp": datetime.now().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error retraining: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


async def retrain_background(updated_interactions):
    """Background retraining task."""
    global state
    try:
        logger.info("🔄 Background retraining started...")

        new_collab = CollaborativeRecommender(interactions=updated_interactions, n_components=50)
        new_collab.fit(updated_interactions)
        new_collab.save(COLLAB_MODEL_DIR)

        state.collaborative_recommender = new_collab
        state.interactions_df = updated_interactions
        state.last_update = datetime.now()

        if state.content_recommender is not None:
            state.hybrid_recommender = HybridRecommender(
                content_recommender=state.content_recommender,
                collaborative_recommender=state.collaborative_recommender,
                users_df=state.users_df,
                interactions_df=state.interactions_df,
                content_weight=0.5,
                collab_weight=0.5,
                adaptive_weights=True
            )

        state.cache.clear()
        state.cache_timestamps.clear()

        logger.info("✅ Background retraining completed successfully!")

    except Exception as e:
        logger.error(f"❌ Background retraining failed: {e}")
        logger.error(traceback.format_exc())


@app.post("/reload")
async def reload():
    """
    Reload all models.

    Always retrains from the current content.csv/interactions.csv/
    users.xlsx on disk (force_retrain=True) — this endpoint's whole
    purpose is "pick up new/changed data", so it must not fall back to
    a cached pickle from a previous dataset.
    """
    try:
        state.cache.clear()
        state.cache_timestamps.clear()
        load_models(force_retrain=True)
        return {"message": "✅ Models reloaded and retrained successfully", "timestamp": datetime.now().isoformat()}
    except Exception as e:
        logger.error(f"❌ Error reloading: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/cache/clear")
async def clear_cache():
    """Clear all cached recommendations."""
    state.cache.clear()
    state.cache_timestamps.clear()
    return {"message": "✅ Cache cleared successfully", "timestamp": datetime.now().isoformat()}


# =========================
# Exception Handlers
# =========================
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now().isoformat()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    logger.error(traceback.format_exc())
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": str(exc) if os.getenv("DEBUG", "False").lower() == "true" else "An unexpected error occurred",
            "timestamp": datetime.now().isoformat()
        }
    )


# =========================
# Run
# =========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info"
    )