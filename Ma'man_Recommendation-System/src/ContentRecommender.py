import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Optional, Union, Dict, Any
import pickle
import os

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # sentence-transformers is optional (heavy dependency)
    SentenceTransformer = None


class ContentBased:
    """
    Content-based recommender using TF-IDF or Sentence Transformers.

    Supports multiple text columns, customizable weights, and model persistence.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        text_cols: Optional[List[str]] = None,
        use_embeddings: bool = False,
        embedding_model: str = 'all-MiniLM-L6-v2',
        tfidf_params: Optional[Dict[str, Any]] = None,
        weights: Optional[Dict[str, float]] = None
    ):
        self.df = df.copy()
        self.use_embeddings = use_embeddings and SentenceTransformer is not None
        self.embedding_model_name = embedding_model

        if text_cols is None:
            text_cols = ['title', 'category', 'level', 'description']

        self.text_cols = [col for col in text_cols if col in self.df.columns]

        if weights is None:
            self.weights = {col: 1.0 for col in self.text_cols}
        else:
            self.weights = {col: weights.get(col, 1.0) for col in self.text_cols}

        for col in self.text_cols:
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna("").astype(str)

        self._prepare_text()

        if self.use_embeddings:
            self._init_embedding_model()
        else:
            self._init_tfidf_model(tfidf_params)

        print(f"✅ Content-based model initialized!")
        print(f"   Text columns: {self.text_cols}")
        print(f"   Method: {'Embeddings' if self.use_embeddings else 'TF-IDF'}")
        print(f"   Items: {len(self.df)}")

    def _prepare_text(self):
        """Prepare combined text with column weights."""
        weighted_texts = []

        for _, row in self.df.iterrows():
            parts = []
            for col in self.text_cols:
                weight = self.weights.get(col, 1.0)
                text = str(row[col])
                if weight > 1:
                    text = (text + " ") * int(weight)
                parts.append(text)
            weighted_texts.append(" ".join(parts))

        self.df['combined_text'] = weighted_texts

    def _init_tfidf_model(self, tfidf_params: Optional[Dict[str, Any]] = None):
        n_items = len(self.df)
        default_params = {
            'stop_words': 'english',
            'ngram_range': (1, 2),
            'max_features': 10000,
            # min_df/max_df need to scale down for tiny datasets, otherwise
            # TfidfVectorizer raises "no terms remain" on small catalogs.
            'min_df': 1 if n_items < 20 else 2,
            'max_df': 1.0 if n_items < 20 else 0.9
        }

        if tfidf_params:
            default_params.update(tfidf_params)

        self.vectorizer = TfidfVectorizer(**default_params)
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df['combined_text'])
        self.model_type = 'tfidf'

    def _init_embedding_model(self):
        print(f"🔄 Loading embedding model: {self.embedding_model_name}")
        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        self.embedding_matrix = self.embedding_model.encode(
            self.df['combined_text'].tolist(),
            show_progress_bar=True
        )
        self.model_type = 'embeddings'

    def recommend(
        self,
        query: Union[str, int],
        k: int = 5,
        return_scores: bool = False,
        exclude_self: bool = True,
        filter_condition: Optional[Dict[str, Any]] = None
    ) -> Union[List[str], List[Dict[str, Any]]]:
        """
        Get content-based recommendations.

        Args:
            query: Title (str) or content_id (int)
            k: Number of recommendations
            return_scores: Include similarity scores in results
            exclude_self: Exclude the item itself from recommendations
            filter_condition: Filter items before recommendation

        Returns:
            List of recommended titles or list of dictionaries with details
        """
        # Empty/blank query has no vector representation -> fall back to
        # the item's own popularity ordering instead of returning nothing.
        if isinstance(query, str) and not query.strip():
            return self._popular_fallback(k, return_scores, filter_condition)

        query_vec = self._get_query_vector(query)

        if query_vec is None:
            return self._popular_fallback(k, return_scores, filter_condition)

        if self.model_type == 'tfidf':
            similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        else:
            similarities = cosine_similarity(query_vec, self.embedding_matrix).flatten()

        query_idx = None
        if isinstance(query, int) or (isinstance(query, str) and query.isdigit()):
            q_id = int(query)
            matches = self.df[self.df['content_id'] == q_id]
            query_idx = matches.index[0] if not matches.empty else None

        indices_to_include = self._apply_filters(filter_condition)

        results = []
        for idx, score in enumerate(similarities):
            if indices_to_include is not None and idx not in indices_to_include:
                continue

            if exclude_self and query_idx is not None and idx == query_idx:
                continue

            if exclude_self and isinstance(query, str) and not query.isdigit():
                if self.df.iloc[idx]['title'].lower() == query.lower():
                    continue

            item = {
                'content_id': int(self.df.iloc[idx]['content_id']),
                'title': self.df.iloc[idx]['title'],
                'category': self.df.iloc[idx].get('category', ''),
                'level': self.df.iloc[idx].get('level', ''),
                'similarity_score': float(score)
            }
            results.append(item)

        results.sort(key=lambda x: x['similarity_score'], reverse=True)

        top_results = results[:k]

        if return_scores:
            return top_results
        else:
            return [item['title'] for item in top_results]

    def _popular_fallback(
        self,
        k: int,
        return_scores: bool,
        filter_condition: Optional[Dict[str, Any]] = None
    ) -> Union[List[str], List[Dict[str, Any]]]:
        """Used when there's no usable query vector (e.g. blank query)."""
        df = self.df
        indices_to_include = self._apply_filters(filter_condition)
        if indices_to_include is not None:
            df = df.loc[df.index.isin(indices_to_include)]

        results = []
        for _, row in df.head(k).iterrows():
            results.append({
                'content_id': int(row['content_id']),
                'title': row['title'],
                'category': row.get('category', ''),
                'level': row.get('level', ''),
                'similarity_score': 0.0
            })

        if return_scores:
            return results
        return [item['title'] for item in results]

    def recommend_by_title(
        self,
        title: str,
        k: int = 5,
        return_scores: bool = False,
        filter_category: Optional[str] = None,
        filter_level: Optional[str] = None
    ) -> Union[List[str], List[Dict[str, Any]]]:
        """Get recommendations based on a title with fuzzy matching."""
        return self.search_by_text_fuzzy(
            text=title,
            k=k,
            return_scores=return_scores,
            filter_category=filter_category,
            filter_level=filter_level
        )

    def _get_query_vector(self, query: Union[str, int]):
        if isinstance(query, int) or (isinstance(query, str) and query.isdigit()):
            q_id = int(query)
            idx = self.df[self.df['content_id'] == q_id].index
            if len(idx) == 0:
                return None
            if self.model_type == 'tfidf':
                return self.tfidf_matrix[idx[0]]
            else:
                return self.embedding_matrix[idx[0]].reshape(1, -1)

        query_text = str(query).lower().strip()
        if not query_text:
            return None

        if self.model_type == 'tfidf':
            return self.vectorizer.transform([query_text])
        else:
            return self.embedding_model.encode([query_text])

    def _apply_filters(self, filter_condition: Optional[Dict[str, Any]] = None):
        if filter_condition is None:
            return None

        # Build one combined boolean mask over the FULL (unreduced) df.
        # The previous version re-applied each new mask against `indices`
        # after it had already been shrunk by the prior filter — e.g.
        # filtering by category first (2000 -> 200 rows), then applying a
        # level mask computed against the full 2000-row column onto the
        # now-200-length indices. That mismatch is exactly what produced
        # "boolean index did not match indexed array... size of axis is
        # 200 but size of corresponding boolean axis is 2000".
        combined_mask = pd.Series(True, index=self.df.index)
        for col, value in filter_condition.items():
            if col in self.df.columns:
                if isinstance(value, list):
                    normalized_values = {str(v).strip().lower() for v in value}
                    mask = self.df[col].astype(str).str.strip().str.lower().isin(normalized_values)
                else:
                    mask = self.df[col].astype(str).str.strip().str.lower() == str(value).strip().lower()
                combined_mask &= mask

        return self.df.index[combined_mask].tolist()

    def recommend_similar_to_item(
        self,
        content_id: int,
        k: int = 5,
        return_scores: bool = False
    ) -> Union[List[str], List[Dict[str, Any]]]:
        return self.recommend(query=content_id, k=k, return_scores=return_scores, exclude_self=True)

    def search_by_text(
        self,
        text: str,
        k: int = 5,
        return_scores: bool = False,
        filter_condition: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search items by free-text query against the combined text
        representation (title/category/level/description), using the
        same TF-IDF/embedding space as `recommend`.

        This was previously referenced by `search_by_text_fuzzy` but did
        not exist, causing an AttributeError whenever a title search had
        no direct substring match.

        Args:
            text: Free text query
            k: Number of results
            return_scores: Include similarity scores
            filter_condition: Optional filter dict, e.g. {'category': 'Programming'}

        Returns:
            List of matched items (dicts) sorted by similarity
        """
        text = (text or "").lower().strip()

        if not text:
            return self._popular_fallback(k, True, filter_condition)

        if self.model_type == 'tfidf':
            query_vec = self.vectorizer.transform([text])
            similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        else:
            query_vec = self.embedding_model.encode([text])
            similarities = cosine_similarity(query_vec, self.embedding_matrix).flatten()

        indices_to_include = self._apply_filters(filter_condition)

        results = []
        for idx, score in enumerate(similarities):
            if indices_to_include is not None and idx not in indices_to_include:
                continue
            row = self.df.iloc[idx]
            results.append({
                'content_id': int(row['content_id']),
                'title': row['title'],
                'category': row.get('category', ''),
                'level': row.get('level', ''),
                'similarity_score': float(score)
            })

        results.sort(key=lambda x: x['similarity_score'], reverse=True)
        top_results = results[:k]

        if return_scores:
            return top_results
        return [item['title'] for item in top_results]

    def search_by_text_fuzzy(
        self,
        text: str,
        k: int = 5,
        return_scores: bool = False,
        filter_category: Optional[str] = None,
        filter_level: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Search for items matching text query with fuzzy matching."""
        text = (text or "").lower().strip()
        results = []

        df_filtered = self.df.copy()
        if filter_category:
            df_filtered = df_filtered[df_filtered['category'] == filter_category]
        if filter_level:
            df_filtered = df_filtered[df_filtered['level'] == filter_level]

        if df_filtered.empty:
            return []

        filter_condition = {}
        if filter_category:
            filter_condition['category'] = filter_category
        if filter_level:
            filter_condition['level'] = filter_level

        if text:
            # Method 1: exact/partial match on title
            matches = df_filtered[df_filtered['title'].str.lower().str.contains(text, na=False, regex=False)]

            if not matches.empty:
                for _, row in matches.head(3).iterrows():
                    similar = self.recommend(
                        query=int(row['content_id']),
                        k=k,
                        return_scores=True,
                        exclude_self=True,
                        filter_condition=filter_condition or None
                    )
                    if similar:
                        results.extend(similar)

            # Method 2: no direct title matches -> semantic text search
            if not results:
                results = self.search_by_text(
                    text=text,
                    k=k,
                    return_scores=True,
                    filter_condition=filter_condition or None
                )
        else:
            # No text given at all -> popular fallback within filters
            results = self._popular_fallback(k, True, filter_condition or None)

        # Remove duplicates while preserving order
        seen = set()
        unique_results = []
        for item in results:
            if isinstance(item, dict):
                item_id = item.get('content_id')
            elif isinstance(item, int):
                item_id = item
            else:
                continue

            if item_id not in seen:
                seen.add(item_id)
                unique_results.append(item)

        return unique_results[:k]

    def get_similar_items(self, content_id: int, k: int = 5) -> List[int]:
        results = self.recommend_similar_to_item(content_id, k, return_scores=True)
        return [item['content_id'] for item in results]

    def save(self, path: str):
        """
        Save the model to a directory.

        Args:
            path: Directory path (e.g. "models/content_based") — NOT a
                  filename. The model files are written inside this
                  directory.
        """
        os.makedirs(path, exist_ok=True)

        model_data = {
            'df': self.df,
            'text_cols': self.text_cols,
            'weights': self.weights,
            'use_embeddings': self.use_embeddings,
            'embedding_model_name': self.embedding_model_name,
            'model_type': self.model_type
        }

        if self.model_type == 'tfidf':
            model_data['tfidf_matrix'] = self.tfidf_matrix
            with open(os.path.join(path, 'tfidf_vectorizer.pkl'), 'wb') as f:
                pickle.dump(self.vectorizer, f)
        else:
            model_data['embedding_matrix'] = self.embedding_matrix

        with open(os.path.join(path, 'content_model.pkl'), 'wb') as f:
            pickle.dump(model_data, f)

        print(f"✅ Content-based model saved to {path}")

    def load(self, path: str) -> 'ContentBased':
        """
        Load a model from a directory (must match what `save()` produced).

        Args:
            path: Directory path containing model files.
        """
        if path.endswith('.pkl'):
            # Be forgiving if a caller passes a file path pointing at the
            # pickle itself instead of the containing directory.
            path = os.path.dirname(path) or '.'

        with open(os.path.join(path, 'content_model.pkl'), 'rb') as f:
            model_data = pickle.load(f)

        self.df = model_data['df']
        self.text_cols = model_data['text_cols']
        self.weights = model_data['weights']
        self.use_embeddings = model_data['use_embeddings']
        self.embedding_model_name = model_data['embedding_model_name']
        self.model_type = model_data['model_type']

        if self.model_type == 'tfidf':
            with open(os.path.join(path, 'tfidf_vectorizer.pkl'), 'rb') as f:
                self.vectorizer = pickle.load(f)
            self.tfidf_matrix = model_data.get('tfidf_matrix')
        else:
            self.embedding_matrix = model_data['embedding_matrix']
            if SentenceTransformer is not None:
                self.embedding_model = SentenceTransformer(self.embedding_model_name)

        print(f"✅ Content-based model loaded from {path}")
        return self

    def add_item(self, item: Dict[str, Any]):
        new_row = pd.DataFrame([item])
        self.df = pd.concat([self.df, new_row], ignore_index=True)

        for col in self.text_cols:
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna("").astype(str)

        self._prepare_text()

        if self.model_type == 'tfidf':
            self.tfidf_matrix = self.vectorizer.fit_transform(self.df['combined_text'])
        else:
            self.embedding_matrix = self.embedding_model.encode(
                self.df['combined_text'].tolist(),
                show_progress_bar=True
            )

        print(f"✅ Item added and model updated!")

    def get_item_count(self) -> int:
        return len(self.df)

    def get_item_details(self, content_id: int) -> Optional[Dict[str, Any]]:
        item = self.df[self.df['content_id'] == content_id]
        if len(item) == 0:
            return None
        return item.iloc[0].to_dict()