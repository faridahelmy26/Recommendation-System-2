import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any, Union, Tuple
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, f1_score
import pickle
import os
import warnings
warnings.filterwarnings('ignore')
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MLModel:
    """
    Advanced ML model for ranking and re-ranking recommendations.

    Supports multiple algorithms with feature engineering and evaluation.
    """

    def __init__(
        self,
        model_type: str = 'random_forest',
        task: str = 'regression',
        hyperparameters: Optional[Dict[str, Any]] = None,
        random_state: int = 42
    ):
        self.model_type = model_type
        self.task = task
        self.random_state = random_state
        self.hyperparameters = hyperparameters or {}

        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = None
        self.is_fitted = False

        self._init_model()

        logger.info(f"✅ MLModel initialized: {model_type} ({task})")

    def _init_model(self):
        if self.model_type == 'random_forest':
            if self.task == 'regression':
                from sklearn.ensemble import RandomForestRegressor
                self.model = RandomForestRegressor(
                    n_estimators=self.hyperparameters.get('n_estimators', 100),
                    max_depth=self.hyperparameters.get('max_depth', None),
                    min_samples_split=self.hyperparameters.get('min_samples_split', 2),
                    min_samples_leaf=self.hyperparameters.get('min_samples_leaf', 1),
                    random_state=self.random_state,
                    n_jobs=-1
                )
            else:
                from sklearn.ensemble import RandomForestClassifier
                self.model = RandomForestClassifier(
                    n_estimators=self.hyperparameters.get('n_estimators', 100),
                    max_depth=self.hyperparameters.get('max_depth', None),
                    min_samples_split=self.hyperparameters.get('min_samples_split', 2),
                    min_samples_leaf=self.hyperparameters.get('min_samples_leaf', 1),
                    random_state=self.random_state,
                    n_jobs=-1
                )

        elif self.model_type == 'xgboost':
            try:
                import xgboost as xgb
                cls = xgb.XGBRegressor if self.task == 'regression' else xgb.XGBClassifier
                self.model = cls(
                    n_estimators=self.hyperparameters.get('n_estimators', 100),
                    max_depth=self.hyperparameters.get('max_depth', 6),
                    learning_rate=self.hyperparameters.get('learning_rate', 0.1),
                    random_state=self.random_state,
                    n_jobs=-1
                )
            except ImportError:
                logger.warning("⚠️ XGBoost not installed, falling back to Random Forest")
                self.model_type = 'random_forest'
                self._init_model()

        elif self.model_type == 'lightgbm':
            try:
                import lightgbm as lgb
                cls = lgb.LGBMRegressor if self.task == 'regression' else lgb.LGBMClassifier
                self.model = cls(
                    n_estimators=self.hyperparameters.get('n_estimators', 100),
                    max_depth=self.hyperparameters.get('max_depth', -1),
                    learning_rate=self.hyperparameters.get('learning_rate', 0.1),
                    random_state=self.random_state,
                    n_jobs=-1
                )
            except ImportError:
                logger.warning("⚠️ LightGBM not installed, falling back to Random Forest")
                self.model_type = 'random_forest'
                self._init_model()

        elif self.model_type == 'linear':
            if self.task == 'regression':
                from sklearn.linear_model import Ridge
                self.model = Ridge(alpha=self.hyperparameters.get('alpha', 1.0), random_state=self.random_state)
            else:
                from sklearn.linear_model import LogisticRegression
                self.model = LogisticRegression(
                    C=self.hyperparameters.get('C', 1.0),
                    random_state=self.random_state,
                    max_iter=1000
                )
        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}")

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        categorical_features: Optional[List[str]] = None,
        eval_set: Optional[Tuple] = None
    ) -> 'MLModel':
        logger.info("🔄 Training ML model...")

        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)

        self.feature_columns = X.columns.tolist() if hasattr(X, 'columns') else list(range(X.shape[1]))

        if categorical_features:
            X = self._encode_categorical(X, categorical_features)

        X_scaled = self.scaler.fit_transform(X)

        self.model.fit(X_scaled, y)
        self.is_fitted = True

        logger.info(f"✅ Model trained successfully!")
        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("❌ Model not fitted. Call fit() first.")

        return self.model.predict(self._preprocess(X))

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        if self.task != 'classification':
            raise ValueError("predict_proba only available for classification")

        if not self.is_fitted:
            raise RuntimeError("Model not fitted")

        X_processed = self._preprocess(X)
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X_processed)
        return self.model.predict(X_processed)

    def _preprocess(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)

        if hasattr(X, 'columns') and self.feature_columns:
            for col in self.feature_columns:
                if col not in X.columns:
                    X[col] = 0
            X = X[self.feature_columns]

        X = self._encode_categorical(X, list(self.label_encoders.keys()))

        return self.scaler.transform(X)

    def _encode_categorical(self, X: pd.DataFrame, categorical_features: List[str]) -> pd.DataFrame:
        X = X.copy()

        for col in categorical_features:
            if col in X.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    X[col] = self.label_encoders[col].fit_transform(X[col].astype(str))
                else:
                    try:
                        X[col] = self.label_encoders[col].transform(X[col].astype(str))
                    except ValueError:
                        known_classes = self.label_encoders[col].classes_
                        X[col] = X[col].astype(str).apply(
                            lambda x: self.label_encoders[col].transform([x])[0]
                            if x in known_classes else -1
                        )

        return X

    def evaluate(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> Dict[str, float]:
        y_pred = self.predict(X)

        metrics = {}

        if self.task == 'regression':
            metrics['mse'] = mean_squared_error(y, y_pred)
            metrics['rmse'] = np.sqrt(metrics['mse'])
            metrics['mae'] = mean_absolute_error(y, y_pred)
            var = np.var(y)
            metrics['r2'] = 1 - (metrics['mse'] / var) if var > 0 else 0.0
        else:
            metrics['accuracy'] = accuracy_score(y, y_pred)
            metrics['f1_macro'] = f1_score(y, y_pred, average='macro')
            metrics['f1_weighted'] = f1_score(y, y_pred, average='weighted')

        logger.info(f"📊 Evaluation metrics: {metrics}")
        return metrics

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)

        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_columns': self.feature_columns,
            'model_type': self.model_type,
            'task': self.task,
            'random_state': self.random_state,
            'hyperparameters': self.hyperparameters,
            'is_fitted': self.is_fitted
        }

        with open(os.path.join(path, 'ml_model.pkl'), 'wb') as f:
            pickle.dump(model_data, f)

        logger.info(f"✅ Model saved to {path}")

    def load(self, path: str) -> 'MLModel':
        if path.endswith('.pkl'):
            path = os.path.dirname(path) or '.'

        with open(os.path.join(path, 'ml_model.pkl'), 'rb') as f:
            model_data = pickle.load(f)

        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.label_encoders = model_data['label_encoders']
        self.feature_columns = model_data['feature_columns']
        self.model_type = model_data['model_type']
        self.task = model_data['task']
        self.random_state = model_data['random_state']
        self.hyperparameters = model_data['hyperparameters']
        self.is_fitted = model_data['is_fitted']

        logger.info(f"✅ Model loaded from {path}")
        return self

    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        if not self.is_fitted:
            return None

        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            importance = np.abs(self.model.coef_).flatten()
        else:
            return None

        if self.feature_columns:
            return pd.DataFrame({
                'feature': self.feature_columns,
                'importance': importance
            }).sort_values('importance', ascending=False)

        return None

    def create_features(
        self,
        interactions: pd.DataFrame,
        content_df: pd.DataFrame,
        users_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        logger.info("🔧 Creating features...")

        features = interactions.merge(content_df, on='content_id')

        if users_df is not None and 'user_id' in users_df.columns:
            features = features.merge(users_df, on='user_id')

        content_popularity = interactions.groupby('content_id')['rating'].agg(['count', 'mean'])
        content_popularity.columns = ['interaction_count', 'avg_rating']
        features = features.merge(content_popularity, on='content_id', how='left')

        user_activity = interactions.groupby('user_id')['rating'].agg(['count', 'mean'])
        user_activity.columns = ['user_interaction_count', 'user_avg_rating']
        features = features.merge(user_activity, on='user_id', how='left')

        if 'category' in features.columns:
            category_popularity = interactions.merge(content_df[['content_id', 'category']], on='content_id')
            category_popularity = category_popularity.groupby('category')['rating'].mean()
            features['category_avg_rating'] = features['category'].map(category_popularity)

        if 'timestamp' in features.columns:
            features['timestamp'] = pd.to_datetime(features['timestamp'])
            features['day_of_week'] = features['timestamp'].dt.dayofweek
            features['hour'] = features['timestamp'].dt.hour
            features['month'] = features['timestamp'].dt.month

        features['user_content_interaction'] = features.groupby(['user_id', 'content_id']).cumcount() + 1

        features = features.fillna(0)

        logger.info(f"✅ Created {features.shape[1]} features for {len(features)} samples")

        return features