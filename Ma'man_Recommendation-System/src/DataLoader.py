import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Union, Dict, Any, List
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataLoader:
    """
    Advanced data loader for recommendation system datasets.

    Supports loading from CSV, Excel, JSON, and Parquet formats.
    Includes data validation, caching, and error handling.
    """

    def __init__(
        self,
        data_dir: Optional[Union[str, Path]] = "data",
        use_cache: bool = True,
        validate_columns: bool = True
    ):
        self.data_dir = Path(data_dir)
        self.use_cache = use_cache
        self.validate_columns = validate_columns
        self._cache = {}

        self.data_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"📂 DataLoader initialized with directory: {self.data_dir}")

    # =========================
    # Main Loading Methods
    # =========================

    def load_content(
        self,
        path: Optional[Union[str, Path]] = None,
        required_cols: Optional[List[str]] = None,
        **kwargs
    ) -> pd.DataFrame:
        if path is None:
            path = self.data_dir / "content.csv"
        else:
            path = Path(path)

        default_required = ['content_id', 'title', 'category', 'level']
        required_cols = required_cols or default_required

        return self._load_file(path, required_cols=required_cols, file_type='content', **kwargs)

    def load_interactions(
        self,
        path: Optional[Union[str, Path]] = None,
        required_cols: Optional[List[str]] = None,
        **kwargs
    ) -> pd.DataFrame:
        if path is None:
            path = self.data_dir / "interactions.csv"
        else:
            path = Path(path)

        default_required = ['user_id', 'content_id', 'rating']
        required_cols = required_cols or default_required

        return self._load_file(path, required_cols=required_cols, file_type='interactions', **kwargs)

    def load_users(
        self,
        path: Optional[Union[str, Path]] = None,
        required_cols: Optional[List[str]] = None,
        **kwargs
    ) -> pd.DataFrame:
        if path is None:
            possible_paths = [
                self.data_dir / "users.xlsx",
                self.data_dir / "users.csv",
                self.data_dir / "users.json"
            ]
            path = None
            for p in possible_paths:
                if p.exists():
                    path = p
                    break

            if path is None:
                raise FileNotFoundError(f"No users file found in {self.data_dir}")
        else:
            path = Path(path)

        default_required = ['user_id']
        required_cols = required_cols or default_required

        return self._load_file(path, required_cols=required_cols, file_type='users', **kwargs)

    def load_all(
        self,
        content_path: Optional[Union[str, Path]] = None,
        interactions_path: Optional[Union[str, Path]] = None,
        users_path: Optional[Union[str, Path]] = None
    ) -> Dict[str, pd.DataFrame]:
        logger.info("🔄 Loading all datasets...")

        data = {
            'content': self.load_content(content_path),
            'interactions': self.load_interactions(interactions_path),
        }

        try:
            data['users'] = self.load_users(users_path)
        except FileNotFoundError:
            logger.warning("⚠️ Users file not found, proceeding without it")
            data['users'] = pd.DataFrame()

        logger.info(
            f"✅ Loaded {len(data['content'])} content items, "
            f"{len(data['interactions'])} interactions, "
            f"{len(data['users'])} users"
        )

        return data

    # =========================
    # Generic File Loader
    # =========================

    def _load_file(
        self,
        path: Path,
        required_cols: Optional[List[str]] = None,
        file_type: str = 'data',
        **kwargs
    ) -> pd.DataFrame:
        cache_key = f"{file_type}_{str(path)}_{str(kwargs)}"
        if self.use_cache and cache_key in self._cache:
            logger.info(f"📦 Loading {file_type} from cache")
            return self._cache[cache_key].copy()

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        logger.info(f"📂 Loading {file_type} from: {path}")

        try:
            df = self._load_by_extension(path, **kwargs)
        except Exception as e:
            logger.error(f"❌ Error loading {path}: {e}")
            raise

        if len(df) == 0:
            logger.warning(f"⚠️ {file_type} file is empty: {path}")

        if self.validate_columns and required_cols:
            self._validate_columns(df, required_cols, file_type)

        df = self._clean_dataframe(df, file_type)

        if self.use_cache:
            self._cache[cache_key] = df.copy()

        logger.info(f"✅ Loaded {len(df)} rows from {file_type}")

        return df

    def _load_by_extension(self, path: Path, **kwargs) -> pd.DataFrame:
        extension = path.suffix.lower()

        if extension == '.csv':
            return self._read_csv_robust(path, **kwargs)
        elif extension in ['.xlsx', '.xls']:
            return pd.read_excel(path, **kwargs)
        elif extension == '.json':
            return pd.read_json(path, **kwargs)
        elif extension == '.parquet':
            return pd.read_parquet(path, **kwargs)
        else:
            logger.warning(f"Unknown extension {extension}, trying CSV...")
            return self._read_csv_robust(path, **kwargs)

    def _read_csv_robust(self, path: Path, **kwargs) -> pd.DataFrame:
        """
        Read a CSV with pandas' fast C engine first; if that fails —
        typically from real-world exports with unescaped quotes, stray
        delimiters inside a field, or embedded newlines (a course title
        containing a comma or a bare `"` is enough) — retry with the
        slower but far more forgiving Python engine, skipping rows it
        can't parse rather than failing the whole load.

        The C engine's error for this class of problem is a misleading
        "ParserError: ... out of memory" — it isn't actually about RAM,
        it's the tokenizer losing track of field/quote boundaries on a
        malformed row and trying to read a huge chunk as a single field.
        """
        try:
            return pd.read_csv(path, **kwargs)
        except pd.errors.ParserError as e:
            logger.warning(
                f"⚠️ C parser failed on {path} ({e}). This usually means the "
                f"file has an unescaped quote, stray delimiter inside a "
                f"field, or an embedded newline — not real memory pressure. "
                f"Retrying with the Python engine, skipping bad rows..."
            )
            fallback_kwargs = dict(kwargs)
            fallback_kwargs.pop('low_memory', None)  # not supported by the python engine
            try:
                df = pd.read_csv(
                    path,
                    engine='python',
                    on_bad_lines='warn',
                    **fallback_kwargs
                )
                logger.warning(
                    f"✅ Recovered {len(df)} rows from {path} using the Python "
                    f"engine. Check the source export for rows with unescaped "
                    f"quotes/commas — those were skipped, not fixed."
                )
                return df
            except Exception as e2:
                logger.error(f"❌ Python-engine fallback also failed on {path}: {e2}")
                raise

    # =========================
    # Validation Methods
    # =========================

    def _validate_columns(self, df: pd.DataFrame, required_cols: List[str], file_type: str):
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            raise ValueError(
                f"❌ {file_type} file missing required columns: {missing_cols}\n"
                f"Available columns: {df.columns.tolist()}"
            )

        for col in required_cols:
            if df[col].isnull().any():
                null_count = df[col].isnull().sum()
                logger.warning(f"⚠️ {file_type}: {col} has {null_count} null values")

    def _clean_dataframe(self, df: pd.DataFrame, file_type: str) -> pd.DataFrame:
        df = df.copy()

        df = df.drop_duplicates()

        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].astype(str).str.strip()

        id_cols = [col for col in df.columns if 'id' in col.lower()]
        for col in id_cols:
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
            except Exception:
                pass

        if 'rating' in df.columns:
            try:
                df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
                if df['rating'].notna().any() and df['rating'].min() >= 1 and df['rating'].max() <= 5:
                    df['rating'] = df['rating'].clip(1, 5)
            except Exception:
                pass

        return df

    # =========================
    # Utility Methods
    # =========================

    def get_data_info(self) -> Dict[str, Any]:
        info = {}

        for key, df in self._cache.items():
            if isinstance(df, pd.DataFrame):
                info[key] = {
                    'rows': len(df),
                    'columns': df.columns.tolist(),
                    'memory_usage': df.memory_usage(deep=True).sum() / 1024 ** 2,
                    'null_count': df.isnull().sum().sum()
                }

        return info

    def clear_cache(self):
        self._cache.clear()
        logger.info("🗑️ Cache cleared")

    def save_dataframe(
        self,
        df: pd.DataFrame,
        path: Union[str, Path],
        format: Optional[str] = None,
        **kwargs
    ):
        path = Path(path)

        if format is None:
            format = path.suffix.lower().replace('.', '')

        if format == 'csv':
            df.to_csv(path, index=False, **kwargs)
        elif format in ['xlsx', 'xls']:
            df.to_excel(path, index=False, **kwargs)
        elif format == 'json':
            df.to_json(path, orient='records', **kwargs)
        elif format == 'parquet':
            df.to_parquet(path, index=False, **kwargs)
        else:
            raise ValueError(f"Unsupported format: {format}")

        logger.info(f"✅ Saved DataFrame to {path}")

    def get_stats(self, df: pd.DataFrame, name: str = 'DataFrame') -> Dict[str, Any]:
        stats = {
            'rows': len(df),
            'columns': len(df.columns),
            'column_names': df.columns.tolist(),
            'null_count': df.isnull().sum().sum(),
            'duplicate_count': df.duplicated().sum(),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 ** 2,
            'dtypes': df.dtypes.to_dict()
        }

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            stats['numeric_stats'] = df[numeric_cols].describe().to_dict()

        categorical_cols = df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            stats['categorical_stats'] = {
                col: df[col].value_counts().head(5).to_dict()
                for col in categorical_cols[:3]
            }

        return stats


# =========================
# Helper Functions (Backward Compatible)
# =========================

def load_content(path: Union[str, Path] = "data/content.csv", **kwargs) -> pd.DataFrame:
    path = Path(path)
    loader = DataLoader(data_dir=path.parent)
    return loader.load_content(path, **kwargs)


def load_interactions(path: Union[str, Path] = "data/interactions.csv", **kwargs) -> pd.DataFrame:
    path = Path(path)
    loader = DataLoader(data_dir=path.parent)
    return loader.load_interactions(path, **kwargs)