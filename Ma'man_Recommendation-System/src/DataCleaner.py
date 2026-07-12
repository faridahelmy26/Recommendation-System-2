import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataCleaner:
    """
    Advanced data cleaning and preprocessing for recommendation systems.

    Handles missing values, outliers, encoding, scaling, and more.
    """

    def __init__(
        self,
        handle_missing: str = 'auto',
        handle_outliers: str = 'auto',
        scaling: str = 'none',
        encode_categorical: bool = True,
        verbose: bool = True
    ):
        self.handle_missing = handle_missing
        self.handle_outliers = handle_outliers
        self.scaling = scaling
        self.encode_categorical = encode_categorical
        self.verbose = verbose

        self.scalers = {}
        self.label_encoders = {}
        self.fitted = False

        self.cleaning_report = {}

        logger.info("✅ DataCleaner initialized")

    # =========================
    # Main Cleaning Method
    # =========================

    def clean(
        self,
        df: pd.DataFrame,
        config: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> pd.DataFrame:
        logger.info("🔄 Starting data cleaning...")

        self.cleaning_report = {
            'original_shape': df.shape,
            'original_missing': df.isnull().sum().sum(),
            'operations': []
        }

        df = df.copy()

        df = self._remove_duplicates(df)

        for col in df.columns:
            if config and col in config:
                col_config = config[col]
            else:
                col_config = self._infer_column_config(df[col])

            df[col] = self._clean_column(df[col], col, col_config)

        df = self._remove_incomplete_rows(df)

        if self.encode_categorical:
            df = self._encode_categorical_columns(df)

        if self.scaling != 'none':
            df = self._scale_numeric_columns(df)

        df = df.reset_index(drop=True)

        self.cleaning_report['final_shape'] = df.shape
        self.cleaning_report['final_missing'] = df.isnull().sum().sum()
        self.cleaning_report['rows_removed'] = self.cleaning_report['original_shape'][0] - df.shape[0]

        self.fitted = True

        if self.verbose:
            self._print_report()

        logger.info(f"✅ Data cleaning complete! {df.shape[0]} rows, {df.shape[1]} columns")

        return df

    # =========================
    # Column Cleaning Methods
    # =========================

    def _clean_column(self, series: pd.Series, col_name: str, config: Dict[str, Any]) -> pd.Series:
        if config.get('handle_missing', self.handle_missing) != 'none':
            series = self._handle_missing(series, col_name, config)

        if config.get('type') == 'numeric' and config.get('handle_outliers', self.handle_outliers) != 'none':
            series = self._handle_outliers(series, col_name, config)

        if config.get('type') == 'text':
            series = self._clean_text(series, config)

        if config.get('min') is not None:
            series = series.clip(lower=config['min'])
        if config.get('max') is not None:
            series = series.clip(upper=config['max'])

        return series

    def _infer_column_config(self, series: pd.Series) -> Dict[str, Any]:
        config = {'type': 'unknown'}

        if series.name and ('id' in str(series.name).lower() or series.name == 'index'):
            config['type'] = 'id'
            config['handle_missing'] = 'none'
            return config

        if pd.api.types.is_datetime64_any_dtype(series):
            config['type'] = 'datetime'
            config['handle_missing'] = 'fill'
            return config

        if pd.api.types.is_numeric_dtype(series):
            config['type'] = 'numeric'

            if series.notna().any() and series.min() >= 1 and series.max() <= 5:
                config['min'] = 1
                config['max'] = 5
                config['handle_outliers'] = 'clip'

            return config

        # NOTE: pd.api.types.is_categorical_dtype is deprecated (removed in
        # newer pandas). Use isinstance check against CategoricalDtype instead.
        is_categorical_dtype = isinstance(series.dtype, pd.CategoricalDtype)

        if pd.api.types.is_object_dtype(series) or is_categorical_dtype:
            unique_ratio = series.nunique() / max(len(series), 1)
            if unique_ratio < 0.5 and series.nunique() < 50:
                config['type'] = 'categorical'
                config['fill_value'] = 'unknown'
            else:
                config['type'] = 'text'
                config['fill_value'] = 'unknown'

            return config

        return config

    # =========================
    # Missing Values Handling
    # =========================

    def _handle_missing(self, series: pd.Series, col_name: str, config: Dict[str, Any]) -> pd.Series:
        missing_count = series.isnull().sum()
        if missing_count == 0:
            return series

        strategy = config.get('handle_missing', self.handle_missing)

        if strategy == 'drop':
            return series

        elif strategy == 'fill':
            if config.get('type') == 'numeric':
                fill_value = config.get('fill_value')
                if fill_value is None:
                    skew = series.skew()
                    if pd.notna(skew) and (skew > 1 or skew < -1):
                        fill_value = series.median()
                    else:
                        fill_value = series.mean()
                series = series.fillna(fill_value)

            elif config.get('type') in ['categorical', 'text']:
                fill_value = config.get('fill_value', 'unknown')
                if fill_value == 'mode':
                    mode = series.mode()
                    fill_value = mode[0] if not mode.empty else 'unknown'
                series = series.fillna(fill_value)

            elif config.get('type') == 'datetime':
                mode = series.mode()
                fill_value = config.get('fill_value', mode[0] if not mode.empty else pd.Timestamp.now())
                series = series.fillna(fill_value)

            else:
                series = series.ffill().bfill()

        elif strategy == 'auto':
            return self._handle_missing(series, col_name, {**config, 'handle_missing': 'fill'})

        self.cleaning_report['operations'].append({
            'column': col_name,
            'operation': f'filled {missing_count} missing values'
        })

        return series

    # =========================
    # Outliers Handling
    # =========================

    def _handle_outliers(self, series: pd.Series, col_name: str, config: Dict[str, Any]) -> pd.Series:
        if not pd.api.types.is_numeric_dtype(series):
            return series

        strategy = config.get('handle_outliers', self.handle_outliers)

        if strategy == 'none':
            return series

        method = config.get('outlier_method', 'iqr')

        if method == 'iqr':
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
        else:
            mean = series.mean()
            std = series.std()
            lower_bound = mean - 3 * std
            upper_bound = mean + 3 * std

        lower_bound = config.get('min', lower_bound)
        upper_bound = config.get('max', upper_bound)

        outlier_count = ((series < lower_bound) | (series > upper_bound)).sum()

        if strategy == 'clip':
            series = series.clip(lower=lower_bound, upper=upper_bound)
            operation = f'clipped {outlier_count} outliers'
        elif strategy == 'remove':
            mask = (series >= lower_bound) & (series <= upper_bound)
            series = series[mask]
            operation = f'removed {outlier_count} outliers'
        else:
            return series

        self.cleaning_report['operations'].append({'column': col_name, 'operation': operation})

        return series

    # =========================
    # Text Cleaning
    # =========================

    def _clean_text(self, series: pd.Series, config: Dict[str, Any]) -> pd.Series:
        series = series.astype(str)
        series = series.str.strip()

        if config.get('lowercase', True):
            series = series.str.lower()

        if config.get('remove_special', True):
            series = series.str.replace(r'[^a-zA-Z0-9\s]', '', regex=True)

        if config.get('clean_whitespace', True):
            series = series.str.replace(r'\s+', ' ', regex=True)

        return series

    # =========================
    # Categorical Encoding
    # =========================

    def _encode_categorical_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                continue

            if df[col].nunique() > 50:
                continue

            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
            else:
                try:
                    df[col] = self.label_encoders[col].transform(df[col].astype(str))
                except ValueError:
                    known_classes = self.label_encoders[col].classes_
                    df[col] = df[col].astype(str).apply(
                        lambda x: self.label_encoders[col].transform([x])[0]
                        if x in known_classes else -1
                    )

            self.cleaning_report['operations'].append({
                'column': col,
                'operation': f'encoded {df[col].nunique()} categories'
            })

        return df

    # =========================
    # Scaling
    # =========================

    def _scale_numeric_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        numeric_cols = []
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                if 'id' in col.lower() or col == 'index':
                    continue
                if df[col].nunique() <= 2:
                    continue
                numeric_cols.append(col)

        if not numeric_cols:
            return df

        if self.scaling == 'standard':
            scaler = StandardScaler()
        elif self.scaling == 'minmax':
            scaler = MinMaxScaler()
        else:
            return df

        if not self.fitted:
            df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
            self.scalers['numeric'] = scaler
        else:
            df[numeric_cols] = self.scalers['numeric'].transform(df[numeric_cols])

        self.cleaning_report['operations'].append({
            'columns': numeric_cols,
            'operation': f'scaled using {self.scaling}'
        })

        return df

    # =========================
    # Utility Methods
    # =========================

    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        before = len(df)
        df = df.drop_duplicates()
        after = len(df)

        if before != after:
            self.cleaning_report['operations'].append({
                'operation': f'removed {before - after} duplicate rows'
            })

        return df

    def _remove_incomplete_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        threshold = len(df.columns) * 0.5
        before = len(df)
        df = df.dropna(thresh=threshold)
        after = len(df)

        if before != after:
            self.cleaning_report['operations'].append({
                'operation': f'removed {before - after} incomplete rows'
            })

        return df

    # =========================
    # Report Generation
    # =========================

    def _print_report(self):
        print("\n" + "=" * 60)
        print("📊 DATA CLEANING REPORT")
        print("=" * 60)

        print(f"\n📐 Original shape: {self.cleaning_report['original_shape']}")
        print(f"📐 Final shape: {self.cleaning_report['final_shape']}")
        print(f"🗑️  Rows removed: {self.cleaning_report['rows_removed']}")

        print(f"\n🔍 Missing values:")
        print(f"   Before: {self.cleaning_report['original_missing']}")
        print(f"   After: {self.cleaning_report['final_missing']}")

        print(f"\n⚙️  Operations performed:")
        for i, op in enumerate(self.cleaning_report['operations'], 1):
            if 'column' in op:
                print(f"   {i}. {op['column']}: {op['operation']}")
            else:
                print(f"   {i}. {op['operation']}")

        print("\n" + "=" * 60 + "\n")

    def get_report(self) -> Dict[str, Any]:
        return self.cleaning_report

    # =========================
    # Specific Cleaning Functions
    # =========================

    def clean_ratings(self, ratings: pd.Series, min_rating: int = 1, max_rating: int = 5) -> pd.Series:
        ratings = ratings.copy()
        ratings = pd.to_numeric(ratings, errors='coerce')
        ratings = ratings.clip(min_rating, max_rating)

        if ratings.dtype == float and (ratings % 1).sum() < len(ratings) * 0.1:
            ratings = ratings.round()

        return ratings

    def clean_date(self, dates: pd.Series, format: Optional[str] = None) -> pd.Series:
        dates = dates.copy()

        if format:
            dates = pd.to_datetime(dates, format=format, errors='coerce')
        else:
            dates = pd.to_datetime(dates, errors='coerce')

        return dates

    def clean_categorical(
        self,
        categories: pd.Series,
        allowed_values: Optional[List[Any]] = None,
        unknown_value: str = 'unknown'
    ) -> pd.Series:
        categories = categories.copy()
        categories = categories.astype(str).str.strip()

        if allowed_values:
            categories = categories.where(categories.isin(allowed_values), unknown_value)

        return categories


def clean_data(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    cleaner = DataCleaner(**kwargs)
    return cleaner.clean(df)