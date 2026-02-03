"""
Data loader with lazy loading support.
Supports CSV, XLSX, Parquet, Feather, ZIP (containing CSV), and GZ (compressed CSV).
"""

import os
import gzip
import zipfile
import tempfile
from pathlib import Path
from typing import List, Optional, Dict, Any

import pandas as pd
import numpy as np


# Virtual column names
INDEX_COLUMN = "index"
FILE_INDEX_COLUMN = "file_index"


class DataLoader:
    """
    Lazy data loader that reads column names first, then loads selected columns on demand.
    """

    SUPPORTED_EXTENSIONS = {'.csv', '.xlsx', '.parquet', '.ftr', '.feather', '.zip', '.gz'}

    def __init__(self, file_path: str, file_index: int = 0):
        """
        Initialize the loader with a file path.

        Args:
            file_path: Path to the data file
            file_index: Index of this file (for multi-file scenarios)
        """
        self.file_path = file_path
        self.file_name = os.path.basename(file_path)
        self.file_index = file_index
        self._columns: Optional[List[str]] = None
        self._row_count: Optional[int] = None
        self._data: Optional[pd.DataFrame] = None
        self._file_type = self._detect_file_type()
        self._temp_file: Optional[str] = None

    def _detect_file_type(self) -> str:
        """Detect file type from extension."""
        ext = Path(self.file_path).suffix.lower()
        if ext == '.gz':
            # Check if it's a .csv.gz file
            stem = Path(self.file_path).stem
            if stem.endswith('.csv'):
                return 'csv_gz'
            return 'gz'
        return ext.lstrip('.')

    @classmethod
    def is_supported(cls, file_path: str) -> bool:
        """Check if the file type is supported."""
        ext = Path(file_path).suffix.lower()
        return ext in cls.SUPPORTED_EXTENSIONS

    def get_columns(self, include_virtual: bool = True) -> List[str]:
        """
        Get column names without loading full data (lazy loading).

        Args:
            include_virtual: Whether to include virtual columns (index, file_index)

        Returns:
            List of column names
        """
        if self._columns is None:
            self._load_column_info()

        columns = self._columns.copy()
        if include_virtual:
            # Add virtual columns at the beginning
            columns = [INDEX_COLUMN, FILE_INDEX_COLUMN] + columns

        return columns

    def _load_column_info(self):
        """Load column names from file."""
        file_type = self._file_type

        try:
            if file_type == 'csv':
                df = pd.read_csv(self.file_path, nrows=0)
                self._columns = df.columns.tolist()

            elif file_type == 'csv_gz' or file_type == 'gz':
                with gzip.open(self.file_path, 'rt') as f:
                    df = pd.read_csv(f, nrows=0)
                self._columns = df.columns.tolist()

            elif file_type == 'xlsx':
                df = pd.read_excel(self.file_path, nrows=0, engine='openpyxl')
                self._columns = df.columns.tolist()

            elif file_type in ('parquet',):
                import pyarrow.parquet as pq
                schema = pq.read_schema(self.file_path)
                self._columns = schema.names

            elif file_type in ('ftr', 'feather'):
                import pyarrow.feather as feather
                table = feather.read_table(self.file_path)
                self._columns = table.schema.names

            elif file_type == 'zip':
                self._columns = self._get_columns_from_zip()

            else:
                raise ValueError(f"Unsupported file type: {file_type}")

        except Exception as e:
            raise IOError(f"Failed to read columns from {self.file_path}: {e}")

    def _get_columns_from_zip(self) -> List[str]:
        """Extract CSV from ZIP and get columns."""
        with zipfile.ZipFile(self.file_path, 'r') as zf:
            csv_files = [f for f in zf.namelist() if f.lower().endswith('.csv')]
            if not csv_files:
                raise ValueError("No CSV file found in ZIP archive")

            with zf.open(csv_files[0]) as f:
                df = pd.read_csv(f, nrows=0)
                return df.columns.tolist()

    def load_columns(self, columns: List[str]) -> pd.DataFrame:
        """
        Load only specified columns from the file.

        Args:
            columns: List of column names to load

        Returns:
            DataFrame with selected columns
        """
        # Separate virtual columns from real columns
        need_index = INDEX_COLUMN in columns
        need_file_index = FILE_INDEX_COLUMN in columns
        virtual_cols = {INDEX_COLUMN, FILE_INDEX_COLUMN}
        real_columns = [c for c in columns if c not in virtual_cols]

        # Remove duplicates while preserving order
        real_columns = list(dict.fromkeys(real_columns))

        file_type = self._file_type

        try:
            if not real_columns:
                # Only virtual columns requested, need to get row count
                df = self._load_minimal_for_row_count()
            elif file_type == 'csv':
                df = pd.read_csv(self.file_path, usecols=real_columns)

            elif file_type == 'csv_gz' or file_type == 'gz':
                with gzip.open(self.file_path, 'rt') as f:
                    df = pd.read_csv(f, usecols=real_columns)

            elif file_type == 'xlsx':
                df = pd.read_excel(self.file_path, usecols=real_columns, engine='openpyxl')

            elif file_type in ('parquet',):
                df = pd.read_parquet(self.file_path, columns=real_columns)

            elif file_type in ('ftr', 'feather'):
                import pyarrow.feather as feather
                df = feather.read_feather(self.file_path, columns=real_columns)

            elif file_type == 'zip':
                df = self._load_columns_from_zip(real_columns)

            else:
                raise ValueError(f"Unsupported file type: {file_type}")

            # Add virtual columns if requested
            if need_index:
                df[INDEX_COLUMN] = np.arange(len(df))
            if need_file_index:
                df[FILE_INDEX_COLUMN] = self.file_index

        except Exception as e:
            raise IOError(f"Failed to load columns from {self.file_path}: {e}")

        return df

    def _load_minimal_for_row_count(self) -> pd.DataFrame:
        """Load minimal data just to get row count."""
        file_type = self._file_type

        if file_type == 'csv':
            # Read just first column to get row count
            df = pd.read_csv(self.file_path, usecols=[0])
        elif file_type == 'csv_gz' or file_type == 'gz':
            with gzip.open(self.file_path, 'rt') as f:
                df = pd.read_csv(f, usecols=[0])
        elif file_type == 'xlsx':
            df = pd.read_excel(self.file_path, usecols=[0], engine='openpyxl')
        elif file_type in ('parquet',):
            pf = pd.read_parquet(self.file_path, columns=[self._columns[0]] if self._columns else None)
            df = pf
        elif file_type in ('ftr', 'feather'):
            import pyarrow.feather as feather
            df = feather.read_feather(self.file_path, columns=[self._columns[0]] if self._columns else None)
        elif file_type == 'zip':
            df = self._load_columns_from_zip([self._columns[0]] if self._columns else None)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

        # Return empty dataframe with correct row count
        return pd.DataFrame(index=range(len(df)))

    def _load_columns_from_zip(self, columns: List[str]) -> pd.DataFrame:
        """Load specific columns from CSV inside ZIP."""
        with zipfile.ZipFile(self.file_path, 'r') as zf:
            csv_files = [f for f in zf.namelist() if f.lower().endswith('.csv')]
            if not csv_files:
                raise ValueError("No CSV file found in ZIP archive")

            with zf.open(csv_files[0]) as f:
                if columns:
                    df = pd.read_csv(f, usecols=columns)
                else:
                    df = pd.read_csv(f, usecols=[0])
                return df

    def load_all(self) -> pd.DataFrame:
        """
        Load all data from the file.

        Returns:
            Complete DataFrame
        """
        if self._data is not None:
            return self._data

        columns = self.get_columns(include_virtual=False)
        self._data = self.load_columns(columns)
        return self._data

    def get_statistics(self, columns: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
        """
        Calculate basic statistics for specified columns.

        Args:
            columns: Columns to analyze (None for all)

        Returns:
            Dictionary with statistics for each column
        """
        if columns is None:
            columns = self.get_columns(include_virtual=False)

        # Filter out virtual columns for statistics
        virtual_cols = {INDEX_COLUMN, FILE_INDEX_COLUMN}
        real_columns = [c for c in columns if c not in virtual_cols]

        df = self.load_columns(real_columns)
        stats = {}

        for col in real_columns:
            col_data = df[col]
            if pd.api.types.is_numeric_dtype(col_data):
                stats[col] = {
                    'count': int(col_data.count()),
                    'mean': float(col_data.mean()) if col_data.count() > 0 else None,
                    'std': float(col_data.std()) if col_data.count() > 1 else None,
                    'min': float(col_data.min()) if col_data.count() > 0 else None,
                    'max': float(col_data.max()) if col_data.count() > 0 else None,
                }
            else:
                stats[col] = {
                    'count': int(col_data.count()),
                    'unique': int(col_data.nunique()),
                    'dtype': str(col_data.dtype),
                }

        return stats

    def __del__(self):
        """Cleanup temporary files if any."""
        if self._temp_file and os.path.exists(self._temp_file):
            try:
                os.unlink(self._temp_file)
            except Exception:
                pass


class MultiFileLoader:
    """Manager for multiple data files."""

    def __init__(self):
        self._loaders: Dict[str, DataLoader] = {}
        self._file_index_counter = 0

    def add_file(self, file_path: str) -> str:
        """
        Add a file to the manager.

        Args:
            file_path: Path to the file

        Returns:
            Unique key for the file
        """
        if not DataLoader.is_supported(file_path):
            raise ValueError(f"Unsupported file type: {file_path}")

        # Use absolute path as key
        key = os.path.abspath(file_path)
        if key not in self._loaders:
            self._loaders[key] = DataLoader(file_path, file_index=self._file_index_counter)
            self._file_index_counter += 1

        return key

    def remove_file(self, key: str):
        """Remove a file from the manager."""
        # Normalize key to absolute path (same as add_file)
        abs_key = os.path.abspath(key)
        if abs_key in self._loaders:
            del self._loaders[abs_key]
        elif key in self._loaders:
            del self._loaders[key]

    def get_loader(self, key: str) -> Optional[DataLoader]:
        """Get a loader by its key."""
        # Try absolute path first
        abs_key = os.path.abspath(key)
        if abs_key in self._loaders:
            return self._loaders[abs_key]
        return self._loaders.get(key)

    def get_all_loaders(self) -> Dict[str, DataLoader]:
        """Get all loaders."""
        return self._loaders.copy()

    def get_file_names(self) -> Dict[str, str]:
        """Get mapping of keys to file names."""
        return {k: v.file_name for k, v in self._loaders.items()}

    def clear(self):
        """Remove all files."""
        self._loaders.clear()
        self._file_index_counter = 0

    def get_all_columns(self) -> List[str]:
        """
        Get union of all columns from all loaded files.

        Returns:
            List of unique column names (with virtual columns)
        """
        all_columns = set()
        for loader in self._loaders.values():
            columns = loader.get_columns(include_virtual=False)
            all_columns.update(columns)

        sorted_columns = sorted(all_columns)
        return [INDEX_COLUMN, FILE_INDEX_COLUMN] + sorted_columns

    def get_columns_per_file(self) -> Dict[str, List[str]]:
        """
        Get columns available in each file.

        Returns:
            Dict mapping file key to list of columns
        """
        return {
            key: loader.get_columns(include_virtual=False)
            for key, loader in self._loaders.items()
        }

    def load_columns_merged(self, columns: List[str]) -> pd.DataFrame:
        """
        Load specified columns from all files and merge them.

        Files that don't have certain columns will have NaN for those columns.

        Args:
            columns: List of column names to load

        Returns:
            Merged DataFrame from all files
        """
        if not self._loaders:
            return pd.DataFrame()

        # Separate virtual columns from real columns
        virtual_cols = {INDEX_COLUMN, FILE_INDEX_COLUMN}
        need_index = INDEX_COLUMN in columns
        need_file_index = FILE_INDEX_COLUMN in columns
        real_columns = [c for c in columns if c not in virtual_cols]

        dfs = []
        for key, loader in self._loaders.items():
            # Get columns that this file actually has
            file_columns = set(loader.get_columns(include_virtual=False))
            available_columns = [c for c in real_columns if c in file_columns]

            if not available_columns and not need_index and not need_file_index:
                # No columns available from this file
                continue

            try:
                # Load available columns
                cols_to_load = available_columns.copy()
                if need_index:
                    cols_to_load.append(INDEX_COLUMN)
                if need_file_index:
                    cols_to_load.append(FILE_INDEX_COLUMN)

                df = loader.load_columns(cols_to_load)

                # Add missing columns as NaN
                for col in real_columns:
                    if col not in df.columns:
                        df[col] = np.nan

                # Reorder columns to match requested order
                ordered_cols = [c for c in columns if c in df.columns]
                df = df[ordered_cols]

                dfs.append(df)

            except Exception as e:
                print(f"Warning: Failed to load from {loader.file_name}: {e}")
                continue

        if not dfs:
            return pd.DataFrame()

        # Concatenate all dataframes
        result = pd.concat(dfs, ignore_index=True)

        # Regenerate index column after concatenation
        if need_index:
            result[INDEX_COLUMN] = np.arange(len(result))

        return result
