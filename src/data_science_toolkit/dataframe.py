from .lib import Lib # from .lib import Lib in production
from .vectorizer import Vectorizer # from .vectorizer import Vectorizer in production
from .chart import Chart # from .chart import Chart in production
from datetime import timedelta
from logging import Logger
from math import ceil
import pandas as pd
from pyparsing import col
import scipy.sparse
from sklearn.decomposition import PCA
from sklearn.feature_extraction import DictVectorizer
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from wordcloud import WordCloud, STOPWORDS
from matplotlib import pyplot as plt
from sklearn.preprocessing import minmax_scale
from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_iris
from collections import Counter
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from datetime import datetime
import os
from datetime import timedelta
import math
from numpy import exp, power, sqrt
import requests
import datetime as dt
import copy
import psutil
import pyarrow as pa
import pyarrow.dataset as ds
from tqdm import tqdm
import pyarrow.parquet as pq
from typing import Callable

class DataFrame:
    """
    """
    vectorizer = None
    __generator = None

    def __init__(self, data_path=None,
                 data_type='csv',
                 columns_names_as_list=None,
                 data_types_in_order=None,
                 delimiter=',',
                 has_header=True,
                 line_index=None,
                 skip_empty_line=False,
                 sheet_name=0,
                 skip_rows=None,
                 n_workers='auto',
                 **kwargs
                 ):
        
        
        self.data_type = data_type
        self.n_workers = n_workers
        self.data_path = data_path

        if data_path is not None:
            if data_type == 'csv':
                if has_header is True:
                    self.dataframe = pd.read_csv(data_path, encoding='utf-8', delimiter=delimiter, 
                                               low_memory=False, on_bad_lines='skip', skip_blank_lines=False,
                                               skiprows=skip_rows, **kwargs)
                else:
                    self.dataframe = pd.read_csv(data_path, encoding='utf-8', delimiter=delimiter, 
                                               low_memory=False, on_bad_lines='skip', skip_blank_lines=False,
                                               header=None, **kwargs)
            elif data_type == 'json':
                self.dataframe = pd.read_json(data_path, encoding='utf-8')
            elif data_type == 'xls':
                self.dataframe = pd.read_excel(data_path, sheet_name=sheet_name,
                                                 skiprows=skip_rows, **kwargs)
            elif data_type == 'pkl':
                self.dataframe = pd.read_pickle(data_path, **kwargs)
            elif data_type == 'dict':
                self.dataframe = pd.DataFrame.from_dict(data_path, **kwargs)
            elif data_type == 'matrix':
                index_name = [i for i in range(len(data_path))]
                colums_name = [i for i in range(len(data_path[0]))]
                self.dataframe = pd.DataFrame(data=data_path, index=index_name, columns=colums_name, **kwargs)
            elif data_type == 'list':
                y = data_path
                if (not isinstance(y, pd.core.series.Series or not isinstance(y, pd.core.frame.DataFrame))):
                    y = np.array(y)
                    y = np.reshape(y, (y.shape[0],))
                    y = pd.Series(y)
                self.dataframe = pd.DataFrame()
                if columns_names_as_list is not None:
                    self.dataframe[columns_names_as_list[0]] = y
                else:
                    self.dataframe['0'] = y
                    
                
                """data = array([['','Col1','Col2'],['Row1',1,2],['Row2',3,4]])
                pd.DataFrame(data=data[1:,1:], index=data[1:,0], columns=data[0,1:]) """
            elif data_type == 'dataframe':
                self.dataframe = data_path
            elif data_type == 'parquet':
                # Set Arrow CPU count for parallel processing
                if self.n_workers == 'auto':
                    n_workers, threads_per_worker = DataFrame.get_optimal_dask_config()['n_workers'], DataFrame.get_optimal_dask_config()['threads_per_worker']
                else:
                    n_workers = self.n_workers
                    threads_per_worker = 1
                    
                pa.set_cpu_count(n_workers * threads_per_worker)
                
                print(f"🧵 PyArrow thread pool configured with {pa.cpu_count()} workers")
                self.dataframe = pd.DataFrame()
                self.dataset = ds.dataset(self.data_path, format="parquet")
                #self.dataframe = pd.read_parquet(data_path, **kwargs)
                
            types = {}
            if data_types_in_order is not None and columns_names_as_list is not None:
                self.dataframe.columns = columns_names_as_list
                for i in range(len(columns_names_as_list)):
                    types[columns_names_as_list[i]] = data_types_in_order[i]
            elif columns_names_as_list is not None:
                self.dataframe.columns = columns_names_as_list
                for p in columns_names_as_list:
                    types[p] = str

            self.dataframe = self.get_dataframe().astype(types)

            if line_index is not None:
                self.dataframe.index = line_index
        else:
            self.dataframe = pd.DataFrame()
        
    def get_generator(self):
        return self.__generator
    
    def remove_stopwords(self, column, language_or_stopwords_list='english', in_place=True):
        if isinstance(language_or_stopwords_list, list) is True:
            stopwords = language_or_stopwords_list
        elif language_or_stopwords_list == 'arabic':
            stopwords = Lib.read_text_file_as_list('data/arabic_stopwords.csv')
        else:
            nltk.download('stopwords')
            stopwords = nltk.corpus.stopwords.words(language_or_stopwords_list)
        self.transform_column(column, DataFrame.remove_stopwords_lambda, in_place, stopwords)
        return self.dataframe
    
    @staticmethod
    def remove_stopwords_lambda(document, stopwords_list):
        document = str.lower(document)
        stopwords = stopwords_list
        words = word_tokenize(document)
        clean_words = []
        for w in words:
            if w not in stopwords:
                clean_words.append(w)
        return ' '.join(clean_words)
    
    def add_random_series_column(self, column_name='random',min=0, max=100, distrubution_type='random', mean=0, sd=1):
        if distrubution_type == 'random':
            series = pd.Series(np.random.randint(min, max, self.get_shape()[0]))
        elif distrubution_type == 'standard_normal':
            series = pd.Series(np.random.standard_normal(self.get_shape()[0]))
        elif distrubution_type == 'normal':
            series = pd.Series(np.random.normal(mean, sd, self.get_shape()[0]))
        else:
            series = pd.Series(np.random.randn(self.get_shape()[0]))
        self.add_column(column_name, series)
        return self.dataframe
    
    def drop_full_nan_columns(self):
        for c in self.dataframe.columns:
                miss = self.dataframe[c].isnull().sum()
                missing_data_percent = round((miss/self.get_shape()[0])*100, 2)
                if missing_data_percent == 100:
                    self.drop_column(c)
                    
    def drop_columns_with_nan_threshold(self, threshold=0.5):
        for c in self.dataframe.columns:
                miss = self.dataframe[c].isnull().sum()
                missing_data_percent = round((miss/self.get_shape()[0])*100, 2)
                if missing_data_percent >= threshold*100:
                    self.drop_column(c)
    
    def get_index(self, as_list=True):
        if as_list is True:
            return self.dataframe.index.to_list()
        return self.dataframe.index
    
    def add_time_serie_row(self, date_column, value_column, value, date_format='%Y'):
        last_date = self.get_index()[-1] + timedelta(days=1)
        dataframe = DataFrame([{value_column: value, date_column: last_date}], data_type='dict')
        dataframe.to_time_series(date_column, value_column, one_row=True, date_format=date_format)
        self.append_dataframe(dataframe.get_dataframe())
        
    def set_generator(self, generator):
        self.__generator = generator

    def set_dataframe(self, data, data_type='df'):
        if data_type == 'matrix':
            index_name = [i for i in range(len(data))]
            colums_name = [i for i in range(len(data[0]))]
            self.dataframe = pd.DataFrame(data=data, index=index_name, columns=colums_name)
        elif data_type == 'df':
            self.dataframe = data

    def is_empty(self):
        return self.get_shape()[0] == 0
    
    def get_columns_types(self, show=True):
        types = self.get_dataframe().dtypes
        if show:
            print(types)
        return types
    
    def set_data_types(self, column_dict_types):
        self.dataframe = self.get_dataframe().astype(column_dict_types)
        
    def set_same_type(self, same_type='float64'):
        """
        example of types: float64, object
        """
        for p in self.get_columns_names():
            self.set_column_type(p, same_type)

    
    def describe(self, show=True, columns=None, sample_size=None):
        """
        Generate descriptive statistics for the dataset.

        For regular DataFrames, uses pandas.DataFrame.describe().
        For Parquet inputs:
          - single parquet file: streams row groups via ParquetFile
          - parquet dataset: uses pyarrow.dataset scanner (to_batches / to_table)
        """
        # Non-parquet: default pandas behavior
        if self.data_type != 'parquet' or not hasattr(self, 'dataset'):
            description = self.get_dataframe().describe()
            if show:
                print(description)
            return description

        import os
        import numpy as np
        import pandas as pd
        import pyarrow as pa
        import pyarrow.compute as pc
        import pyarrow.parquet as pq

        # Helpers
        def is_numeric_type(t: pa.DataType) -> bool:
            return pa.types.is_integer(t) or pa.types.is_floating(t)

        def finalize_stats(values_np: np.ndarray) -> dict:
            # values_np should be 1D numeric ndarray with NaNs removed
            if values_np.size == 0:
                return None
            return {
                'count': int(values_np.size),
                'mean': float(np.mean(values_np)),
                'std': float(np.std(values_np, ddof=0)),
                'min': float(np.min(values_np)),
                '25%': float(np.percentile(values_np, 25)),
                '50%': float(np.percentile(values_np, 50)),
                '75%': float(np.percentile(values_np, 75)),
                'max': float(np.max(values_np)),
            }

        # Determine if input path is a single parquet file vs dataset
        is_single_file = os.path.isfile(self.data_path) and str(self.data_path).lower().endswith('.parquet')

        # Resolve schema and target columns
        if is_single_file:
            pf = pq.ParquetFile(self.data_path)
            schema = pf.schema_arrow
            all_cols = list(schema.names)
            if columns is None:
                target_cols = [f.name for f in schema if is_numeric_type(f.type)]
            else:
                missing = [c for c in columns if c not in all_cols]
                if missing:
                    print(f"⚠️ Columns not found: {missing}")
                target_cols = [c for c in columns if c in all_cols]
        else:
            ds_schema = self.dataset.schema
            all_cols = list(ds_schema.names)
            if columns is None:
                target_cols = [f.name for f in ds_schema if is_numeric_type(f.type)]
            else:
                missing = [c for c in columns if c not in all_cols]
                if missing:
                    print(f"⚠️ Columns not found: {missing}")
                target_cols = [c for c in columns if c in all_cols]

        if not target_cols:
            return pd.DataFrame()

        # Decide sampling strategy
        if is_single_file:
            try:
                est_rows = pf.metadata.num_rows
            except Exception:
                est_rows = None
        else:
            # best effort estimate
            try:
                est_rows = sum(f.count_rows() for f in self.dataset.get_fragments())
            except Exception:
                est_rows = None

        if sample_size is None:
            if est_rows is not None and est_rows >= 500_000:
                sample_size = min(500_000, max(1, est_rows // 10))
            else:
                sample_size = None  # full
        else:
            sample_size = int(sample_size) if sample_size is not None else None
            if sample_size is not None and sample_size <= 0:
                sample_size = None

        stats = {}

        # Compute stats per column
        for col in target_cols:
            try:
                if is_single_file:
                    # Single parquet file path
                    if sample_size:
                        # Collect up to sample_size values streaming row groups
                        collected = 0
                        chunks = []
                        for rg_idx in range(pf.num_row_groups):
                            rg_tbl = pf.read_row_group(rg_idx, columns=[col], use_threads=True)
                            arr = rg_tbl.column(0).combine_chunks()
                            vals = arr.to_numpy(zero_copy_only=False)
                            # drop NaNs
                            vals = vals[np.isfinite(vals)]
                            if vals.size == 0:
                                continue
                            need = sample_size - collected
                            if need <= 0:
                                break
                            take = vals[:need] if vals.size >= need else vals
                            chunks.append(take)
                            collected += take.size
                            if collected >= sample_size:
                                break
                        values_np = np.concatenate(chunks) if chunks else np.array([], dtype=float)
                        col_stats = finalize_stats(values_np)
                    else:
                        # Full column via row groups (concatenate)
                        pieces = []
                        for rg_idx in range(pf.num_row_groups):
                            rg_tbl = pf.read_row_group(rg_idx, columns=[col], use_threads=True)
                            arr = rg_tbl.column(0).combine_chunks()
                            vals = arr.to_numpy(zero_copy_only=False)
                            vals = vals[np.isfinite(vals)]
                            if vals.size:
                                pieces.append(vals)
                        values_np = np.concatenate(pieces) if pieces else np.array([], dtype=float)
                        col_stats = finalize_stats(values_np)
                else:
                    # Dataset path (use scanner). Avoid Scanner.scan(); use to_batches/to_table.
                    if sample_size:
                        scanner = self.dataset.scanner(columns=[col], batch_size=min(100_000, sample_size))
                        collected = 0
                        chunks = []
                        for batch in scanner.to_batches():
                            arr = batch.column(0)
                            vals = arr.to_numpy(zero_copy_only=False)
                            vals = vals[np.isfinite(vals)]
                            if vals.size == 0:
                                continue
                            need = sample_size - collected
                            if need <= 0:
                                break
                            take = vals[:need] if vals.size >= need else vals
                            chunks.append(take)
                            collected += take.size
                            if collected >= sample_size:
                                break
                        values_np = np.concatenate(chunks) if chunks else np.array([], dtype=float)
                        col_stats = finalize_stats(values_np)
                    else:
                        # Full column table
                        table = self.dataset.scanner(columns=[col]).to_table()
                        arr = table.column(0).combine_chunks()
                        vals = arr.to_numpy(zero_copy_only=False)
                        vals = vals[np.isfinite(vals)]
                        col_stats = finalize_stats(vals)

                if col_stats is not None:
                    stats[col] = col_stats
            except Exception as e:
                print(f"⚠️ Error computing statistics for {col}: {e}")

        # Convert to pandas DataFrame with standard order
        if stats:
            description = pd.DataFrame(stats).T
            description = description.reindex(
                columns=['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
            )
        else:
            description = pd.DataFrame()

        if show:
            print(description)
        return description
    
    
    def describe_old(self, show=True, columns=None, sample_size=None):
        """
        Generate descriptive statistics for the dataset.
        
        For regular DataFrames, uses pandas.DataFrame.describe().
        For Parquet datasets, computes statistics efficiently without loading all data.
        
        Parameters:
        -----------
        show : bool, default True
            Whether to print the description
        columns : list, optional
            Specific columns to describe. If None, describes all numeric columns.
        sample_size : int, optional
            For Parquet datasets, maximum number of rows to sample for statistics.
            If None, uses heuristics based on dataset size.
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with descriptive statistics
        """
        # For standard pandas dataframes, use existing behavior
        if self.data_type != 'parquet' or not hasattr(self, 'dataset'):
            description = self.get_dataframe().describe()
            if show:
                print(description)
            return description
            
        # For Parquet datasets, compute statistics efficiently
        import pyarrow.compute as pc
        import numpy as np
        import pandas as pd
        
        # Determine columns to describe
        schema = self.dataset.schema
        if columns is None:
            # Filter to include only numeric columns by default
            numeric_types = (pa.int8(), pa.int16(), pa.int32(), pa.int64(), 
                             pa.uint8(), pa.uint16(), pa.uint32(), pa.uint64(),
                             pa.float16(), pa.float32(), pa.float64())
            columns = [field.name for field in schema if field.type in numeric_types]
        else:
            # Validate that requested columns exist
            all_cols = set(schema.names)
            missing = [col for col in columns if col not in all_cols]
            if missing:
                print(f"⚠️ Columns not found: {missing}")
                columns = [col for col in columns if col in all_cols]
                if not columns:
                    return pd.DataFrame()
        
        # Determine if we should use sampling
        if sample_size is None:
            # Try to estimate dataset size
            try:
                fragments = list(self.dataset.get_fragments())
                est_size = sum(fragment.count_rows() for fragment in fragments)
                # Use full dataset for small datasets, sample for large ones
                sample_size = None if est_size < 500000 else min(500000, est_size // 10)
            except Exception:
                # Fallback to a reasonable default
                sample_size = 100000
        
        # Prepare to collect statistics
        stats = {}
        
        # For each column, compute statistics
        for col in columns:
            try:
                # For sampling approach
                if sample_size:
                    # Read a sample of the data
                    scanner = self.dataset.scanner(columns=[col], batch_size=min(100000, sample_size))
                    batches = list(scanner.scan())
                    if not batches:
                        continue
                        
                    # Combine batches
                    combined = pa.concat_arrays([batch.column(0) for batch in batches])
                    
                    # Compute statistics
                    count = pc.count(combined).as_py()
                    if count == 0:
                        continue
                    
                    # Basic statistics
                    col_stats = {
                        'count': count,
                        'mean': pc.mean(combined).as_py(),
                        'std': pc.stddev(combined).as_py(),
                        'min': pc.min(combined).as_py(),
                        '25%': np.percentile(combined.to_numpy(), 25),
                        '50%': pc.approximate_median(combined).as_py(),
                        '75%': np.percentile(combined.to_numpy(), 75),
                        'max': pc.max(combined).as_py()
                    }
                else:
                    # Full dataset approach - use PyArrow compute functions
                    col_stats = {
                        'count': self.dataset.count_rows(filter=pc.is_valid(pc.field(col))),
                        'mean': self.dataset.to_table([col]).column(0).mean().as_py(),
                        'std': self.dataset.to_table([col]).column(0).std().as_py(),
                        'min': self.dataset.to_table([col]).column(0).min().as_py(),
                        'max': self.dataset.to_table([col]).column(0).max().as_py(),
                    }
                    # We need to read the data for percentiles
                    data = self.dataset.to_table([col]).column(0).to_numpy()
                    col_stats.update({
                        '25%': np.percentile(data, 25),
                        '50%': np.percentile(data, 50),
                        '75%': np.percentile(data, 75),
                    })
                    
                stats[col] = col_stats
            except Exception as e:
                print(f"⚠️ Error computing statistics for {col}: {e}")
        
        # Convert to pandas DataFrame with the same format as pandas.describe()
        if stats:
            description = pd.DataFrame(stats).T
            # Ensure consistent column order with pandas
            description = description.reindex(columns=['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'])
        else:
            description = pd.DataFrame()
        
        if show:
            print(description)
        return description
    
    
    def scale_offset(self, columns_names_list, scale: float = 2.75e-05, offset: float = -0.2):
        """
        Apply a linear transform x -> x * scale + offset to the given columns.

        Args:
            columns_names_list (list[str] | str): Column name or list of columns to transform.
            scale (float): Multiplicative factor.
            offset (float): Additive offset applied after scaling.

        Returns:
            pandas.DataFrame: The updated dataframe.

        Notes:
            - Non-numeric values are coerced to NaN before transformation.
            - This method operates on the in-memory DataFrame. If the source is a Parquet
              dataset, ensure data is loaded into self.dataframe first (e.g., via a prior read).
        """
        # Normalize to list
        if isinstance(columns_names_list, str):
            columns = [columns_names_list]
        else:
            columns = list(columns_names_list or [])

        if not columns:
            return self.get_dataframe()

        if self.data_type == 'parquet' and hasattr(self, 'dataset') and (self.dataframe is None or self.dataframe.empty):
            print("⚠️ scale_offset works on the in-memory DataFrame. Load data first or add a parquet-specific method.")
            return self.get_dataframe()

        for col in columns:
            if col not in self.dataframe.columns:
                print(f"⚠️ Column '{col}' not found; skipping.")
                continue
            self.dataframe[col] = pd.to_numeric(self.dataframe[col], errors="coerce") * scale + offset

        return self.get_dataframe()
    
    def add_ndvi_column(
        self,
        red_band_name: str = "SR_B4_mean",
        nir_band_name: str = "SR_B5_mean",
        new_column_name: str = "NDVI",
        eps: float = 1e-6,
    ):
        """
        NDVI = (NIR - RED) / (NIR + RED + eps)
        """
        import numpy as np
        r = pd.to_numeric(self.get_dataframe()[red_band_name], errors="coerce")
        n = pd.to_numeric(self.get_dataframe()[nir_band_name], errors="coerce")
        ndvi = (n - r) / (n + r + eps)
        self.set_column(new_column_name, ndvi.clip(-1.0, 1.0))
        return self.get_dataframe()

    def add_albedo_column(
        self,
        b2_col: str = "SR_B2_mean",
        b3_col: str = "SR_B3_mean",
        b4_col: str = "SR_B4_mean",
        b5_col: str = "SR_B5_mean",
        b6_col: str = "SR_B6_mean",
        b7_col: str = "SR_B7_mean",
        new_column_name: str = "ALBEDO",
    ):
        """
        Broadband albedo (OLI, Liang-style):
        α ≈ 0.356*B2 + 0.130*B3 + 0.373*B4 + 0.085*B5 + 0.072*B6 + 0.072*B7 − 0.0018
        Uses SR_B* means. Clips to [0,1].
        """
        import numpy as np
        df = self.get_dataframe()
        B2 = pd.to_numeric(df[b2_col], errors="coerce")
        B3 = pd.to_numeric(df[b3_col], errors="coerce")
        B4 = pd.to_numeric(df[b4_col], errors="coerce")
        B5 = pd.to_numeric(df[b5_col], errors="coerce")
        B6 = pd.to_numeric(df[b6_col], errors="coerce")
        B7 = pd.to_numeric(df[b7_col], errors="coerce")
        albedo = 0.356 * B2 + 0.130 * B3 + 0.373 * B4 + 0.085 * B5 + 0.072 * B6 + 0.072 * B7 - 0.0018
        self.set_column(new_column_name, albedo.clip(lower=0.0, upper=1.0))
        return self.get_dataframe()

    def add_ta_rs_column(
        self,
        st_b10_col: str = "ST_B10_mean",
        ndvi_col: str = "NDVI",
        a0: float = 2.0,
        a1: float = 6.0,
        st_scale: float = 1.0,
        st_offset: float = 0.0,
        new_column_name: str = "Ta_RS_C",
    ):
        """
        T_a^RS (°C) = (T_surface_K - 273.15) - [a0 + a1(1 - NDVI)]
        T_surface_K = st_scale * ST_B10 + st_offset
        """
        Ts = st_scale * pd.to_numeric(self.get_dataframe()[st_b10_col], errors="coerce") + st_offset
        ndvi = pd.to_numeric(self.get_dataframe()[ndvi_col], errors="coerce")
        ta_rs = (Ts - 273.15) - (a0 + a1 * (1.0 - ndvi))
        self.set_column(new_column_name, ta_rs)
        return self.get_dataframe()

    def add_swdown_rs_column(
        self,
        doy_col: str = "doy",
        lat_col: str = "lat",
        elevation_col: str = "elevation",
        TL: float = 2.3,
        new_column_name: str = "SWdown_RS",
        tl_used_col: str = "TL_used",
    ):
        """
        Clear-sky SW↓ at overpass:
        d = 1 + 0.033 cos(2π DOY/365)
        δ = 0.409 sin(2π DOY/365 - 1.39)
        μ0 = max(0, sinφ sinδ + cosφ cosδ)  [solar-noon approx]
        I0 = 1361 * μ0 / d^2
        θ = arccos(μ0) [deg]; m = 1/(μ0 + 0.50572 (6.07995+θ)^-1.6364)
        P(z)=101.325 ((293-0.0065 z)/293)^5.26 (kPa)
        SW↓ = I0 * exp{-0.8662 TL m (P/101.325)}
        """
        import numpy as np
        df = self.get_dataframe()
        DOY = pd.to_numeric(df[doy_col], errors="coerce")
        lat = np.deg2rad(pd.to_numeric(df[lat_col], errors="coerce"))
        z = pd.to_numeric(df[elevation_col], errors="coerce")

        B = 2.0 * np.pi * DOY / 365.0
        d = 1.0 + 0.033 * np.cos(B)
        delta = 0.409 * np.sin(B - 1.39)
        mu0 = np.sin(lat) * np.sin(delta) + np.cos(lat) * np.cos(delta)  # noon approx (ω=0)
        mu0 = np.clip(mu0, 0.0, 1.0)

        I0 = 1361.0 * mu0 / (d * d)

        # Kasten–Young airmass
        theta_rad = np.arccos(np.clip(mu0, 0.0, 1.0))
        theta_deg = np.degrees(theta_rad)
        m = 1.0 / (mu0 + 0.50572 * (6.07995 + theta_deg) ** (-1.6364))
        # Pressure (kPa)
        P = 101.325 * ((293.0 - 0.0065 * z) / 293.0) ** 5.26
        trans = np.exp(-0.8662 * TL * m * (P / 101.325))
        sw = I0 * trans
        # Zero out for μ0<=0
        sw = np.where(mu0 <= 0.0, 0.0, sw)

        self.set_column(new_column_name, sw)
        self.set_column(tl_used_col, pd.Series(TL, index=df.index))
        return self.get_dataframe()

    def add_cos_sza(
        self,
        lat_column_name: str,
        date_column_name: str,
        new_column_name: str = "cos_sza",
        output_path: str | None = None,
        *,
        in_place: bool = False,
        overwrite: bool = False,
        compression: str = "zstd",
        preserve_partitions: bool = True,
        row_group_size: int | None = None,
        show_progress: bool = True,
        datetime_format: str | None = None,
        timezone: str | None = None,
    ):
        """Add a cosine solar zenith angle (cos(SZA)) column.

        cos(SZA) is computed at solar noon using the same formulation as
        :meth:`add_swdown_rs_column` (μ0):

            B = 2π * DOY / 365
            δ = 0.409 * sin(B - 1.39)
            φ = latitude [rad]
            cos(SZA) = μ0 = sinφ sinδ + cosφ cosδ

        Parameters
        ----------
        lat_column_name : str
            Name of the latitude column in degrees.
        date_column_name : str
            Name of the date/datetime column from which DOY is extracted.
        new_column_name : str, default "cos_sza"
            Name of the column to create.
        output_path : str, optional
            For parquet inputs: destination file (single parquet) or directory
            (dataset root). If None and ``in_place=True``, overwrites source.
            For in-memory DataFrames, if provided, writes a single parquet file
            (or "with_cos_sza.parquet" inside a directory) and returns summary.
        in_place : bool, default False
            When reading from parquet, overwrite the source path atomically
            (file or directory), similar to :meth:`add_doy_parquet`.
        overwrite : bool, default False
            Allow overwriting ``output_path`` if it already exists.
        compression : str, default "zstd"
            Parquet compression codec.
        preserve_partitions : bool, default True
            For dataset mode, keep the original partition folder structure.
        row_group_size : int, optional
            Optional row-group size for output; if None, writes incoming
            row-groups as-is.
        show_progress : bool, default True
            Show tqdm progress bars during parquet processing.
        datetime_format : str, optional
            Optional strptime format when the date column is string.
        timezone : str, optional
            If provided, naive datetimes are localized to this timezone before
            computing DOY (mainly relevant if the source column is timezone-
            aware or should be interpreted as such).

        Returns
        -------
        pandas.DataFrame or dict
            - For non-parquet inputs: returns the modified in-memory
              DataFrame (and optionally writes a parquet file if
              ``output_path`` is given).
            - For parquet inputs: returns a summary ``dict`` similar to
              :meth:`add_doy_parquet` describing the output.
        """
        import os
        import shutil
        import numpy as np
        import pandas as pd
        import pyarrow as pa
        import pyarrow.parquet as pq
        import pyarrow.compute as pc
        from tqdm import tqdm

        # ---------- helpers ----------
        def _compute_cos_sza_numpy(lat_deg_arr: np.ndarray, doy_arr: np.ndarray) -> np.ndarray:
            lat_rad = np.deg2rad(lat_deg_arr.astype(float))
            B = 2.0 * np.pi * doy_arr.astype(float) / 365.0
            delta = 0.409 * np.sin(B - 1.39)
            mu0 = np.sin(lat_rad) * np.sin(delta) + np.cos(lat_rad) * np.cos(delta)
            return np.clip(mu0, 0.0, 1.0)

        def _compute_cos_sza_pandas(df: pd.DataFrame) -> pd.Series:
            if lat_column_name not in df.columns:
                raise ValueError(f"Column '{lat_column_name}' not found.")
            if date_column_name not in df.columns:
                raise ValueError(f"Column '{date_column_name}' not found.")

            lat_deg = pd.to_numeric(df[lat_column_name], errors="coerce")

            ser = df[date_column_name]
            if pd.api.types.is_string_dtype(ser):
                dt = pd.to_datetime(ser, format=datetime_format, errors="coerce")
            else:
                dt = pd.to_datetime(ser, errors="coerce")

            if timezone and getattr(dt.dt.tz, "zone", None) is None:
                try:
                    dt = dt.dt.tz_localize(timezone)
                except Exception:
                    # If localization fails, continue with naive timestamps
                    pass

            doy = dt.dt.dayofyear.astype(float)
            mu0 = _compute_cos_sza_numpy(lat_deg.to_numpy(), doy.to_numpy())
            return pd.Series(mu0, index=df.index)

        def add_or_replace_col(tbl: pa.Table, name: str, arr: pa.Array) -> pa.Table:
            if name in tbl.column_names:
                idx = tbl.column_names.index(name)
                tbl = tbl.remove_column(idx)
            return tbl.append_column(name, arr)

        def compute_doy_arrow(col_arr: pa.ChunkedArray | pa.Array) -> pa.Array | None:
            """Try to compute DOY using Arrow; return None if unsupported."""
            try:
                t = col_arr
                typ = t.type
                if pa.types.is_date32(typ) or pa.types.is_date64(typ):
                    t = pc.cast(t, pa.timestamp("us"))
                elif pa.types.is_string(typ):
                    if datetime_format:
                        t = pc.strptime(t, format=datetime_format, unit="us", error_is_null=True)
                    else:
                        return None
                elif pa.types.is_timestamp(typ):
                    unit = typ.unit
                    if unit != "us":
                        t = pc.cast(t, pa.timestamp("us"))
                else:
                    return None

                if timezone and isinstance(t.type, pa.TimestampType) and t.type.tz is None:
                    t = pc.assume_timezone(t, timezone)

                doy = pc.day_of_year(t)
                return doy
            except Exception:
                return None

        def compute_doy_pandas(col_arr: pa.ChunkedArray | pa.Array) -> pa.Array:
            """Fallback DOY via pandas, preserving nulls."""
            ser = col_arr.to_pandas(types_mapper=None)
            if pd.api.types.is_string_dtype(ser):
                ser_dt = pd.to_datetime(ser, format=datetime_format, errors="coerce", utc=False)
            else:
                ser_dt = pd.to_datetime(ser, errors="coerce", utc=False)
            if timezone and getattr(ser_dt.dt.tz, "zone", None) is None:
                try:
                    ser_dt = ser_dt.dt.tz_localize(timezone)
                except Exception:
                    pass
            doy = ser_dt.dt.dayofyear.astype("Int32")
            return pa.array(doy, type=pa.int32())

        # -------------- In-memory pandas path --------------
        if self.data_type != "parquet" or not hasattr(self, "dataset"):
            df = self.get_dataframe()
            mu0 = _compute_cos_sza_pandas(df)
            self.set_column(new_column_name, mu0)

            out_file = None
            if output_path:
                out_path = output_path
                if not out_path.lower().endswith(".parquet"):
                    os.makedirs(out_path, exist_ok=True)
                    out_path = os.path.join(out_path, "with_cos_sza.parquet")
                if os.path.exists(out_path) and not overwrite:
                    raise FileExistsError(f"{out_path} exists (overwrite=False).")
                df.to_parquet(out_path, index=False, compression=compression)
                out_file = out_path

            # Keep existing behavior (return DataFrame) while optionally writing
            # parquet if requested.
            if output_path:
                return {
                    "mode": "pandas",
                    "output_file": out_file,
                    "rows": len(df),
                    "added_column": new_column_name,
                    "source": "dataframe",
                }
            return self.get_dataframe()

        # -------------- Parquet path --------------
        if not hasattr(self, "dataset"):
            raise ValueError("Parquet dataset is not initialized on this DataFrame.")

        schema_names = list(self.dataset.schema.names)
        missing_cols = [
            col for col in [lat_column_name, date_column_name] if col not in schema_names
        ]
        if missing_cols:
            raise ValueError(f"Columns not found in parquet schema: {missing_cols}")

        is_single_file = os.path.isfile(self.data_path) and self.data_path.lower().endswith(".parquet")

        # Resolve output target
        if is_single_file:
            src_file = os.path.abspath(self.data_path)
            if in_place and output_path is None:
                out_file = src_file + ".tmp_with_cos_sza.parquet"
            else:
                if output_path:
                    if output_path.lower().endswith(".parquet"):
                        out_file = os.path.abspath(output_path)
                        os.makedirs(os.path.dirname(out_file) or ".", exist_ok=True)
                    else:
                        os.makedirs(output_path, exist_ok=True)
                        base = os.path.splitext(os.path.basename(src_file))[0]
                        out_file = os.path.join(output_path, f"{base}_with_cos_sza.parquet")
                else:
                    base = os.path.splitext(os.path.basename(src_file))[0]
                    out_file = os.path.join(os.path.dirname(src_file), f"{base}_with_cos_sza.parquet")
            if os.path.exists(out_file) and not overwrite:
                raise FileExistsError(f"{out_file} exists (overwrite=False).")
        else:
            src_root = os.path.abspath(self.data_path)
            if in_place and output_path is None:
                out_root = src_root + "_tmp_with_cos_sza"
            else:
                if output_path is None:
                    raise ValueError("For dataset mode, provide output_path or set in_place=True.")
                out_root = os.path.abspath(output_path)
            if os.path.exists(out_root) and not overwrite:
                raise FileExistsError(f"{out_root} exists (overwrite=False).")
            os.makedirs(out_root, exist_ok=True)

        total_rows = 0
        writer = None
        files_written = 0
        row_groups_written = 0

        # -------- Single-file mode --------
        if is_single_file:
            pf = pq.ParquetFile(self.data_path)
            if date_column_name not in pf.schema_arrow.names or lat_column_name not in pf.schema_arrow.names:
                raise ValueError("Required columns not found in parquet file.")

            # Try to keep source compression
            try:
                src_codec = pf.metadata.row_group(0).column(0).compression
                if src_codec and compression == "zstd":
                    compression = src_codec
            except Exception:
                pass

            rg_iter = tqdm(range(pf.num_row_groups), desc="RowGroups", disable=not show_progress)
            for rg_idx in rg_iter:
                rg_tbl = pf.read_row_group(rg_idx, use_threads=True)
                total_rows += rg_tbl.num_rows
                if rg_tbl.num_rows == 0:
                    continue

                if date_column_name not in rg_tbl.column_names or lat_column_name not in rg_tbl.column_names:
                    raise ValueError(
                        f"Columns '{date_column_name}' and/or '{lat_column_name}' not found in row-group {rg_idx}."
                    )

                lat_col = rg_tbl[lat_column_name]
                date_col = rg_tbl[date_column_name]

                doy_arrow = compute_doy_arrow(date_col)
                if doy_arrow is None:
                    doy_arr = compute_doy_pandas(date_col)
                else:
                    doy_arr = pc.cast(doy_arrow, pa.int32())

                lat_np = lat_col.to_numpy(zero_copy_only=False)
                doy_np = doy_arr.to_numpy(zero_copy_only=False)
                mu0_np = _compute_cos_sza_numpy(lat_np, doy_np)
                mu0_arr = pa.array(mu0_np.astype("float32"))

                rg_tbl = add_or_replace_col(rg_tbl, new_column_name, mu0_arr)

                if writer is None:
                    writer = pq.ParquetWriter(
                        out_file,
                        rg_tbl.schema,
                        compression=compression,
                        use_dictionary=True,
                        write_statistics=True,
                    )

                if row_group_size:
                    n = rg_tbl.num_rows
                    start = 0
                    while start < n:
                        end = min(start + row_group_size, n)
                        writer.write_table(rg_tbl.slice(start, end - start))
                        start = end
                else:
                    writer.write_table(rg_tbl)

                row_groups_written += 1
                rg_iter.set_postfix(rows=f"{total_rows:,}")

            if writer is not None:
                writer.close()

            final_path = out_file
            if in_place and output_path is None:
                os.replace(out_file, src_file)
                final_path = src_file

            return {
                "mode": "single_file",
                "output_file": final_path,
                "rows": total_rows,
                "row_groups_written": row_groups_written,
                "added_column": new_column_name,
                "compression": compression,
            }

        # -------- Dataset mode --------
        fragments = list(self.dataset.get_fragments())
        frag_bar = tqdm(fragments, desc="Fragments", disable=not show_progress)

        for fragment in frag_bar:
            frag_rel = fragment.path
            src_abs = frag_rel if os.path.isabs(frag_rel) else os.path.join(src_root, frag_rel)

            try:
                pf = pq.ParquetFile(src_abs)
            except Exception as e:
                print(f"⚠️ Skip unreadable fragment {frag_rel}: {e}")
                continue

            if date_column_name not in pf.schema_arrow.names or lat_column_name not in pf.schema_arrow.names:
                raise ValueError(f"Required columns not found in fragment: {frag_rel}")

            if os.path.isabs(frag_rel):
                rel_path = os.path.relpath(frag_rel, src_root)
            else:
                rel_path = frag_rel
            if preserve_partitions:
                out_dir = os.path.join(out_root, os.path.dirname(rel_path))
            else:
                out_dir = out_root
            os.makedirs(out_dir, exist_ok=True)
            out_file = os.path.join(out_dir, os.path.basename(rel_path))

            frag_compression = compression
            try:
                src_codec = pf.metadata.row_group(0).column(0).compression
                if src_codec and compression == "zstd":
                    frag_compression = src_codec
            except Exception:
                pass

            writer = None
            for rg_idx in range(pf.num_row_groups):
                rg_tbl = pf.read_row_group(rg_idx, use_threads=True)
                total_rows += rg_tbl.num_rows
                if rg_tbl.num_rows == 0:
                    continue

                if date_column_name not in rg_tbl.column_names or lat_column_name not in rg_tbl.column_names:
                    raise ValueError(
                        f"Columns '{date_column_name}' and/or '{lat_column_name}' not found in row-group {rg_idx} of {frag_rel}"
                    )

                lat_col = rg_tbl[lat_column_name]
                date_col = rg_tbl[date_column_name]

                doy_arrow = compute_doy_arrow(date_col)
                if doy_arrow is None:
                    doy_arr = compute_doy_pandas(date_col)
                else:
                    doy_arr = pc.cast(doy_arrow, pa.int32())

                lat_np = lat_col.to_numpy(zero_copy_only=False)
                doy_np = doy_arr.to_numpy(zero_copy_only=False)
                mu0_np = _compute_cos_sza_numpy(lat_np, doy_np)
                mu0_arr = pa.array(mu0_np.astype("float32"))

                rg_tbl = add_or_replace_col(rg_tbl, new_column_name, mu0_arr)

                if writer is None:
                    writer = pq.ParquetWriter(
                        out_file,
                        rg_tbl.schema,
                        compression=frag_compression,
                        use_dictionary=True,
                        write_statistics=True,
                    )

                if row_group_size:
                    n = rg_tbl.num_rows
                    start = 0
                    while start < n:
                        end = min(start + row_group_size, n)
                        writer.write_table(rg_tbl.slice(start, end - start))
                        start = end
                else:
                    writer.write_table(rg_tbl)

                row_groups_written += 1

            if writer is not None:
                writer.close()
                files_written += 1
                frag_bar.set_postfix(files=files_written, rows=f"{total_rows:,}")

        final_root = out_root
        if in_place and output_path is None:
            shutil.rmtree(src_root)
            os.replace(out_root, src_root)
            final_root = src_root

        if in_place:
            self.data_path = final_root
            try:
                self.dataset = ds.dataset(self.data_path, format="parquet")
            except Exception:
                pass

        return {
            "mode": "dataset",
            "output_root": final_root,
            "files_written": files_written,
            "rows": total_rows,
            "row_groups_written": row_groups_written,
            "added_column": new_column_name,
            "compression": compression,
            "preserve_partitions": preserve_partitions,
        }

    def add_pressure_from_elevation(
        self,
        elevation_column: str = "elevation",
        new_column: str = "P_kPa",
        output_path: str | None = None,
        *,
        in_place: bool = False,
        overwrite: bool = False,
        compression: str = "zstd",
        preserve_partitions: bool = True,
        row_group_size: int | None = None,
        show_progress: bool = True,
    ):
        """Add atmospheric pressure (kPa) from elevation using FAO-56 style formula.

        Uses the standard barometric approximation (consistent with FAO-56 and
        :meth:`add_et_pt_bound_column`):

            P = 101.325 * ((293 - 0.0065 z) / 293)^5.26

        where ``z`` is elevation above sea level in meters and ``P`` is in kPa.

        Parameters
        ----------
        elevation_column : str, default "elevation"
            Name of the elevation (DEM) column in meters.
        new_column : str, default "P_kPa"
            Name of the pressure column to create.
        output_path : str, optional
            For parquet inputs: destination file (single parquet) or directory
            (dataset root). If None and ``in_place=True``, overwrites source.
            For in-memory DataFrames, if provided, writes a single parquet file
            (or "with_pressure.parquet" inside a directory).
        in_place : bool, default False
            When reading from parquet, overwrite the source path atomically
            (file or directory), similarly to :meth:`add_doy_parquet`.
        overwrite : bool, default False
            Allow overwriting ``output_path`` if it already exists.
        compression : str, default "zstd"
            Parquet compression codec.
        preserve_partitions : bool, default True
            For dataset mode, keep the original partition folder structure.
        row_group_size : int, optional
            Optional row-group size for output; if None, writes incoming
            row-groups as-is.
        show_progress : bool, default True
            Show tqdm progress bars during parquet processing.

        Returns
        -------
        pandas.DataFrame or dict
            - For non-parquet inputs: returns the modified in-memory
              DataFrame (and optionally writes a parquet file if
              ``output_path`` is given).
            - For parquet inputs: returns a summary ``dict`` describing the
              output (single file or dataset).
        """
        import os
        import shutil
        import numpy as np
        import pandas as pd
        import pyarrow as pa
        import pyarrow.parquet as pq
        from tqdm import tqdm

        def _pressure_from_z_numpy(z_arr: np.ndarray) -> np.ndarray:
            """Compute pressure (kPa) from elevation (m) using FAO-56-like formula."""
            return 101.325 * ((293.0 - 0.0065 * z_arr.astype(float)) / 293.0) ** 5.26

        def _pressure_from_z_pandas(df: pd.DataFrame) -> pd.Series:
            if elevation_column not in df.columns:
                raise ValueError(f"Column '{elevation_column}' not found.")
            z = pd.to_numeric(df[elevation_column], errors="coerce")
            P = _pressure_from_z_numpy(z.to_numpy())
            return pd.Series(P, index=df.index)

        def add_or_replace_col(tbl: pa.Table, name: str, arr: pa.Array) -> pa.Table:
            if name in tbl.column_names:
                idx = tbl.column_names.index(name)
                tbl = tbl.remove_column(idx)
            return tbl.append_column(name, arr)

        # -------------- In-memory pandas path --------------
        if self.data_type != "parquet" or not hasattr(self, "dataset"):
            df = self.get_dataframe()
            P_series = _pressure_from_z_pandas(df)
            self.set_column(new_column, P_series)

            out_file = None
            if output_path:
                out_path = output_path
                if not out_path.lower().endswith(".parquet"):
                    os.makedirs(out_path, exist_ok=True)
                    out_path = os.path.join(out_path, "with_pressure.parquet")
                if os.path.exists(out_path) and not overwrite:
                    raise FileExistsError(f"{out_path} exists (overwrite=False).")
                df.to_parquet(out_path, index=False, compression=compression)
                out_file = out_path

            if output_path:
                return {
                    "mode": "pandas",
                    "output_file": out_file,
                    "rows": len(df),
                    "added_column": new_column,
                    "source": "dataframe",
                }
            return self.get_dataframe()

        # -------------- Parquet path --------------
        if not hasattr(self, "dataset"):
            raise ValueError("Parquet dataset is not initialized on this DataFrame.")

        schema_names = list(self.dataset.schema.names)
        if elevation_column not in schema_names:
            raise ValueError(f"Column '{elevation_column}' not found in parquet schema.")

        is_single_file = os.path.isfile(self.data_path) and self.data_path.lower().endswith(".parquet")

        # Resolve output target
        if is_single_file:
            src_file = os.path.abspath(self.data_path)
            if in_place and output_path is None:
                out_file = src_file + ".tmp_with_pressure.parquet"
            else:
                if output_path:
                    if output_path.lower().endswith(".parquet"):
                        out_file = os.path.abspath(output_path)
                        os.makedirs(os.path.dirname(out_file) or ".", exist_ok=True)
                    else:
                        os.makedirs(output_path, exist_ok=True)
                        base = os.path.splitext(os.path.basename(src_file))[0]
                        out_file = os.path.join(output_path, f"{base}_with_pressure.parquet")
                else:
                    base = os.path.splitext(os.path.basename(src_file))[0]
                    out_file = os.path.join(os.path.dirname(src_file), f"{base}_with_pressure.parquet")
            if os.path.exists(out_file) and not overwrite:
                raise FileExistsError(f"{out_file} exists (overwrite=False).")
        else:
            src_root = os.path.abspath(self.data_path)
            if in_place and output_path is None:
                out_root = src_root + "_tmp_with_pressure"
            else:
                if output_path is None:
                    raise ValueError("For dataset mode, provide output_path or set in_place=True.")
                out_root = os.path.abspath(output_path)
            if os.path.exists(out_root) and not overwrite:
                raise FileExistsError(f"{out_root} exists (overwrite=False).")
            os.makedirs(out_root, exist_ok=True)

        total_rows = 0
        writer = None
        files_written = 0
        row_groups_written = 0

        # -------- Single-file mode --------
        if is_single_file:
            pf = pq.ParquetFile(self.data_path)
            if elevation_column not in pf.schema_arrow.names:
                raise ValueError(f"Column '{elevation_column}' not found in parquet file.")

            # Try to keep source compression
            try:
                src_codec = pf.metadata.row_group(0).column(0).compression
                if src_codec and compression == "zstd":
                    compression = src_codec
            except Exception:
                pass

            rg_iter = tqdm(range(pf.num_row_groups), desc="RowGroups", disable=not show_progress)
            for rg_idx in rg_iter:
                rg_tbl = pf.read_row_group(rg_idx, use_threads=True)
                total_rows += rg_tbl.num_rows
                if rg_tbl.num_rows == 0:
                    continue

                if elevation_column not in rg_tbl.column_names:
                    raise ValueError(f"Column '{elevation_column}' not found in row-group {rg_idx}.")

                z_col = rg_tbl[elevation_column]
                z_np = z_col.to_numpy(zero_copy_only=False)
                P_np = _pressure_from_z_numpy(z_np)
                P_arr = pa.array(P_np.astype("float32"))

                rg_tbl = add_or_replace_col(rg_tbl, new_column, P_arr)

                if writer is None:
                    writer = pq.ParquetWriter(
                        out_file,
                        rg_tbl.schema,
                        compression=compression,
                        use_dictionary=True,
                        write_statistics=True,
                    )

                if row_group_size:
                    n = rg_tbl.num_rows
                    start = 0
                    while start < n:
                        end = min(start + row_group_size, n)
                        writer.write_table(rg_tbl.slice(start, end - start))
                        start = end
                else:
                    writer.write_table(rg_tbl)

                row_groups_written += 1
                rg_iter.set_postfix(rows=f"{total_rows:,}")

            if writer is not None:
                writer.close()

            final_path = out_file
            if in_place and output_path is None:
                os.replace(out_file, src_file)
                final_path = src_file

            return {
                "mode": "single_file",
                "output_file": final_path,
                "rows": total_rows,
                "row_groups_written": row_groups_written,
                "added_column": new_column,
                "compression": compression,
            }

        # -------- Dataset mode --------
        fragments = list(self.dataset.get_fragments())
        frag_bar = tqdm(fragments, desc="Fragments", disable=not show_progress)

        for fragment in frag_bar:
            frag_rel = fragment.path
            src_abs = frag_rel if os.path.isabs(frag_rel) else os.path.join(src_root, frag_rel)

            try:
                pf = pq.ParquetFile(src_abs)
            except Exception as e:
                print(f"⚠️ Skip unreadable fragment {frag_rel}: {e}")
                continue

            if elevation_column not in pf.schema_arrow.names:
                raise ValueError(f"Column '{elevation_column}' not found in fragment: {frag_rel}")

            if os.path.isabs(frag_rel):
                rel_path = os.path.relpath(frag_rel, src_root)
            else:
                rel_path = frag_rel
            if preserve_partitions:
                out_dir = os.path.join(out_root, os.path.dirname(rel_path))
            else:
                out_dir = out_root
            os.makedirs(out_dir, exist_ok=True)
            out_file = os.path.join(out_dir, os.path.basename(rel_path))

            frag_compression = compression
            try:
                src_codec = pf.metadata.row_group(0).column(0).compression
                if src_codec and compression == "zstd":
                    frag_compression = src_codec
            except Exception:
                pass

            writer = None
            for rg_idx in range(pf.num_row_groups):
                rg_tbl = pf.read_row_group(rg_idx, use_threads=True)
                total_rows += rg_tbl.num_rows
                if rg_tbl.num_rows == 0:
                    continue

                if elevation_column not in rg_tbl.column_names:
                    raise ValueError(
                        f"Column '{elevation_column}' not found in row-group {rg_idx} of {frag_rel}"
                    )

                z_col = rg_tbl[elevation_column]
                z_np = z_col.to_numpy(zero_copy_only=False)
                P_np = _pressure_from_z_numpy(z_np)
                P_arr = pa.array(P_np.astype("float32"))

                rg_tbl = add_or_replace_col(rg_tbl, new_column, P_arr)

                if writer is None:
                    writer = pq.ParquetWriter(
                        out_file,
                        rg_tbl.schema,
                        compression=frag_compression,
                        use_dictionary=True,
                        write_statistics=True,
                    )

                if row_group_size:
                    n = rg_tbl.num_rows
                    start = 0
                    while start < n:
                        end = min(start + row_group_size, n)
                        writer.write_table(rg_tbl.slice(start, end - start))
                        start = end
                else:
                    writer.write_table(rg_tbl)

                row_groups_written += 1

            if writer is not None:
                writer.close()
                files_written += 1
                frag_bar.set_postfix(files=files_written, rows=f"{total_rows:,}")

        final_root = out_root
        if in_place and output_path is None:
            shutil.rmtree(src_root)
            os.replace(out_root, src_root)
            final_root = src_root

        if in_place:
            self.data_path = final_root
            try:
                self.dataset = ds.dataset(self.data_path, format="parquet")
            except Exception:
                pass

        return {
            "mode": "dataset",
            "output_root": final_root,
            "files_written": files_written,
            "rows": total_rows,
            "row_groups_written": row_groups_written,
            "added_column": new_column,
            "compression": compression,
            "preserve_partitions": preserve_partitions,
        }

    def add_lwdown_and_rn_rs_columns(
        self,
        ta_rs_col: str = "Ta_RS_C",
        st_b10_col: str = "ST_B10_mean",
        albedo_col: str = "ALBEDO",
        swdown_col: str = "SWdown_RS",
        eps_a: float = 0.82,
        eps_s: float = 0.97,
        st_scale: float = 1.0,
        st_offset: float = 0.0,
        new_lw_col: str = "LWdown_RS",
        new_rn_col: str = "Rn_RS",
    ):
        """
        LW↓_RS = ε_a σ (Ta_RS_K)^4
        Rn_RS = (1-α) SW↓_RS + LW↓_RS − ε_s σ (T_surf_K)^4
        """
        import numpy as np
        sigma = 5.670374419e-8  # W m-2 K-4

        TaK = pd.to_numeric(self.get_dataframe()[ta_rs_col], errors="coerce") + 273.15
        TsK = st_scale * pd.to_numeric(self.get_dataframe()[st_b10_col], errors="coerce") + st_offset
        alpha = pd.to_numeric(self.get_dataframe()[albedo_col], errors="coerce")
        sw = pd.to_numeric(self.get_dataframe()[swdown_col], errors="coerce")

        lw_down = eps_a * sigma * np.power(TaK, 4)
        rn = (1.0 - alpha) * sw + lw_down - eps_s * sigma * np.power(TsK, 4)

        self.set_column(new_lw_col, lw_down)
        self.set_column(new_rn_col, rn)
        return self.get_dataframe()

    def add_g_rs_column(
        self,
        rn_col: str = "Rn_RS",
        ndvi_col: str = "NDVI",
        cg_min: float = 0.05,
        cg_max: float = 0.25,
        new_column_name: str = "G_RS",
        cg_min_col: str = "cg_min",
        cg_max_col: str = "cg_max",
    ):
        """
        c_g(NDVI) = c_g_min + (c_g_max - c_g_min) (1 - NDVI)
        G_RS = c_g * Rn_RS
        """
        ndvi = pd.to_numeric(self.get_dataframe()[ndvi_col], errors="coerce")
        rn = pd.to_numeric(self.get_dataframe()[rn_col], errors="coerce")
        cg = cg_min + (cg_max - cg_min) * (1.0 - ndvi)
        self.set_column(new_column_name, cg * rn)
        self.set_column(cg_min_col, pd.Series(cg_min, index=self.get_dataframe().index))
        self.set_column(cg_max_col, pd.Series(cg_max, index=self.get_dataframe().index))
        return self.get_dataframe()

    def add_et_pt_bound_column(
        self,
        ta_rs_col: str = "Ta_RS_C",
        rn_col: str = "Rn_RS",
        g_col: str = "G_RS",
        elevation_col: str = "elevation",
        alpha_PT: float = 1.26,
        dt_seconds: int = 1800,
        Lv_MJ_per_kg: float = 2.45,
        new_column_name: str = "ET_PT_bound",
    ):
        """
        Priestley–Taylor RS upper bound:
        Δ(T)=4098*es(T)/(T+237.3)^2, es(T)=0.6108*exp(17.27T/(T+237.3))
        γ(P,T)=cp*P/(ε*Lv); cp=1013 J/kg/K -> 0.001013 MJ/kg/K, ε=0.622
        LE_PT = α_PT * Δ/(Δ+γ) * max(0, Rn_RS - G_RS)  [W/m2]
        ET_PT_bound (mm/Δt) = LE_PT * Δt / (Lv*1e6)
        """
        import numpy as np

        T = pd.to_numeric(self.get_dataframe()[ta_rs_col], errors="coerce")
        Rn = pd.to_numeric(self.get_dataframe()[rn_col], errors="coerce")
        G = pd.to_numeric(self.get_dataframe()[g_col], errors="coerce")
        z = pd.to_numeric(self.get_dataframe()[elevation_col], errors="coerce")

        # Saturation vapor pressure and slope Δ
        es = 0.6108 * np.exp(17.27 * T / (T + 237.3))  # kPa
        Delta = 4098.0 * es / np.power(T + 237.3, 2)

        # Pressure (kPa)
        P = 101.325 * ((293.0 - 0.0065 * z) / 293.0) ** 5.26
        cp_MJ = 0.001013  # MJ kg-1 K-1
        eps = 0.622
        gamma = cp_MJ * P / (eps * Lv_MJ_per_kg)

        avail = np.maximum(0.0, Rn - G)  # W/m2
        # Convert W/m2 to MJ m-2 s-1: 1 W = 1 J/s = 1e-6 MJ/s
        LE_PT = alpha_PT * (Delta / (Delta + gamma)) * avail  # still W/m2
        ET = (LE_PT * dt_seconds) / (Lv_MJ_per_kg * 1e6)  # mm over Δt

        self.set_column(new_column_name, ET)
        return self.get_dataframe()
    
    
    
    
    def reset_index(self, drop=True):
        if drop is True:
            self.set_dataframe(self.dataframe.reset_index(drop=True))
        else:
            self.set_dataframe(self.dataframe.reset_index())
            
    def get_dataframe_as_sparse_matrix(self):
        return scipy.sparse.csr_matrix(self.dataframe.to_numpy())

    def get_column_as_list(self, column):
        return list(self.get_column(column))
    
    def get_column_as_joined_text(self, column):
        return ' '.join(list(self.get_column(column)))
    
    def rename_index(self, new_name):
        self.dataframe.index.rename(new_name, inplace=True)
        return self.get_dataframe()

    def get_term_doc_matrix_as_df(self, text_column_name, vectorizer_type='count'):
        corpus = list(self.get_column(text_column_name))
        indice = ['doc' + str(i) for i in range(len(corpus))]
        v = Vectorizer(corpus, vectorizer_type=vectorizer_type)
        self.set_dataframe(DataFrame(v.get_sparse_matrix().toarray(), v.get_features_names(),
                                      line_index=indice, data_type='matrix').get_dataframe())

    def get_dataframe_from_dic_list(self, dict_list):
        v = DictVectorizer()
        matrice = v.fit_transform(dict_list)
        self.vectorizer = v
        self.set_dataframe(DataFrame(matrice.toarray(), v.get_feature_names()).get_dataframe())

    def check_decision_function_on_column(self, column_name, decision_func):
        """
        Check if a decision function returns True for all values in a column.
        
        Parameters:
        -----------
        column_name : str
            Column to evaluate
        decision_func : callable
            Function returning True/False given a value
            
        Returns:
        --------
        bool
            True if decision function returns True for all non-null values
        """
        # Standard pandas path
        if self.data_type != 'parquet' or not hasattr(self, 'dataset'):
            return all(self.get_column(column_name).dropna().apply(decision_func))
        
        # Parquet path
        import pyarrow.compute as pc
        import numpy as np
        from tqdm import tqdm
        
        # Get thread count from instance
        use_threads = self.n_workers is not None
        
        # Check if column exists in schema
        if column_name not in self.dataset.schema.names:
            raise ValueError(f"Column '{column_name}' not found in parquet schema.")
        
        # First count total rows for better progress tracking
        fragments = list(self.dataset.get_fragments())
        total_rows = 0
        for fragment in fragments:
            try:
                # Try to get row count from fragment metadata if available (fast)
                total_rows += fragment.count_rows()
            except Exception:
                try:
                    # Fall back to loading column and counting (slower)
                    total_rows += len(fragment.to_table(columns=[column_name]), use_threads=use_threads)
                except Exception:
                    pass  # Skip if we can't count
        
        # Process data in batches for better progress tracking
        batch_size = 100000  # Process this many rows before updating progress
        processed_rows = 0
        
        # Use tqdm with total rows for better progress tracking
        with tqdm(total=total_rows, desc=f"Checking {column_name}", unit="rows") as pbar:
            for fragment in fragments:
                try:
                    # Get just the needed column from this fragment
                    table = fragment.to_table(columns=[column_name], use_threads=use_threads)
                    column_chunk = table[column_name]
                    chunk_size = len(column_chunk)
                    
                    # Skip empty fragments
                    if chunk_size == 0:
                        continue
                    
                    # Get only non-null values
                    valid_mask = pc.is_valid(column_chunk).to_numpy()
                    if not np.any(valid_mask):
                        continue  # Skip if all nulls
                    
                    values = column_chunk.to_numpy()[valid_mask]
                    valid_count = len(values)
                    
                    # Process in smaller groups for large arrays
                    for i in range(0, valid_count, batch_size):
                        end_idx = min(i + batch_size, valid_count)
                        batch = values[i:end_idx]
                        
                        # Try vectorized evaluation first
                        try:
                            results = np.fromiter((decision_func(v) for v in batch), 
                                                dtype=bool, count=len(batch))
                            if not np.all(results):
                                return False
                        except Exception:
                            # Fall back to one-by-one evaluation
                            for value in batch:
                                if not decision_func(value):
                                    return False
                        
                        # Update progress after each batch
                        processed_batch_size = end_idx - i
                        pbar.update(processed_batch_size)
                        processed_rows += processed_batch_size
                        
                    # Account for any nulls in the progress update
                    null_count = chunk_size - valid_count
                    if null_count > 0:
                        pbar.update(null_count)
                        processed_rows += null_count
                        
                except Exception as e:
                    raise ValueError(f"Error evaluating fragment: {e}")
        
        return True
    
    def check_decision_function_on_column_v1(self, column, decision_func):
        if all(self.get_column(column).apply(decision_func)):
            return True
        return False
    
    def show_word_occurrences_plot(self, column_name, most_common=50):
        """Generating word occurrences plot from a column

        Args:
            column_name (_type_): column to be used
            most_common (int, optional): number of most frequent term to use. Defaults to 50.
        """
        text = self.get_column_as_joined_text(column_name)
        counter = Counter(text.split(' '))
        data = DataFrame(counter.most_common(most_common), ['term', 'count'], data_type='dict', data_types_in_order=[str,int])
        chart = Chart(data.get_dataframe(), column4x='term', chart_type='bar')
        chart.add_data_to_show('count')
        chart.config('Term occurrences bar chart', 'Terms', 'Occurrences', titile_font_size=30,)
        chart.show()

    def set_dataframe_index(self, liste_indices):
        self.dataframe.index = liste_indices

    def get_shape(self):
        """
        Get the shape of the dataset (rows, columns).
        
        For pandas DataFrames, uses .shape attribute.
        For Parquet datasets, computes shape efficiently without loading all data.
        
        Returns:
        --------
        tuple
            (number_of_rows, number_of_columns)
        """
        # Standard pandas case
        if self.data_type != 'parquet' or not hasattr(self, 'dataset'):
            return self.dataframe.shape
            
        # Parquet dataset case - efficient computation
        try:
            # Get column count from schema
            num_columns = len(self.dataset.schema.names)
            
            # Get row count by scanning fragments (avoids loading all data)
            num_rows = 0
            fragments = list(self.dataset.get_fragments())
            
            # Try fast metadata approach first
            try:
                import pyarrow.parquet as pq
                for fragment in fragments:
                    try:
                        # Use metadata if available (very fast)
                        pf = pq.ParquetFile(fragment.path)
                        num_rows += pf.metadata.num_rows
                    except Exception:
                        # Fallback to fragment count_rows() method
                        try:
                            num_rows += fragment.count_rows()
                        except Exception:
                            # Last resort: actual data read (expensive)
                            table = fragment.to_table()
                            num_rows += table.num_rows
            except Exception as e:
                print(f"Warning: Using slow method to get row count: {e}")
                # Final fallback - this will be slow for large datasets
                table = self.dataset.to_table()
                num_rows = table.num_rows
                
            return (num_rows, num_columns)
            
        except Exception as e:
            print(f"Error getting Parquet dataset shape: {e}")
            # If all methods fail, return (0, 0) as a safe fallback
            return (0, 0)
    

    def set_column(self, column_name, new_column):
        self.dataframe[column_name] = new_column

    def set_column_type(self, column, column_type):
        self.dataframe[column] = self.dataframe[column].astype(column_type)

    def get_lines_columns(self, lines, columns):
        if Lib.check_all_elements_type(columns, str):
            return self.get_dataframe().loc[lines, columns]
        return self.get_dataframe().iloc[lines, columns]
    
    def get_rows(self, 
                 nbr_of_rows=None, 
                 start_index=None, 
                 end_index=None, 
                 index_type=None, 
                 frequency='d', 
                 datetime_format='%Y-%m-%d %H:%M:%S'):
        """
        give a negative value if you want begin from last row
        """
        
        if index_type == 'int':
            if start_index is not None and end_index is not None:
                return self.get_dataframe().iloc[start_index:end_index]
            elif start_index is not None and nbr_of_rows is not None:
                return self.get_dataframe().iloc[start_index:start_index+nbr_of_rows]
            elif start_index is None and end_index is None and nbr_of_rows is not None:
                if nbr_of_rows < 0:
                    return self.get_dataframe().tail(abs(nbr_of_rows))
                else:
                    return self.get_dataframe().head(nbr_of_rows)
        elif index_type == 'datetime':
            if start_index is not None and end_index is not None:
                return self.get_dataframe().loc[start_index:end_index]
            elif start_index is not None and nbr_of_rows is not None:
                start_index = Lib.to_datetime(start_index, datetime_format)
                end_index = start_index + datetime.timedelta(nbr_of_rows)
                return self.get_dataframe().loc[start_index:end_index]
            elif start_index is None and end_index is None and nbr_of_rows is not None:
                if nbr_of_rows < 0:
                    return self.get_dataframe().tail(abs(nbr_of_rows))
                else:
                    return self.get_dataframe().head(nbr_of_rows)
        else:
            if start_index is not None and end_index is not None:
                return self.get_dataframe().loc[start_index:end_index]
            elif start_index is not None and nbr_of_rows is not None:
                return self.get_dataframe().loc[start_index:start_index+nbr_of_rows]
            elif start_index is None and end_index is None and nbr_of_rows is not None:
                if nbr_of_rows < 0:
                    return self.get_dataframe().tail(abs(nbr_of_rows))
                else:
                    return self.get_dataframe().head(nbr_of_rows)
            
    def get_column(self, column):
        return self.get_dataframe()[column]
    
    def get_columns(self, columns_names_as_list):
        return self.get_dataframe()[columns_names_as_list]
    
    def add_noise(self, column_name, num_noises=100):
        """
        Adds random noise to a Pandas time series.
        
        Parameters:
        ts (pandas.Series): the time series to which to add noise
        num_noises (int): the number of noise values to add to the time series
        
        Returns:
        pandas.Series: the time series with noise added
        """
        # Calculate the range of the time series data
        data_range = self.get_column(column_name).max() - self.get_column(column_name).min()
        
        # Add the specified number of random noise values
        for i in range(num_noises):
            # Generate a random index within the time series
            rand_index = self.dataframe.sample().index[0]
            
            # Generate a random noise value within the data range
            noise_value = np.random.uniform(low=-data_range, high=data_range)
            
            # Add the noise value to the time series at the random date
            self.set_row(column_name, rand_index, noise_value)
        
        return 0

    
    def add_doy_parquet(
        self,
        datetime_column: str,
        new_column: str = "doy",
        output_path: str | None = None,
        *,
        in_place: bool = False,
        overwrite: bool = False,
        compression: str = "zstd",
        preserve_partitions: bool = True,
        row_group_size: int | None = None,
        show_progress: bool = True,
        datetime_format: str | None = None,   # used when datetime column is string
        timezone: str | None = None           # optional timezone hint if you need to assume tz
    ) -> dict:
        """
        Add a day-of-year column computed from a datetime column and write a new parquet.

        Works with:
          - in-memory pandas DataFrame (adds column in memory; optionally writes a single file if output_path set)
          - parquet single file (rewrites file)
          - parquet dataset (rewrites to output directory, preserving partitions if desired)

        Args:
            datetime_column: Name of the datetime column (string, date, or timestamp).
            new_column: Name of the new DOY column to create (overwrites if exists).
            output_path: Destination path. For single file, a .parquet file path or a directory.
                         For dataset, a directory root. If None and in_place=True, overwrites source.
            in_place: Overwrite the source (uses temp swap).
            overwrite: Allow overwriting output_path (file/dir).
            compression: Output parquet compression codec.
            preserve_partitions: Keep source partition folder structure (dataset mode).
            row_group_size: Optional row-group size for output (None = write incoming RGs as-is).
            show_progress: Show tqdm progress bars.
            datetime_format: strptime format when the datetime column is string (faster/stricter).
            timezone: If source is naive but should be treated as tz-aware (rarely needed).

        Returns:
            dict summary
        """
        import os
        import shutil
        import numpy as np
        import pandas as pd
        import pyarrow as pa
        import pyarrow.parquet as pq
        import pyarrow.compute as pc
        from tqdm import tqdm

        def add_or_replace_col(tbl: pa.Table, name: str, arr: pa.Array) -> pa.Table:
            if name in tbl.column_names:
                idx = tbl.column_names.index(name)
                tbl = tbl.remove_column(idx)
            return tbl.append_column(name, arr)

        def compute_doy_arrow(col_arr: pa.ChunkedArray | pa.Array) -> pa.Array | None:
            """
            Try to compute DOY using Arrow compute. Return None if not supported -> fallback to pandas.
            """
            try:
                t = col_arr
                typ = t.type
                # Cast date types to timestamp[us] for day_of_year()
                if pa.types.is_date32(typ) or pa.types.is_date64(typ):
                    t = pc.cast(t, pa.timestamp("us"))
                elif pa.types.is_string(typ):
                    # Parse strings (format helps if provided)
                    if datetime_format:
                        # Arrow strptime
                        t = pc.strptime(t, format=datetime_format, unit="us", error_is_null=True)
                    else:
                        # Without a format this may fail; let fallback handle it
                        return None
                elif pa.types.is_timestamp(typ):
                    # Normalize resolution to us
                    unit = typ.unit
                    if unit != "us":
                        t = pc.cast(t, pa.timestamp("us"))
                else:
                    # Unsupported direct type for Arrow path
                    return None

                # Optional timezone assumption if requested and tz-naive
                if timezone and isinstance(t.type, pa.TimestampType) and t.type.tz is None:
                    t = pc.assume_timezone(t, timezone)

                # day_of_year returns int64
                doy = pc.day_of_year(t)
                return doy
            except Exception:
                return None

        def compute_doy_pandas(col_arr: pa.ChunkedArray | pa.Array) -> pa.Array:
            """
            Fallback: convert to pandas datetime and compute dayofyear, preserving nulls.
            """
            ser = col_arr.to_pandas(types_mapper=None)
            # Parse when string, otherwise pd.to_datetime handles date/timestamp fine
            if pd.api.types.is_string_dtype(ser):
                ser_dt = pd.to_datetime(ser, format=datetime_format, errors="coerce", utc=False)
            else:
                ser_dt = pd.to_datetime(ser, errors="coerce", utc=False)
            # Optionally localize if timezone provided and naive
            if timezone and getattr(ser_dt.dt.tz, "zone", None) is None:
                try:
                    ser_dt = ser_dt.dt.tz_localize(timezone)
                except Exception:
                    # If localize fails, ignore
                    pass
            doy = ser_dt.dt.dayofyear.astype("Int32")  # nullable int
            return pa.array(doy, type=pa.int32())

        # -------------- In-memory pandas path --------------
        if self.data_type != "parquet" or not hasattr(self, "dataset"):
            if datetime_column not in self.dataframe.columns:
                raise ValueError(f"Column '{datetime_column}' not found in DataFrame.")
            ser = self.dataframe[datetime_column]
            if pd.api.types.is_string_dtype(ser):
                dt = pd.to_datetime(ser, format=datetime_format, errors="coerce")
            else:
                dt = pd.to_datetime(ser, errors="coerce")
            if timezone and getattr(dt.dt.tz, "zone", None) is None:
                try:
                    dt = dt.dt.tz_localize(timezone)
                except Exception:
                    pass
            self.dataframe[new_column] = dt.dt.dayofyear
            out_file = None
            if output_path:
                out_path = output_path
                if not out_path.lower().endswith(".parquet"):
                    os.makedirs(out_path, exist_ok=True)
                    out_path = os.path.join(out_path, "with_doy.parquet")
                if os.path.exists(out_path) and not overwrite:
                    raise FileExistsError(f"{out_path} exists (overwrite=False).")
                self.dataframe.to_parquet(out_path, index=False, compression=compression)
                out_file = out_path
            return {
                "mode": "pandas",
                "output_file": out_file,
                "rows": len(self.dataframe),
                "added_column": new_column,
                "source": "dataframe",
            }

        # -------------- Parquet path --------------
        is_single_file = os.path.isfile(self.data_path) and self.data_path.lower().endswith(".parquet")

        # Resolve output target
        if is_single_file:
            src_file = os.path.abspath(self.data_path)
            if in_place and output_path is None:
                out_file = src_file + ".tmp_with_doy.parquet"
            else:
                if output_path:
                    if output_path.lower().endswith(".parquet"):
                        out_file = os.path.abspath(output_path)
                        os.makedirs(os.path.dirname(out_file) or ".", exist_ok=True)
                    else:
                        os.makedirs(output_path, exist_ok=True)
                        base = os.path.splitext(os.path.basename(src_file))[0]
                        out_file = os.path.join(output_path, f"{base}_with_doy.parquet")
                else:
                    # default next to source
                    base = os.path.splitext(os.path.basename(src_file))[0]
                    out_file = os.path.join(os.path.dirname(src_file), f"{base}_with_doy.parquet")
            if os.path.exists(out_file) and not overwrite:
                raise FileExistsError(f"{out_file} exists (overwrite=False).")
        else:
            src_root = os.path.abspath(self.data_path)
            if in_place and output_path is None:
                out_root = src_root + "_tmp_with_doy"
            else:
                if output_path is None:
                    raise ValueError("For dataset mode, provide output_path or set in_place=True.")
                out_root = os.path.abspath(output_path)
            if os.path.exists(out_root) and not overwrite:
                raise FileExistsError(f"{out_root} exists (overwrite=False).")
            os.makedirs(out_root, exist_ok=True)

        total_rows = 0
        writer = None
        files_written = 0
        row_groups_written = 0

        if is_single_file:
            pf = pq.ParquetFile(self.data_path)
            if datetime_column not in pf.schema_arrow.names:
                raise ValueError(f"Column '{datetime_column}' not found in parquet file.")
            # Try to keep source compression
            try:
                src_codec = pf.metadata.row_group(0).column(0).compression
                if src_codec and compression == "zstd":
                    compression = src_codec
            except Exception:
                pass

            rg_iter = tqdm(range(pf.num_row_groups), desc="RowGroups", disable=not show_progress)
            for rg_idx in rg_iter:
                rg_tbl = pf.read_row_group(rg_idx, use_threads=True)
                total_rows += rg_tbl.num_rows
                if rg_tbl.num_rows == 0:
                    continue

                if datetime_column not in rg_tbl.column_names:
                    raise ValueError(f"Column '{datetime_column}' not found in row-group {rg_idx}.")

                col = rg_tbl[datetime_column]
                doy_arrow = compute_doy_arrow(col)
                if doy_arrow is None:
                    doy_arr = compute_doy_pandas(col)
                else:
                    # Cast to int32 for compactness
                    doy_arr = pc.cast(doy_arrow, pa.int32())

                rg_tbl = add_or_replace_col(rg_tbl, new_column, doy_arr)

                if writer is None:
                    writer = pq.ParquetWriter(
                        out_file,
                        rg_tbl.schema,
                        compression=compression,
                        use_dictionary=True,
                        write_statistics=True,
                    )

                if row_group_size:
                    n = rg_tbl.num_rows
                    start = 0
                    while start < n:
                        end = min(start + row_group_size, n)
                        writer.write_table(rg_tbl.slice(start, end - start))
                        start = end
                else:
                    writer.write_table(rg_tbl)

                row_groups_written += 1
                rg_iter.set_postfix(rows=f"{total_rows:,}")

            if writer is not None:
                writer.close()

            # In-place swap if requested
            final_path = out_file
            if in_place and output_path is None:
                os.replace(out_file, src_file)  # atomic replace on Windows too
                final_path = src_file

            return {
                "mode": "single_file",
                "output_file": final_path,
                "rows": total_rows,
                "row_groups_written": row_groups_written,
                "added_column": new_column,
                "compression": compression,
            }

        # -------- Dataset mode --------
        fragments = list(self.dataset.get_fragments())
        frag_bar = tqdm(fragments, desc="Fragments", disable=not show_progress)

        for fragment in frag_bar:
            frag_rel = fragment.path
            # Source absolute
            src_abs = frag_rel if os.path.isabs(frag_rel) else os.path.join(src_root, frag_rel)

            try:
                pf = pq.ParquetFile(src_abs)
            except Exception as e:
                print(f"⚠️ Skip unreadable fragment {frag_rel}: {e}")
                continue

            if datetime_column not in pf.schema_arrow.names:
                raise ValueError(f"Column '{datetime_column}' not found in fragment: {frag_rel}")

            # Determine output path
            if os.path.isabs(frag_rel):
                rel_path = os.path.relpath(frag_rel, src_root)
            else:
                rel_path = frag_rel
            if preserve_partitions:
                out_dir = os.path.join(out_root, os.path.dirname(rel_path))
            else:
                out_dir = out_root
            os.makedirs(out_dir, exist_ok=True)
            out_file = os.path.join(out_dir, os.path.basename(rel_path))

            # Try to keep source compression
            frag_compression = compression
            try:
                src_codec = pf.metadata.row_group(0).column(0).compression
                if src_codec and compression == "zstd":
                    frag_compression = src_codec
            except Exception:
                pass

            writer = None
            for rg_idx in range(pf.num_row_groups):
                rg_tbl = pf.read_row_group(rg_idx, use_threads=True)
                total_rows += rg_tbl.num_rows
                if rg_tbl.num_rows == 0:
                    continue

                if datetime_column not in rg_tbl.column_names:
                    raise ValueError(f"Column '{datetime_column}' not found in row-group {rg_idx} of {frag_rel}")

                col = rg_tbl[datetime_column]
                doy_arrow = compute_doy_arrow(col)
                if doy_arrow is None:
                    doy_arr = compute_doy_pandas(col)
                else:
                    doy_arr = pc.cast(doy_arrow, pa.int32())

                rg_tbl = add_or_replace_col(rg_tbl, new_column, doy_arr)

                if writer is None:
                    writer = pq.ParquetWriter(
                        out_file,
                        rg_tbl.schema,
                        compression=frag_compression,
                        use_dictionary=True,
                        write_statistics=True,
                    )

                if row_group_size:
                    n = rg_tbl.num_rows
                    start = 0
                    while start < n:
                        end = min(start + row_group_size, n)
                        writer.write_table(rg_tbl.slice(start, end - start))
                        start = end
                else:
                    writer.write_table(rg_tbl)

                row_groups_written += 1

            if writer is not None:
                writer.close()
                files_written += 1
                frag_bar.set_postfix(files=files_written, rows=f"{total_rows:,}")

        # In-place swap (replace source directory)
        final_root = out_root
        if in_place and output_path is None:
            # Remove original dir and replace
            tmp_root = out_root
            # Ensure all handles are closed before replace
            shutil.rmtree(src_root)
            os.replace(tmp_root, src_root)
            final_root = src_root

        # Refresh dataset handle if pointing in-place
        if in_place:
            self.data_path = final_root
            try:
                self.dataset = ds.dataset(self.data_path, format="parquet")
            except Exception:
                pass

        return {
            "mode": "dataset",
            "output_root": final_root,
            "files_written": files_written,
            "rows": total_rows,
            "row_groups_written": row_groups_written,
            "added_column": new_column,
            "compression": compression,
            "preserve_partitions": preserve_partitions,
        }
    
    
    def rename_columns_parquet(
        self,
        rename_map: dict[str, str],
        output_path: str,
        overwrite: bool = False,
        compression: str = "zstd",
        preserve_partitions: bool = True,
        show_progress: bool = True,
        row_group_size: int | None = None,
    ) -> dict:
        """
        Rename columns in a parquet input (single file or dataset) and write to output_path.

        Args:
            rename_map: dict old_name -> new_name (only existing physical columns will be renamed).
                        Note: partition (directory) columns are virtual and cannot be renamed here.
            output_path: Destination file (.parquet) or directory (dataset root).
            overwrite: Overwrite output if it exists.
            compression: Parquet compression codec for output.
            preserve_partitions: Keep source partition folder structure (dataset mode).
            show_progress: Show tqdm progress bars.
            row_group_size: Optional row-group size (rows) for output; if None, writes incoming RGs as-is.

        Returns:
            dict summary
        """
        import os
        import pyarrow as pa
        import pyarrow.parquet as pq
        from tqdm import tqdm

        if not rename_map:
            raise ValueError("rename_map must be a non-empty dict of {'old':'new'} names.")

        # In-memory fallback (non-parquet)
        if self.data_type != "parquet" or not hasattr(self, "dataset"):
            df = getattr(self, "dataframe", None)
            if df is None or df.empty:
                raise ValueError("No in-memory DataFrame loaded.")
            missing = [c for c in rename_map.keys() if c not in df.columns]
            if missing:
                print(f"⚠️ Missing columns in DataFrame (ignored): {missing}")
            df = df.rename(columns=rename_map)
            # Write output_path as a single parquet file
            out = output_path
            if not out.lower().endswith(".parquet"):
                os.makedirs(out, exist_ok=True)
                out = os.path.join(out, "renamed.parquet")
            if os.path.exists(out) and not overwrite:
                raise FileExistsError(f"{out} exists (overwrite=False).")
            df.to_parquet(out, index=False, compression=compression)
            self.dataframe = df
            return {
                "mode": "pandas",
                "output_file": out,
                "renamed": {k: v for k, v in rename_map.items() if k in df.columns},
                "rows_out": len(df),
                "compression": compression,
            }

        # Parquet path
        is_single_file = os.path.isfile(self.data_path) and self.data_path.lower().endswith(".parquet")

        # Validate schema and prepare rename set
        if is_single_file:
            pf0 = pq.ParquetFile(self.data_path)
            source_cols = list(pf0.schema_arrow.names)
        else:
            source_cols = list(self.dataset.schema.names)

        # Warn for missing (note: partition columns won't be present physically)
        missing_in_schema = [c for c in rename_map.keys() if c not in source_cols]
        if missing_in_schema:
            print(f"⚠️ Columns not found in schema (ignored here): {missing_in_schema}")

        # Prepare output paths
        if is_single_file:
            if output_path.lower().endswith(".parquet"):
                out_file = output_path
                out_dir = os.path.dirname(out_file) or "."
            else:
                os.makedirs(output_path, exist_ok=True)
                base = os.path.splitext(os.path.basename(self.data_path))[0]
                out_file = os.path.join(output_path, f"{base}_renamed.parquet")
                out_dir = output_path
            if os.path.exists(out_file) and not overwrite:
                raise FileExistsError(f"{out_file} exists (overwrite=False).")
            os.makedirs(out_dir, exist_ok=True)
        else:
            out_root = output_path
            if os.path.exists(out_root) and not overwrite:
                raise FileExistsError(f"{out_root} exists (overwrite=False).")
            os.makedirs(out_root, exist_ok=True)

        total_rows_in = 0
        total_rows_out = 0
        total_files = 0
        total_row_groups = 0

        def rename_table_columns(tbl: pa.Table, old_names: list[str]) -> pa.Table:
            new_names = [rename_map.get(n, n) for n in old_names]
            # Ensure uniqueness
            if len(set(new_names)) != len(new_names):
                # Find duplicates
                from collections import Counter
                dup = [n for n, c in Counter(new_names).items() if c > 1]
                raise ValueError(f"Renaming would create duplicate column names: {dup}")
            return tbl.rename_columns(new_names)

        if is_single_file:
            pf = pq.ParquetFile(self.data_path)
            schema_names = list(pf.schema_arrow.names)
            # Validate target duplicates on full-file schema
            _ = rename_table_columns(pa.table({n: pa.array([], type=pf.schema_arrow.field(n).type) for n in schema_names}), schema_names)

            writer = None
            rg_iter = tqdm(range(pf.num_row_groups), desc="RowGroups", disable=not show_progress)
            for rg_idx in rg_iter:
                rg_tbl = pf.read_row_group(rg_idx, use_threads=True)
                n_in = pf.metadata.row_group(rg_idx).num_rows
                total_rows_in += n_in
                if rg_tbl.num_rows == 0:
                    continue

                renamed_tbl = rename_table_columns(rg_tbl, list(rg_tbl.column_names))
                if writer is None:
                    writer = pq.ParquetWriter(
                        out_file,
                        renamed_tbl.schema,
                        compression=compression,
                        use_dictionary=True,
                        write_statistics=True
                    )
                if row_group_size:
                    n = renamed_tbl.num_rows
                    start = 0
                    while start < n:
                        end = min(start + row_group_size, n)
                        writer.write_table(renamed_tbl.slice(start, end - start))
                        start = end
                else:
                    writer.write_table(renamed_tbl)

                total_rows_out += renamed_tbl.num_rows
                total_row_groups += 1
                if show_progress:
                    rg_iter.set_postfix(rows=f"{total_rows_out:,}")
            if writer is not None:
                writer.close()
            total_files = 1

            summary = {
                "mode": "single_file",
                "output_file": out_file,
                "rows_in": total_rows_in,
                "rows_out": total_rows_out,
                "row_groups_written": total_row_groups,
                "renamed": {k: v for k, v in rename_map.items() if k in schema_names},
                "skipped": [k for k in rename_map.keys() if k not in schema_names],
                "compression": compression,
            }
        else:
            # Dataset mode
            import os
            root_abs = os.path.abspath(self.data_path)
            fragments = list(self.dataset.get_fragments())
            frag_bar = tqdm(fragments, desc="Fragments", disable=not show_progress)

            for fragment in frag_bar:
                frag_rel = fragment.path
                if os.path.isabs(frag_rel):
                    rel_path = os.path.relpath(frag_rel, root_abs)
                else:
                    rel_path = frag_rel

                if preserve_partitions:
                    out_dir = os.path.join(out_root, os.path.dirname(rel_path))
                else:
                    out_dir = out_root
                os.makedirs(out_dir, exist_ok=True)

                src_abs = frag_rel if os.path.isabs(frag_rel) else os.path.join(root_abs, frag_rel)
                try:
                    pf = pq.ParquetFile(src_abs)
                except Exception as e:
                    print(f"⚠️ Skip unreadable fragment {frag_rel}: {e}")
                    continue

                file_schema_names = list(pf.schema_arrow.names)
                # Pre-validate duplicates for this file
                _ = rename_table_columns(
                    pa.table({n: pa.array([], type=pf.schema_arrow.field(n).type) for n in file_schema_names}),
                    file_schema_names
                )

                out_file = os.path.join(out_dir, os.path.basename(rel_path))
                writer = None
                for rg_idx in range(pf.num_row_groups):
                    rg_tbl = pf.read_row_group(rg_idx, use_threads=True)
                    n_in = pf.metadata.row_group(rg_idx).num_rows
                    total_rows_in += n_in
                    if rg_tbl.num_rows == 0:
                        continue

                    renamed_tbl = rename_table_columns(rg_tbl, list(rg_tbl.column_names))
                    if writer is None:
                        writer = pq.ParquetWriter(
                            out_file,
                            renamed_tbl.schema,
                            compression=compression,
                            use_dictionary=True,
                            write_statistics=True
                        )
                    if row_group_size:
                        n = renamed_tbl.num_rows
                        start = 0
                        while start < n:
                            end = min(start + row_group_size, n)
                            writer.write_table(renamed_tbl.slice(start, end - start))
                            start = end
                    else:
                        writer.write_table(renamed_tbl)

                    total_rows_out += renamed_tbl.num_rows
                    total_row_groups += 1

                if writer is not None:
                    writer.close()
                    total_files += 1

                if show_progress:
                    frag_bar.set_postfix(files=total_files, rows=f"{total_rows_out:,}")

            summary = {
                "mode": "dataset",
                "output_root": out_root,
                "files_written": total_files,
                "rows_in": total_rows_in,
                "rows_out": total_rows_out,
                "row_groups_written": total_row_groups,
                "renamed": {k: v for k, v in rename_map.items() if k in source_cols},
                "skipped": [k for k in rename_map.keys() if k not in source_cols],
                "compression": compression,
                "preserve_partitions": preserve_partitions,
            }

        print(f"✅ rename_colmns_parquet complete | Files: {total_files} | Rows: {total_rows_out:,}")
        return summary
    
    
    def rename_columns(self, column_dict_or_all_list, all_columns=False):
        if all_columns is True:
            types = {}
            self.dataframe.columns = column_dict_or_all_list
            for p in column_dict_or_all_list:
                types[p] = str
            self.dataframe = self.get_dataframe().astype(types)
        else:
            self.get_dataframe().rename(columns=column_dict_or_all_list, inplace=True) 

    def add_column(self, column_name, column):
        y = column
        if (not isinstance(column, pd.core.series.Series or not isinstance(column, pd.core.frame.DataFrame))):
            y = np.array(column)
            y = np.reshape(y, (y.shape[0],))
            if len(self.get_index()) == 0:
                y = pd.Series(y)
            else:
                y = pd.Series(y, self.get_index())
        self.dataframe[column_name] = y
        
    def add_transformed_columns(self, dest_column_name="new_column", transformation_rule="okk*2"):
        columns_names = self.get_columns_names()
        columns_dict = {}
        operations = {'sqrt': sqrt, 
         'pow': power,
         'exp': exp,
         }
        columns_dict.update(operations)
        for column_name in columns_names:
            if column_name in transformation_rule:
                columns_dict.update({column_name: self.get_column(column_name)})
        y_transformed = eval(transformation_rule, columns_dict)
        self.dataframe[dest_column_name] = y_transformed
        
    def add_one_value_column(self, column_name, value, length=None):
        if length is not None:
            y = np.zeros(length)
            y.fill(value)
        else:
            y = np.zeros((self.get_shape()[0]))
            y.fill(value)
        self.dataframe[column_name] = y
        return self.get_dataframe()
        
    def get_dataframe(self):
        return self.dataframe

    def request(self, select, order_by=None, ascending=None):
        if order_by is not None:
            self.dataframe = self.dataframe.sort_values(order_by, ascending=ascending)
        return self.dataframe[select]

    def contains(self, column_name, regex):
        return self.get_dataframe()[column_name].str.contains(regex)

    def column_to_upper(self, column_name):
        self.set_column(column_name, self.get_column(column_name).str.upper())
        
    def column_to_celsius(self, column_name):
        self.apply_fun_to_column(column_name, lambda x: x - 273.15)
        
    def column_to_kelvin(self, column_name):
        self.apply_fun_to_column(column_name, lambda x: x + 273.15)
        
    def combine_date_and_time_columns(self, 
                                      date_column_name='date', 
                                      time_column_name='time', 
                                      new_date_time_column_name='datetime',
                                      date_time_format='%Y-%m-%d %H:%M:%S',
                                      time_column_format='standard_time',
                                      new_time_column_name='time',
                                      drop_date_and_time_columns=True):
        
        if time_column_format == 'standard_time':
            self.add_column(new_date_time_column_name, self.get_column(date_column_name).astype(str) + ' ' + self.get_column(time_column_name).astype(str))
        elif time_column_format == 'military_time':
            self.military_time_to_standard_time_v1(time_column_name, new_time_column_name)
            self.add_column(new_date_time_column_name, self.get_column(date_column_name).astype(str) + ' ' + self.get_column(new_time_column_name).astype(str))
        
        self.column_to_date(new_date_time_column_name, format=date_time_format)
        self.transform_column(new_date_time_column_name, lambda d: d + timedelta(days=1) if d.hour == 0 else d)
        
        if drop_date_and_time_columns is True:
            self.drop_columns([date_column_name, time_column_name])
        
    def military_time_to_standard_time(self, military_time_column_name, new_time_column_name='time'):
        # Pad with zeros to ensure it is four digits
        self.transform_column(military_time_column_name, lambda x: str(int(x)).zfill(4))
        self.transform_column(military_time_column_name, lambda x: '0000' if x == '2400' else x)
        # First two digits are the hour
        # Last two digits are the minutes
        self.add_column_based_on_function(new_time_column_name, lambda row: pd.Timestamp(f'{int(row[military_time_column_name][:2])}:{int(row[military_time_column_name][2:])}').time())
   
    # Function to format the hour correctly
    @staticmethod
    def format_hour(hour):
        hours = str(int(hour[:2])-1).zfill(2)
        minutes = hour[2:]
        return f"{hours}:{minutes}"
    
    def military_time_to_standard_time_v1(self, military_time_column_name, new_time_column_name='time'):
        # Pad with zeros to ensure it is four digits
        self.transform_column(military_time_column_name, lambda x: str(int(x)).zfill(4))
        #self.transform_column(military_time_column_name, lambda x: '0000' if x == '2400' else x)
        # First two digits are the hour
        # Last two digits are the minutes
        self.add_column_based_on_function(new_time_column_name, lambda row: DataFrame.format_hour(row[military_time_column_name]))
        
        #self.add_column_based_on_function(new_time_column_name, lambda row: pd.Timestamp(f'{int(row[military_time_column_name][:2])}:{int(row[military_time_column_name][2:])}').time())
        
    def column_to_lower(self, column):
        self.set_column(column, self.get_column(column).str.lower())

    def sub(self, column, pattern, replacement):
        self.dataframe = self.get_dataframe()[column].str.replace(pattern, replacement)
    
    def drop_column(self, column_name, output_folder=None, overwrite=False, preserve_partitions=True, show_progress=True):
        """
        Drop a given column.

        Non-parquet: drop in-memory column.
        Parquet: rewrite dataset without the column (optionally preserving folder structure).

        Parameters
        ----------
        column_name : str
            Column to remove.
        output_folder : str | None
            Destination root for rewritten parquet dataset (required for parquet).
        overwrite : bool
            Overwrite destination if exists.
        preserve_partitions : bool
            If True, replicate original subfolder (partition) structure.
        show_progress : bool
            If True, display % progress (fragments & rows).
        """
        if self.data_type != 'parquet':
            if column_name not in self.dataframe.columns:
                print(f"⚠️ Column '{column_name}' not found.")
                return self.dataframe
            self.dataframe = self.dataframe.drop(columns=[column_name])
            print(f"✅ Dropped column '{column_name}' (pandas).")
            return self.dataframe

        import os, shutil
        import pyarrow.parquet as pq
        import pyarrow.dataset as ds

        if output_folder is None:
            raise ValueError("output_folder is required for parquet column drop.")
        schema_cols = list(self.dataset.schema.names)
        if column_name not in schema_cols:
            print(f"⚠️ Column '{column_name}' not in parquet schema.")
            return getattr(self, "dataframe", None)
        keep_cols = [c for c in schema_cols if c != column_name]
        if not keep_cols:
            raise ValueError("Cannot drop the only column.")

        if os.path.exists(output_folder):
            if not overwrite:
                raise FileExistsError(f"{output_folder} exists. Set overwrite=True.")
        os.makedirs(output_folder, exist_ok=True)

        print(f"🛠 Rewriting parquet without '{column_name}'")
        print(f"➡️  Keeping {len(keep_cols)} columns")

        from tqdm import tqdm

        root_abs = os.path.abspath(self.data_path)

        # ---------- Pre-scan for total rows & row groups (for % progress) ----------
        total_rows_est = 0
        total_row_groups = 0
        fragments = list(self.dataset.get_fragments())
        for fragment in fragments:
            frag_path = fragment.path
            if not os.path.isabs(frag_path):
                frag_abs = os.path.abspath(os.path.join(root_abs, frag_path))
            else:
                frag_abs = frag_path
            try:
                pf_meta = pq.ParquetFile(frag_abs)
                total_rows_est += pf_meta.metadata.num_rows
                total_row_groups += pf_meta.metadata.num_row_groups
            except Exception:
                # fallback: read fragment to count
                try:
                    tbl = fragment.to_table(columns=keep_cols)
                    total_rows_est += tbl.num_rows
                    total_row_groups += 1
                except Exception:
                    pass

        if show_progress:
            print(f"📏 Estimated total rows: {total_rows_est:,} across {len(fragments)} files / {total_row_groups} row groups")

        written_files = 0
        total_rows_written = 0
        processed_row_groups = 0

        # Progress bars
        frag_pbar = tqdm(total=len(fragments), desc="Fragments", disable=not show_progress)
        rg_pbar = tqdm(total=total_row_groups, desc="RowGroups", leave=False, disable=not show_progress)

        try:
            for fragment in fragments:
                frag_path = fragment.path
                if not os.path.isabs(frag_path):
                    frag_abs = os.path.abspath(os.path.join(root_abs, frag_path))
                else:
                    frag_abs = frag_path

                rel_path = os.path.relpath(frag_abs, root_abs)
                if preserve_partitions:
                    rel_dir = os.path.dirname(rel_path)
                    out_dir = os.path.join(output_folder, rel_dir)
                    os.makedirs(out_dir, exist_ok=True)
                    out_file = os.path.join(out_dir, os.path.basename(rel_path))
                else:
                    out_file = os.path.join(output_folder, f"part-{written_files:05d}.parquet")

                pf = pq.ParquetFile(frag_abs)
                writer = None
                try:
                    for rg_idx in range(pf.num_row_groups):
                        rg_tbl = pf.read_row_group(rg_idx, columns=keep_cols)
                        if writer is None:
                            writer = pq.ParquetWriter(out_file,
                                                      rg_tbl.schema,
                                                      compression=pf.metadata.row_group(0).column(0).compression)
                        writer.write_table(rg_tbl)
                        total_rows_written += rg_tbl.num_rows
                        processed_row_groups += 1
                        rg_pbar.update(1)

                        if show_progress and total_rows_est > 0:
                            pct_rows = (total_rows_written / total_rows_est) * 100
                            # Inline concise status
                            rg_pbar.set_postfix(rows=f"{total_rows_written:,}/{total_rows_est:,} ({pct_rows:5.1f}%)")
                    written_files += 1
                finally:
                    if writer is not None:
                        writer.close()
                frag_pbar.update(1)
        finally:
            frag_pbar.close()
            rg_pbar.close()

        print(f"✅ Done. Files written: {written_files}, rows: {total_rows_written:,} "
              f"({(total_rows_written/total_rows_est*100 if total_rows_est else 0):.1f}% of estimate)")

        # Point to new dataset
        self.data_path = output_folder
        self.dataset = ds.dataset(self.data_path, format="parquet")

        if hasattr(self, "dataframe") and isinstance(self.dataframe, pd.DataFrame):
            if column_name in self.dataframe.columns:
                self.dataframe = self.dataframe.drop(columns=[column_name])

        return getattr(self, "dataframe", None)
        
    def index_to_column(self, column_name=None, drop_actual_index=False, **kwargs):
        self.dataframe.reset_index(drop=drop_actual_index, inplace=True, **kwargs) 
        if column_name is not None:
            self.rename_columns({'index': column_name})
        
    def drop_columns(self, columns_names_as_list):
        for p in columns_names_as_list:
            self.dataframe = self.dataframe.drop(p, axis=1)
        return self.dataframe
    
    def reorder_columns(self, new_order_as_list):
        self.dataframe.reindex_axis(new_order_as_list, axis=1)
        return self.dataframe
            
    def keep_columns(self, columns_names_as_list):
        for p in self.get_columns_names():
            if p not in columns_names_as_list:
                self.dataframe = self.dataframe.drop(p, axis=1)
        return self.dataframe
    
    
    def keep_columns_parquet(
        self,
        columns_to_keep: list[str],
        tarquet_file_path: str,
        overwrite: bool = False,
        compression: str = "zstd",
        show_progress: bool = True,
        row_group_size: int | None = None,
    ) -> dict:
        """
        Keep only the specified columns and export a single parquet file to tarquet_file_path.

        Works with:
          - in-memory pandas DataFrame
          - parquet single file
          - parquet dataset (multiple files) -> merged into one output file

        Args:
            columns_to_keep: Columns to keep (others will be dropped).
            tarquet_file_path: Output file path. If it does not end with '.parquet',
                               a file named 'kept_columns.parquet' is created inside that folder.
            overwrite: Overwrite the output file if it exists.
            compression: Parquet compression codec.
            show_progress: Show tqdm progress bars.
            row_group_size: Optional output row group size (rows). If None, writes incoming groups as-is.

        Returns:
            dict summary
        """
        import os
        import pyarrow as pa
        import pyarrow.parquet as pq
        from tqdm import tqdm

        # If not parquet, operate on in-memory DataFrame
        if self.data_type != 'parquet' or not hasattr(self, 'dataset'):
            if not hasattr(self, "dataframe") or self.dataframe is None:
                raise ValueError("No data loaded.")
            keep = [c for c in columns_to_keep if c in self.dataframe.columns]
            missing = [c for c in columns_to_keep if c not in self.dataframe.columns]
            if missing:
                print(f"⚠️ Columns not found in DataFrame (ignored): {missing}")
            if not keep:
                raise ValueError("None of the requested columns exist in the DataFrame.")
            # Normalize output path to file
            out_path = tarquet_file_path
            if not out_path.lower().endswith(".parquet"):
                os.makedirs(out_path, exist_ok=True)
                out_path = os.path.join(out_path, "kept_columns.parquet")
            if os.path.exists(out_path) and not overwrite:
                raise FileExistsError(f"{out_path} exists (overwrite=False).")
            self.dataframe[keep].to_parquet(out_path, index=False, compression=compression)
            return {
                "mode": "pandas",
                "output_file": out_path,
                "rows_out": len(self.dataframe),
                "kept_columns": keep,
                "dropped_columns": [c for c in self.dataframe.columns if c not in keep]
            }

        # -------- Parquet path --------
        import os
        is_single_file = os.path.isfile(self.data_path) and self.data_path.lower().endswith(".parquet")

        # Resolve schema and validate columns
        if is_single_file:
            pf0 = pq.ParquetFile(self.data_path)
            source_cols = list(pf0.schema_arrow.names)
        else:
            source_cols = list(self.dataset.schema.names)

        keep_cols = [c for c in columns_to_keep if c in source_cols]
        missing = [c for c in columns_to_keep if c not in source_cols]
        if missing:
            print(f"⚠️ Columns not found in source parquet (ignored): {missing}")
        if not keep_cols:
            raise ValueError("None of the requested columns exist in the parquet schema.")

        # Normalize output path to file
        out_path = tarquet_file_path
        if not out_path.lower().endswith(".parquet"):
            os.makedirs(out_path, exist_ok=True)
            out_path = os.path.join(out_path, "kept_columns.parquet")
        out_dir = os.path.dirname(out_path) or "."
        os.makedirs(out_dir, exist_ok=True)
        if os.path.exists(out_path) and not overwrite:
            raise FileExistsError(f"{out_path} exists (overwrite=False).")

        total_rows_in = 0
        total_rows_out = 0
        writer = None

        def write_table(tbl: pa.Table):
            nonlocal writer, total_rows_out
            if tbl.num_rows == 0:
                return
            if writer is None:
                writer = pq.ParquetWriter(
                    out_path,
                    tbl.schema,
                    compression=compression,
                    use_dictionary=True,
                    write_statistics=True
                )
            if row_group_size:
                n = tbl.num_rows
                start = 0
                while start < n:
                    end = min(start + row_group_size, n)
                    writer.write_table(tbl.slice(start, end - start))
                    start = end
            else:
                writer.write_table(tbl)
            total_rows_out += tbl.num_rows

        if is_single_file:
            pf = pq.ParquetFile(self.data_path)
            rg_iter = tqdm(range(pf.num_row_groups), desc="RowGroups", disable=not show_progress)
            for rg_idx in rg_iter:
                # count input rows from metadata (fast)
                total_rows_in += pf.metadata.row_group(rg_idx).num_rows
                rg_tbl = pf.read_row_group(rg_idx, columns=keep_cols, use_threads=True)
                write_table(rg_tbl)
                if show_progress:
                    rg_iter.set_postfix(rows_out=f"{total_rows_out:,}")
        else:
            # Dataset: iterate fragments & row-groups, write into one file
            root_abs = os.path.abspath(self.data_path)
            fragments = list(self.dataset.get_fragments())
            frag_bar = tqdm(fragments, desc="Fragments", disable=not show_progress)
            for fragment in frag_bar:
                frag_rel = getattr(fragment, "path", "")
                frag_abs = frag_rel if os.path.isabs(frag_rel) else os.path.join(root_abs, frag_rel)
                try:
                    pf = pq.ParquetFile(frag_abs)
                except Exception as e:
                    print(f"⚠️ Skip unreadable fragment {frag_rel}: {e}")
                    continue
                rg_iter = range(pf.num_row_groups)
                if show_progress:
                    rg_iter = tqdm(rg_iter, desc=f"RowGroups:{os.path.basename(frag_abs)}", leave=False)
                for rg_idx in rg_iter:
                    total_rows_in += pf.metadata.row_group(rg_idx).num_rows
                    rg_tbl = pf.read_row_group(rg_idx, columns=keep_cols, use_threads=True)
                    write_table(rg_tbl)
                    if show_progress:
                        if hasattr(rg_iter, "set_postfix"):
                            rg_iter.set_postfix(rows_out=f"{total_rows_out:,}")

        # If nothing was written, create an empty file with the right schema
        if writer is None:
            # Build empty table with proper types from schema
            if is_single_file:
                schema_src = pf0.schema_arrow
            else:
                schema_src = self.dataset.schema
            fields = [schema_src.field(c) for c in keep_cols]
            empty_tbl = pa.table({f.name: pa.array([], type=f.type) for f in fields})
            pq.write_table(empty_tbl, out_path, compression=compression)
        else:
            writer.close()

        print(f"✅ keep_columns_parquet complete | Rows in: {total_rows_in:,} | Rows out: {total_rows_out:,}")
        return {
            "mode": "single_file" if is_single_file else "dataset",
            "output_file": out_path,
            "rows_in": total_rows_in,
            "rows_out": total_rows_out,
            "kept_columns": keep_cols,
            "dropped_columns": [c for c in source_cols if c not in keep_cols],
            "compression": compression
        }

    def add_row(self, row_as_dict, index=None):
        if index is not None:
            row = pd.DataFrame(row_as_dict, index=[index])
            self.dataframe = pd.concat([self.dataframe.iloc[:index], row, self.dataframe.iloc[index:]]).reset_index(drop=True)
            #self.reset_index()
        else:
            self.dataframe = pd.concat([self.dataframe, pd.DataFrame([row_as_dict])], ignore_index=True)

    def pivot(self, index_columns_as_list, column_columns_as_list, column_of_values, agg_func):
        return self.get_dataframe().pivot_table(index=index_columns_as_list, columns=column_columns_as_list, values=column_of_values, aggfunc=agg_func)

    def group_by(self, column_name, agg_func='sum', in_place=False, **kwargs):
        if in_place is True:
            if agg_func == 'sum':
                self.set_dataframe(self.get_dataframe().groupby(column_name, **kwargs).sum())
            elif agg_func == 'count':
                self.set_dataframe(self.get_dataframe().groupby(column_name, **kwargs).count())
            elif agg_func == 'min':
                self.set_dataframe(self.get_dataframe().groupby(column_name, **kwargs).min())
            elif agg_func == 'max':
                self.set_dataframe(self.get_dataframe().groupby(column_name, **kwargs).max())
            elif agg_func == 'mean':
                self.set_dataframe(self.get_dataframe().groupby(column_name, **kwargs).mean())
        else:
            if agg_func == 'sum':
                return self.get_dataframe().groupby(column_name, **kwargs).sum()
            elif agg_func == 'count':
                return self.get_dataframe().groupby(column_name, **kwargs).count()
            elif agg_func == 'min':
                return self.get_dataframe().groupby(column_name, **kwargs).min()
            elif agg_func == 'max':
                return self.get_dataframe().groupby(column_name, **kwargs).max()
            elif agg_func == 'mean':
                return self.get_dataframe().groupby(column_name, **kwargs).mean()
        
    def column_to_one_hot_encoding(self, column_name, drop_original_column=True, **kwargs):
        # Perform one-hot encoding on the 'Country' column
        df_encoded = pd.get_dummies(self.get_dataframe()[column_name], dtype=int, **kwargs)

        # Concatenate the original DataFrame with the encoded columns
        df = pd.concat([self.get_dataframe(), df_encoded], axis=1, **kwargs)

        # Print the updated DataFrame
        self.dataframe = df
         # Drop the original 'Country' column
        if drop_original_column is True:
            self.drop_column(column_name)
        
        return self.dataframe
    
    def get_nan_indexes_of_column(self, column_name):
        return list(self.get_dataframe().loc[pd.isna(self.get_column(column_name)), :].index)
    
    
    @staticmethod
    def get_optimal_dask_config(max_ram_per_worker_gb=2):
        """
        Auto-configure optimal Dask worker/thread settings based on available system resources.

        Args:
            max_ram_per_worker_gb (int): Approx. RAM (in GB) to assign per worker.

        Returns:
            dict: {'n_workers': int, 'threads_per_worker': int}
        """
        total_cores = os.cpu_count()
        total_ram_gb = psutil.virtual_memory().total / 1e9

        # Estimate number of workers based on RAM
        n_workers_by_ram = int(total_ram_gb // max_ram_per_worker_gb)
        n_workers = min(total_cores, max(1, n_workers_by_ram))

        # Use at least 1 thread per worker; divide evenly
        threads_per_worker = max(1, total_cores // n_workers)

        print(f"🔧 Optimal Dask config: {n_workers} workers, {threads_per_worker} threads each")

        return {
            "n_workers": n_workers,
            "threads_per_worker": threads_per_worker
        }

    
    def encode_categorical_to_numeric(self,
                                      source_column: str,
                                      new_column_name: str = "y",
                                      mapping: dict | None = None,
                                      strategy: str = "appearance",
                                      inplace: bool = True,
                                      return_mapping: bool = True):
        """
        Encode a categorical column into integer codes in a new column (default name 'y').

        Parameters:
            source_column   : str  -> name of the categorical column to encode.
            new_column_name : str  -> destination numeric column (default 'y').
            mapping         : dict -> optional existing mapping {category: code}. If None, a new one is built.
            strategy        : 'appearance' | 'sorted'
                               - appearance: order of first occurrence
                               - sorted: alphabetical / natural sorted order
            inplace         : bool -> keep DataFrame mutated (default True).
            return_mapping  : bool -> if True return (DataFrame, mapping); else only DataFrame.

        Returns:
            DataFrame or (DataFrame, mapping)
        """
        if source_column not in self.get_columns_names():
            raise ValueError(f"Column '{source_column}' not found.")

        col = self.get_column(source_column)

        # Build mapping if not provided
        if mapping is None:
            if strategy == "sorted":
                uniques = sorted(col.dropna().unique())
            else:  # appearance
                seen = []
                for v in col:
                    if pd.isna(v):
                        continue
                    if v not in seen:
                        seen.append(v)
                uniques = seen
            mapping = {cat: i for i, cat in enumerate(uniques)}

        # Encode (NaNs -> -1)
        encoded = col.map(mapping).fillna(-1).astype(int)

        if inplace:
            self.set_column(new_column_name, encoded)
            result_df = self.get_dataframe()
        else:
            result_df = self.get_dataframe().copy()
            result_df[new_column_name] = encoded

        return (result_df, mapping) if return_mapping else result_df
    
    def missing_data_checking(self, sample_size=None, verbose=True):
        """
        Check and analyze missing values in the dataframe.
        
        Parameters:
        -----------
        sample_size : int, optional
            If provided, analyzes a random sample of this size. Useful for very large datasets.
            If None, analyzes the entire dataset with progress tracking.
        verbose : bool, optional
            Whether to print detailed statistics about missing values.
            
        Returns:
        --------
        dict
            Dictionary with missing value statistics
        """
        from tqdm import tqdm
        import numpy as np
        
        # Handle Parquet data with PyArrow for large datasets
        if self.data_type == 'parquet':
            import pyarrow as pa
            import pyarrow.compute as pc
            import pyarrow.parquet as pq
            import os
            
            try:
                # Get schema information
                schema = self.dataset.schema
                total_columns = len(schema.names)
                
                # Estimate total rows by scanning the dataset
                if verbose:
                    print(f"📊 Estimating dataset size...")
                
                # Use dataset fragments to estimate row count
                fragments = list(self.dataset.get_fragments())
                total_rows = 0
                for fragment in fragments:
                    try:
                        total_rows += fragment.count_rows()
                    except:
                        # Fallback if count_rows() fails
                        pass
                
                if verbose:
                    print(f"📊 Dataset has approximately {total_rows:,} rows across {total_columns} columns")
                    print(f"📂 Dataset located at: {self.data_path}")
                
                # Process sample of dataset for missing values analysis
                missing_stats = {}
                missing_columns = []
                
                # Determine sample size for analysis
                if sample_size is None:
                    sample_size = total_rows
                    if verbose:
                        print(f"🔍 Analyzing entire dataset ({total_rows:,} rows) with progress...")
                    
                    # Initialize null counters for all columns
                    cols = schema.names
                    null_counters = {col: 0 for col in cols}
                    total_processed = 0
                    
                    # Get dataset root path
                    root_abs = os.path.abspath(self.data_path)
                    
                    # Count row groups for progress bar
                    total_row_groups = 0
                    for fragment in fragments:
                        frag_path = fragment.path if os.path.isabs(fragment.path) else os.path.join(root_abs, fragment.path)
                        try:
                            pf = pq.ParquetFile(frag_path)
                            total_row_groups += pf.num_row_groups
                        except Exception:
                            total_row_groups += 1  # Assume at least one row group
                    
                    # Process each fragment and row group with progress bar
                    with tqdm(total=total_row_groups, desc="Processing row groups", unit="group") as pbar:
                        for fragment in fragments:
                            frag_path = fragment.path if os.path.isabs(fragment.path) else os.path.join(root_abs, fragment.path)
                            try:
                                pf = pq.ParquetFile(frag_path)
                                for rg_idx in range(pf.num_row_groups):
                                    # First try to get nulls from metadata (fast)
                                    md = pf.metadata.row_group(rg_idx)
                                    rg_rows = md.num_rows
                                    meta_success = True
                                    
                                    for i, col in enumerate(cols):
                                        try:
                                            col_md = md.column(i)
                                            stats = col_md.statistics
                                            if stats and hasattr(stats, 'null_count') and stats.null_count is not None:
                                                null_counters[col] += stats.null_count
                                            else:
                                                meta_success = False
                                                break
                                        except Exception:
                                            meta_success = False
                                            break
                                    
                                    # If metadata doesn't have nulls, read the row group
                                    if not meta_success:
                                        rg = pf.read_row_group(rg_idx)
                                        rg_rows = rg.num_rows
                                        for col in cols:
                                            if col in rg.column_names:
                                                null_counters[col] += rg.column(col).null_count
                                    
                                    total_processed += rg_rows
                                    pbar.set_postfix(rows=f"{total_processed:,}/{total_rows:,}")
                                    pbar.update(1)
                            except Exception as e:
                                # Fallback to reading whole fragment if needed
                                try:
                                    tbl = fragment.to_table()
                                    for col in cols:
                                        if col in tbl.column_names:
                                            null_counters[col] += tbl.column(col).null_count
                                    total_processed += tbl.num_rows
                                except Exception:
                                    pass
                                pbar.update(1)
                    
                    # Calculate statistics from null counters
                    for col in cols:
                        null_count = null_counters[col]
                        percent_missing = (null_count / total_processed) * 100 if total_processed > 0 else 0
                        
                        missing_stats[col] = {
                            'missing_count': null_count,
                            'percent_missing': round(percent_missing, 2),
                            'dtype': str(schema.field(col).type)
                        }
                        
                        if null_count > 0:
                            missing_columns.append(col)
                            if verbose:
                                print(f"{col} has {null_count:,} missing value(s) which represents {percent_missing:.2f}% of dataset")
                
                else:
                    # Handle sample-based analysis
                    if sample_size > total_rows:
                        sample_size = total_rows
                    if verbose:
                        print(f"🔍 Analyzing {sample_size:,} rows from dataset")
                    
                    # Get the sample
                    sample_table = self.dataset.head(sample_size)
                    
                    # Analyze each column in the sample for missing values
                    for column_name in tqdm(schema.names, desc="Analyzing columns", unit="column"):
                        try:
                            column = sample_table[column_name]
                            null_count = column.null_count
                            percent_missing = (null_count / len(column)) * 100
                            
                            missing_stats[column_name] = {
                                'missing_count': null_count,
                                'percent_missing': round(percent_missing, 2),
                                'dtype': str(column.type)
                            }
                            
                            if null_count > 0:
                                missing_columns.append(column_name)
                                if verbose:
                                    print(f"{column_name} has {null_count:,} missing value(s) which represents {percent_missing:.2f}% of sample size")
                        except Exception as e:
                            if verbose:
                                print(f"⚠️ Error analyzing column {column_name}: {e}")
                
                if verbose:
                    if missing_columns:
                        print(f"\nThe data contains missing values in {len(missing_columns)} columns: {missing_columns}")
                    else:
                        print(f"\nNo column contains missing data in the analyzed dataset")
                
                return {
                    'dataset_info': {
                        'total_rows': total_rows,
                        'total_columns': total_columns,
                        'sample_size': sample_size,
                        'processed_rows': total_processed if sample_size == total_rows else sample_size
                    },
                    'missing_columns': missing_columns,
                    'missing_stats': missing_stats,
                    'has_missing_data': len(missing_columns) > 0
                }
                
            except Exception as e:
                print(f"Error analyzing Parquet dataset: {e}")
                return {'error': str(e)}
        
        # Standard pandas approach for in-memory dataframes
        else:
            miss_data_columns = []
            missing_stats = {}
            
            # Handle sampling for large datasets
            if sample_size is not None and sample_size < self.get_shape()[0]:
                if verbose:
                    print(f"🔍 Sampling {sample_size:,} rows from dataset")
                df_sample = self.dataframe.sample(n=sample_size)
            else:
                if verbose:
                    print(f"🔍 Analyzing all {self.get_shape()[0]:,} rows")
                df_sample = self.dataframe
            
            # Process each column with a progress bar
            for c in tqdm(df_sample.columns, desc="Analyzing columns", unit="column"):
                miss = df_sample[c].isnull().sum()
                missing_data_percent = round((miss/len(df_sample))*100, 2)
                
                missing_stats[c] = {
                    'missing_count': miss,
                    'percent_missing': missing_data_percent,
                    'dtype': str(df_sample[c].dtype)
                }
                
                if any(pd.isna(df_sample[c])) is True:
                    miss_data_columns.append(c)
                    if verbose:
                        print(f"{c} has {miss:,} missing value(s) which represents {missing_data_percent}% of dataset size")
            
            if verbose:
                if len(miss_data_columns) > 0:
                    print(f'\nThe data contains missing values in {len(miss_data_columns)} columns: {miss_data_columns}')
                else:
                    print(f'\nNo column contains missing data')
            
            return {
                'dataset_info': {
                    'total_rows': self.get_shape()[0],
                    'total_columns': self.get_shape()[1],
                    'sample_size': len(df_sample)
                },
                'missing_columns': miss_data_columns,
                'missing_stats': missing_stats,
                'has_missing_data': len(miss_data_columns) > 0
            }
    
    
    
    def missing_data_checking_v1(self, sample_size=None, verbose=True):
        """
        Check and analyze missing values in the dataframe.
        
        Parameters:
        -----------
        sample_size : int, optional
            If provided, analyzes a random sample of this size. Useful for very large datasets.
        verbose : bool, optional
            Whether to print detailed statistics about missing values.
            
        Returns:
        --------
        dict
            Dictionary with missing value statistics
        """
        from tqdm import tqdm
        import numpy as np
        
        # Handle Parquet data with PyArrow for large datasets
        if self.data_type == 'parquet':
            import pyarrow as pa
            import pyarrow.compute as pc
            
            try:
                # Get schema information
                schema = self.dataset.schema
                total_columns = len(schema.names)
                
                # Estimate total rows by scanning the dataset
                if verbose:
                    print(f"📊 Estimating dataset size...")
                
                # Use dataset fragments to estimate row count
                fragments = list(self.dataset.get_fragments())
                total_rows = 0
                for fragment in fragments:
                    try:
                        total_rows += fragment.count_rows()
                    except:
                        # Fallback if count_rows() fails
                        pass
                
                if verbose:
                    print(f"📊 Dataset has approximately {total_rows:,} rows across {total_columns} columns")
                    print(f"📂 Dataset located at: {self.data_path}")
                
                # Process sample of dataset for missing values analysis
                missing_stats = {}
                missing_columns = []
                
                # Determine sample size for analysis
                if sample_size is None:
                    sample_size = total_rows
                
                elif sample_size > total_rows:
                    sample_size = min(total_rows, 100000)  # Default to 100k rows max
                    if verbose and total_rows > 100000:
                        print(f"ℹ️ Analyzing first {sample_size:,} rows (use sample_size parameter to change)")
                else:
                    if verbose:
                        print(f"🔍 Analyzing {sample_size:,} rows from dataset")
                
                # Get the sample
                # read the sample_size and show progress with tqdm
                sample_table = self.dataset.head(sample_size)
                
                # Analyze each column in the sample for missing values
                for column_name in tqdm(schema.names, desc="Analyzing columns", unit="column"):
                    try:
                        column = sample_table[column_name]
                        null_count = column.null_count
                        percent_missing = (null_count / len(column)) * 100
                        
                        missing_stats[column_name] = {
                            'missing_count': null_count,
                            'percent_missing': round(percent_missing, 2),
                            'dtype': str(column.type)
                        }
                        
                        if null_count > 0:
                            missing_columns.append(column_name)
                            if verbose:
                                print(f"{column_name} has {null_count} missing value(s) which represents {percent_missing:.2f}% of sample size")
                    except Exception as e:
                        if verbose:
                            print(f"⚠️ Error analyzing column {column_name}: {e}")
                
                if verbose:
                    if missing_columns:
                        print(f"\nThe data contains missing values in {len(missing_columns)} columns: {missing_columns}")
                    else:
                        print(f"\nNo column contains missing data in the analyzed sample")
                
                return {
                    'dataset_info': {
                        'total_rows': total_rows,
                        'total_columns': total_columns,
                        'sample_size': sample_size
                    },
                    'missing_columns': missing_columns,
                    'missing_stats': missing_stats,
                    'has_missing_data': len(missing_columns) > 0
                }
                
            except Exception as e:
                print(f"Error analyzing Parquet dataset: {e}")
                return {'error': str(e)}
        
        # Standard pandas approach for in-memory dataframes
        else:
            miss_data_columns = []
            missing_stats = {}
            
            # Handle sampling for large datasets
            if sample_size is not None and sample_size < self.get_shape()[0]:
                if verbose:
                    print(f"🔍 Sampling {sample_size:,} rows from dataset")
                df_sample = self.dataframe.sample(n=sample_size)
            else:
                df_sample = self.dataframe
            
            # Process each column with a progress bar
            for c in tqdm(df_sample.columns, desc="Analyzing columns", unit="column"):
                miss = df_sample[c].isnull().sum()
                missing_data_percent = round((miss/len(df_sample))*100, 2)
                
                missing_stats[c] = {
                    'missing_count': miss,
                    'percent_missing': missing_data_percent,
                    'dtype': str(df_sample[c].dtype)
                }
                
                if any(pd.isna(df_sample[c])) is True:
                    miss_data_columns.append(c)
                    if verbose:
                        print(f"{c} has {miss} missing value(s) which represents {missing_data_percent}% of dataset size")
            
            if verbose:
                if len(miss_data_columns) > 0:
                    print(f'\nThe data contains missing values in {len(miss_data_columns)} columns: {miss_data_columns}')
                else:
                    print(f'\nNo column contains missing data')
            
            return {
                'dataset_info': {
                    'total_rows': self.get_shape()[0],
                    'total_columns': self.get_shape()[1],
                },
                'missing_columns': miss_data_columns,
                'missing_stats': missing_stats,
                'has_missing_data': len(miss_data_columns) > 0
            }
    
    def missing_data_checking_old(self):
        miss_data_columns = []
        for c in self.dataframe.columns:
            if any(pd.isna(self.get_dataframe()[c])) is True:
                miss_data_columns.append(c)
        if len(miss_data_columns) > 0:
            print(f'The data contains missing values in columns {miss_data_columns}')
        else:
            print(f'No column contains missing data')
            
        return len(miss_data_columns) > 0
        
    def missing_data_statistics(self, column_name=None):
        if column_name is not None:
            if any(pd.isna(self.get_dataframe()[column_name])) is True:
                miss = self.dataframe[column_name].isnull().sum()
                missing_data_percent = round((miss/self.get_shape()[0])*100, 2)
                print("{} has {} missing value(s) which represents {}% of dataset size".format(column_name, miss, missing_data_percent))
            else:
                print("No missed data in column " + column_name)
        else:
            miss = []
            for c in self.dataframe.columns:
                miss_by_column = self.dataframe[c].isnull().sum()
                if miss_by_column>0:
                    missing_data_percent = round((miss_by_column/self.get_shape()[0])*100, 2)
                    print("{} has {} missing value(s) which represents {}% of dataset size".format(c, miss_by_column, missing_data_percent))
                else:
                    print("{} has NO missing value!".format(c))
                miss.append(miss_by_column)
    
    def missing_data_column_percent(self, column_name):
        return self.dataframe[column_name].isnull().sum()/self.get_shape()[0]
    
    def get_missing_data_indexes_in_column(self, column_name):
        return self.dataframe[self.dataframe[column_name].isnull()].index.tolist()

    
    def split_generate_subfiles_by_column(
        self,
        split_column: str,
        output_folder: str,
        *,
        overwrite: bool = False,
        compression: str = "zstd",
        show_progress: bool = True,
        include_nulls: bool = False,
        value_formatter: Callable | None = None,  # optional: fn(value)->str for filename
        export_type: str = "parquet",             # NEW: "parquet" or "csv"
    ) -> dict:
        """
        Split the input into multiple single files by distinct values in `split_column`.

        Output filenames:
          - Single-file parquet:           <base>_<col>=<value>.(parquet|csv)
          - Parquet dataset (per fragment) <fragment_base>_<col>=<value>.(parquet|csv)
          - In-memory pandas:              frame_<col>=<value>.(parquet|csv)

        Args:
            split_column: Column to split on.
            output_folder: Destination directory for split files.
            overwrite: Overwrite output folder if it exists.
            compression: Parquet compression codec ("zstd", "snappy", ...).
            show_progress: Show tqdm bars.
            include_nulls: If True, rows with null in split_column go to value 'NULL'.
            value_formatter: Optional custom formatter for the value part of the filename.
            export_type: "parquet" (default) or "csv".

        Returns:
            dict with summary and file->rowcount.
        """
        import os
        import re
        import pyarrow as pa
        import pyarrow.parquet as pq
        import pyarrow.compute as pc
        import pandas as pd
        from collections import defaultdict
        from tqdm import tqdm

        export_type = (export_type or "parquet").lower()
        if export_type not in ("parquet", "csv"):
            raise ValueError("export_type must be 'parquet' or 'csv'.")

        def sanitize_filename_part(s: str) -> str:
            s = re.sub(r"[\\/:*?\"<>|]+", "_", s)
            s = s.strip()
            return s or "NA"

        def value_to_name(v) -> str:
            if value_formatter:
                try:
                    return sanitize_filename_part(str(value_formatter(v)))
                except Exception:
                    return sanitize_filename_part(str(v))
            if v is None:
                return "NULL"
            if isinstance(v, float):
                return sanitize_filename_part(f"{v:.6g}")
            return sanitize_filename_part(str(v))

        def write_chunk_to_value_files(
            base_name: str,
            tbl: pa.Table,
            writers: dict,
            rows_per_file: dict,
            codec: str,
            headers_written: dict,
        ):
            # Split a table by distinct values in split_column and write each to its target file.
            if split_column not in tbl.column_names:
                raise ValueError(f"Column '{split_column}' not found.")
            col = tbl[split_column]
            if col.null_count == col.length() and not include_nulls:
                return
            uniq_vals = pc.unique(col)
            if not include_nulls:
                uniq_vals = pc.drop_null(uniq_vals)

            for i in range(len(uniq_vals)):
                val = uniq_vals[i]
                py_val = val.as_py()
                if py_val is None and not include_nulls:
                    continue
                mask = pc.equal(col, val)
                sub = tbl.filter(mask)
                if sub.num_rows == 0:
                    continue
                part = value_to_name(py_val)
                ext = ".parquet" if export_type == "parquet" else ".csv"
                out_file = os.path.join(output_folder, f"{base_name}_{split_column}={part}{ext}")
                os.makedirs(os.path.dirname(out_file) or ".", exist_ok=True)

                if export_type == "parquet":
                    w = writers.get(out_file)
                    if w is None:
                        w = pq.ParquetWriter(
                            out_file,
                            sub.schema,
                            compression=codec,
                            use_dictionary=True,
                            write_statistics=True
                        )
                        writers[out_file] = w
                    w.write_table(sub)
                    rows_per_file[out_file] += sub.num_rows
                else:
                    # CSV streaming with one header per file
                    header_written = headers_written.get(out_file, False)
                    mode = "a" if header_written else "w"
                    df_chunk = sub.to_pandas()
                    df_chunk.to_csv(out_file, mode=mode, header=not header_written, index=False)
                    headers_written[out_file] = True
                    rows_per_file[out_file] += sub.num_rows

        # ---------- In-memory pandas ----------
        if self.data_type != "parquet" or not hasattr(self, "dataset"):
            df = getattr(self, "dataframe", None)
            if df is None or df.empty:
                raise ValueError("No in-memory DataFrame loaded.")
            if split_column not in df.columns:
                raise ValueError(f"Column '{split_column}' not found in DataFrame.")
            if os.path.exists(output_folder) and not overwrite:
                raise FileExistsError(f"{output_folder} exists (overwrite=False).")
            os.makedirs(output_folder, exist_ok=True)
            base = "frame"
            rows_per_file = {}
            for val, g in df.groupby(split_column, dropna=not include_nulls):
                name = value_to_name(None if pd.isna(val) else val)
                ext = ".parquet" if export_type == "parquet" else ".csv"
                out_file = os.path.join(output_folder, f"{base}_{split_column}={name}{ext}")
                if export_type == "parquet":
                    g.to_parquet(out_file, index=False, compression=compression)
                else:
                    g.to_csv(out_file, index=False)
                rows_per_file[out_file] = len(g)
            return {
                "mode": "pandas",
                "output_root": output_folder,
                "files_written": len(rows_per_file),
                "rows_per_file": rows_per_file,
                "format": export_type,
            }

        # ---------- Parquet path ----------
        is_single_file = os.path.isfile(self.data_path) and self.data_path.lower().endswith(".parquet")

        if os.path.exists(output_folder):
            if not overwrite:
                raise FileExistsError(f"{output_folder} exists (overwrite=False).")
        os.makedirs(output_folder, exist_ok=True)

        rows_per_file = defaultdict(int)
        files_opened_total = 0

        if is_single_file:
            pf = pq.ParquetFile(self.data_path)
            if split_column not in pf.schema_arrow.names:
                raise ValueError(f"Column '{split_column}' not found in parquet file.")

            # Respect source codec if user kept default 'zstd'
            codec = compression
            try:
                src_codec = pf.metadata.row_group(0).column(0).compression
                if src_codec and compression == "zstd":
                    codec = src_codec
            except Exception:
                pass

            base = os.path.splitext(os.path.basename(self.data_path))[0]
            writers = {}           # parquet writers per out_file
            headers_written = {}   # csv header flags per out_file
            try:
                rg_iter = tqdm(range(pf.num_row_groups), desc="RowGroups", disable=not show_progress)
                for rg_idx in rg_iter:
                    tbl = pf.read_row_group(rg_idx, use_threads=True)
                    if tbl.num_rows == 0:
                        continue
                    write_chunk_to_value_files(base, tbl, writers, rows_per_file, codec, headers_written)
                files_opened_total = len(writers) if export_type == "parquet" else len(headers_written)
            finally:
                for w in writers.values():
                    try:
                        w.close()
                    except Exception:
                        pass

            return {
                "mode": "single_file",
                "output_root": output_folder,
                "files_written": len(rows_per_file),
                "rows_per_file": dict(rows_per_file),
                "compression": codec if export_type == "parquet" else None,
                "writers_opened": files_opened_total,
                "format": export_type,
            }

        # Dataset mode: split per fragment; each fragment’s base name is used
        import os
        root_abs = os.path.abspath(self.data_path)
        fragments = list(self.dataset.get_fragments())
        frag_bar = tqdm(fragments, desc="Fragments", disable=not show_progress)

        total_files_written = 0
        total_writers_opened = 0

        for fragment in frag_bar:
            frag_rel = getattr(fragment, "path", "")
            frag_abs = frag_rel if os.path.isabs(frag_rel) else os.path.join(root_abs, frag_rel)

            try:
                pf = pq.ParquetFile(frag_abs)
            except Exception as e:
                print(f"⚠️ Skip unreadable fragment {frag_rel}: {e}")
                continue

            if split_column not in pf.schema_arrow.names:
                raise ValueError(f"Column '{split_column}' not found in fragment: {frag_rel}")

            # Try to mirror source codec if default was 'zstd'
            codec = compression
            try:
                src_codec = pf.metadata.row_group(0).column(0).compression
                if src_codec and compression == "zstd":
                    codec = src_codec
            except Exception:
                pass

            base = os.path.splitext(os.path.basename(frag_abs))[0]
            writers = {}           # parquet writers per out_file
            headers_written = {}   # csv header flags per out_file
            try:
                for rg_idx in range(pf.num_row_groups):
                    tbl = pf.read_row_group(rg_idx, use_threads=True)
                    if tbl.num_rows == 0:
                        continue
                    write_chunk_to_value_files(base, tbl, writers, rows_per_file, codec, headers_written)
            finally:
                for w in writers.values():
                    try:
                        w.close()
                    except Exception:
                        pass

            total_files_written = len(set(rows_per_file.keys()))
            total_writers_opened += len(writers) if export_type == "parquet" else len(headers_written)
            if show_progress:
                frag_bar.set_postfix(files=total_files_written)

        return {
            "mode": "dataset",
            "output_root": output_folder,
            "files_written": len(rows_per_file),
            "rows_per_file": dict(rows_per_file),
            "compression": compression if export_type == "parquet" else None,
            "writers_opened": total_writers_opened,
            "format": export_type,
        }
    
    
    
    def split_by_column_parquet_old(
        self,
        split_column: str,
        output_folder: str,
        *,
        overwrite: bool = False,
        compression: str = "zstd",
        show_progress: bool = True,
        include_nulls: bool = False,
        value_formatter: Callable | None = None,  # optional: fn(value)->str for filename
        export_type: str = "parquet",  
    ) -> dict:
        """
        Split the input into multiple single Parquet files by distinct values in `split_column`.

        Output filenames:
          - Single-file parquet:           <base>_<col>=<value>.parquet
          - Parquet dataset (per fragment) <fragment_base>_<col>=<value>.parquet
          - In-memory pandas:              frame_<col>=<value>.parquet

        Args:
            split_column: Column to split on.
            output_folder: Destination directory for split files.
            overwrite: Overwrite output folder if it exists.
            compression: Parquet compression codec ("zstd", "snappy", ...).
            show_progress: Show tqdm bars.
            include_nulls: If True, rows with null in split_column go to value 'NULL'.
            value_formatter: Optional custom formatter for the value part of the filename.

        Returns:
            dict with summary and file->rowcount.
        """
        import os
        import re
        import pyarrow as pa
        import pyarrow.parquet as pq
        import pyarrow.compute as pc
        from collections import defaultdict
        from tqdm import tqdm
        
        
        export_type = (export_type or "parquet").lower()
        if export_type not in ("parquet", "csv"):
            raise ValueError("export_type must be 'parquet' or 'csv'.")
        

        def sanitize_filename_part(s: str) -> str:
            # Replace path-unfriendly chars with '_'
            s = re.sub(r"[\\/:*?\"<>|]+", "_", s)
            s = s.strip()
            return s or "NA"

        def value_to_name(v) -> str:
            if value_formatter:
                try:
                    return sanitize_filename_part(str(value_formatter(v)))
                except Exception:
                    return sanitize_filename_part(str(v))
            if v is None:
                return "NULL"
            if isinstance(v, float):
                # Compact float formatting
                return sanitize_filename_part(f"{v:.6g}")
            return sanitize_filename_part(str(v))

        def write_chunk_to_value_files(base_name: str, tbl: pa.Table, writers: dict, rows_per_file: dict, codec: str):
            # Split a table by distinct values in split_column and write each to its target file.
            if split_column not in tbl.column_names:
                raise ValueError(f"Column '{split_column}' not found.")
            col = tbl[split_column]
            if col.null_count == col.length() and not include_nulls:
                return
            # Collect distinct values in this chunk
            uniq_vals = pc.unique(col)
            if not include_nulls:
                uniq_vals = pc.drop_null(uniq_vals)
            # For each value, filter and write
            for i in range(len(uniq_vals)):
                val = uniq_vals[i]
                py_val = val.as_py()  # for filename
                if py_val is None and not include_nulls:
                    continue
                mask = pc.equal(col, val)
                sub = tbl.filter(mask)
                if sub.num_rows == 0:
                    continue
                part = value_to_name(py_val)
                out_file = os.path.join(output_folder, f"{base_name}_{split_column}={part}.parquet")
                w = writers.get(out_file)
                if w is None:
                    os.makedirs(os.path.dirname(out_file) or ".", exist_ok=True)
                    w = pq.ParquetWriter(
                        out_file,
                        sub.schema,
                        compression=codec,
                        use_dictionary=True,
                        write_statistics=True
                    )
                    writers[out_file] = w
                w.write_table(sub)
                rows_per_file[out_file] += sub.num_rows

        # ---------- In-memory pandas ----------
        if self.data_type != "parquet" or not hasattr(self, "dataset"):
            df = getattr(self, "dataframe", None)
            if df is None or df.empty:
                raise ValueError("No in-memory DataFrame loaded.")
            if split_column not in df.columns:
                raise ValueError(f"Column '{split_column}' not found in DataFrame.")
            if os.path.exists(output_folder) and not overwrite:
                raise FileExistsError(f"{output_folder} exists (overwrite=False).")
            os.makedirs(output_folder, exist_ok=True)
            base = "frame"
            rows_per_file = {}
            for val, g in df.groupby(split_column, dropna=not include_nulls):
                name = value_to_name(None if pd.isna(val) else val)
                out_file = os.path.join(output_folder, f"{base}_{split_column}={name}.parquet")
                g.to_parquet(out_file, index=False, compression=compression)
                rows_per_file[out_file] = len(g)
            return {
                "mode": "pandas",
                "output_root": output_folder,
                "files_written": len(rows_per_file),
                "rows_per_file": rows_per_file
            }

        # ---------- Parquet path ----------
        is_single_file = os.path.isfile(self.data_path) and self.data_path.lower().endswith(".parquet")

        if os.path.exists(output_folder):
            if not overwrite:
                raise FileExistsError(f"{output_folder} exists (overwrite=False).")
        os.makedirs(output_folder, exist_ok=True)

        rows_per_file = defaultdict(int)
        files_opened_total = 0

        if is_single_file:
            pf = pq.ParquetFile(self.data_path)
            if split_column not in pf.schema_arrow.names:
                raise ValueError(f"Column '{split_column}' not found in parquet file.")

            # Respect source codec if user kept default 'zstd'
            codec = compression
            try:
                src_codec = pf.metadata.row_group(0).column(0).compression
                if src_codec and compression == "zstd":
                    codec = src_codec
            except Exception:
                pass

            base = os.path.splitext(os.path.basename(self.data_path))[0]
            writers = {}
            try:
                rg_iter = tqdm(range(pf.num_row_groups), desc="RowGroups", disable=not show_progress)
                for rg_idx in rg_iter:
                    tbl = pf.read_row_group(rg_idx, use_threads=True)
                    if tbl.num_rows == 0:
                        continue
                    write_chunk_to_value_files(base, tbl, writers, rows_per_file, codec)
                files_opened_total = len(writers)
            finally:
                for w in writers.values():
                    try:
                        w.close()
                    except Exception:
                        pass

            return {
                "mode": "single_file",
                "output_root": output_folder,
                "files_written": len(rows_per_file),
                "rows_per_file": dict(rows_per_file),
                "compression": codec,
                "writers_opened": files_opened_total
            }

        # Dataset mode: split per fragment; each fragment’s base name is used
        import os
        root_abs = os.path.abspath(self.data_path)
        fragments = list(self.dataset.get_fragments())
        frag_bar = tqdm(fragments, desc="Fragments", disable=not show_progress)

        total_files_written = 0
        total_writers_opened = 0

        for fragment in frag_bar:
            frag_rel = getattr(fragment, "path", "")
            frag_abs = frag_rel if os.path.isabs(frag_rel) else os.path.join(root_abs, frag_rel)

            try:
                pf = pq.ParquetFile(frag_abs)
            except Exception as e:
                print(f"⚠️ Skip unreadable fragment {frag_rel}: {e}")
                continue

            if split_column not in pf.schema_arrow.names:
                raise ValueError(f"Column '{split_column}' not found in fragment: {frag_rel}")

            # Try to mirror source codec if default was 'zstd'
            codec = compression
            try:
                src_codec = pf.metadata.row_group(0).column(0).compression
                if src_codec and compression == "zstd":
                    codec = src_codec
            except Exception:
                pass

            base = os.path.splitext(os.path.basename(frag_abs))[0]
            writers = {}
            try:
                for rg_idx in range(pf.num_row_groups):
                    tbl = pf.read_row_group(rg_idx, use_threads=True)
                    if tbl.num_rows == 0:
                        continue
                    write_chunk_to_value_files(base, tbl, writers, rows_per_file, codec)
            finally:
                for w in writers.values():
                    try:
                        w.close()
                    except Exception:
                        pass
            total_files_written = len(set(rows_per_file.keys()))
            total_writers_opened += len(writers)
            if show_progress:
                frag_bar.set_postfix(files=total_files_written)

        return {
            "mode": "dataset",
            "output_root": output_folder,
            "files_written": len(rows_per_file),
            "rows_per_file": dict(rows_per_file),
            "compression": compression,
            "writers_opened": total_writers_opened
        }
    
    
    def missing_data_parquet(
        self,
        target_path: str,
        *,
        method: str | None = None,                 # 'constant' | 'mean' | 'median' | 'mode'
        columns: list[str] | None = None,          # columns to process; default=all columns
        fill_value: int | float | str | dict | None = None,  # scalar or per-column dict (for method='constant')
        drop_row_if_nan_in: str | list[str] | None = None,   # column(s) rows must be non-null in; rows violating are dropped
        overwrite: bool = False,
        compression: str = "zstd",
        preserve_partitions: bool = True,
        row_group_size: int | None = None,
        show_progress: bool = True,
        sample_limit: int = 2_000_000,             # for median/mode sampling
    ) -> dict:
        """
        Clean missing data in Parquet by dropping rows with nulls in given columns and/or
        filling nulls in selected columns, and write the result to target_path.

        Supports:
          - method='constant' (fill_value: scalar or {col: value})
          - method='mean'     (numeric columns only; precise global mean via 2-pass sum/count)
          - method='median'   (numeric columns only; approximate via sampling up to sample_limit)
          - method='mode'     (numeric or string; approximate via sampling up to sample_limit)
          - drop_row_if_nan_in=['colA', 'colB'] to drop rows missing in any of these columns

        Writes:
          - If source is a single file: target_path can be a .parquet file or a directory
          - If source is a dataset: target_path must be a directory; preserves partitions if requested
        """
        import os
        import numpy as np
        import pyarrow as pa
        import pyarrow.compute as pc
        import pyarrow.parquet as pq
        from tqdm import tqdm

        if self.data_type != "parquet" or not hasattr(self, "dataset"):
            raise ValueError("missing_data_parquet works only with Parquet inputs (single file or dataset).")

        # Detect single file vs dataset
        is_single_file = os.path.isfile(self.data_path) and self.data_path.lower().endswith(".parquet")

        # Resolve schema + target columns
        if is_single_file:
            pf0 = pq.ParquetFile(self.data_path)
            source_cols = list(pf0.schema_arrow.names)
            schema = pf0.schema_arrow
        else:
            schema = self.dataset.schema
            source_cols = list(schema.names)

        if columns is None:
            cols_to_fill = source_cols.copy()
        else:
            cols_to_fill = [c for c in columns if c in source_cols]
            missing = [c for c in (columns or []) if c not in source_cols]
            if missing:
                print(f"⚠️ Columns not in schema (ignored): {missing}")
        if not cols_to_fill and fill_value is not None:
            raise ValueError("None of the requested columns to fill exist in schema.")

        drop_cols = []
        if drop_row_if_nan_in is not None:
            drop_cols = drop_row_if_nan_in if isinstance(drop_row_if_nan_in, list) else [drop_row_if_nan_in]
            drop_cols = [c for c in drop_cols if c in source_cols]
            if not drop_cols:
                print("⚠️ drop_row_if_nan_in contains no valid columns; skipping row drop.")

        # Prepare output paths
        if is_single_file:
            if target_path.lower().endswith(".parquet"):
                out_file = os.path.abspath(target_path)
                out_dir = os.path.dirname(out_file) or "."
                os.makedirs(out_dir, exist_ok=True)
            else:
                os.makedirs(target_path, exist_ok=True)
                base = os.path.splitext(os.path.basename(self.data_path))[0]
                out_file = os.path.join(target_path, f"{base}_clean.parquet")
            if os.path.exists(out_file) and not overwrite:
                raise FileExistsError(f"{out_file} exists (overwrite=False).")
        else:
            out_root = os.path.abspath(target_path)
            if os.path.exists(out_root) and not overwrite:
                raise FileExistsError(f"{out_root} exists (overwrite=False).")
            os.makedirs(out_root, exist_ok=True)

        # Helpers
        def _is_numeric(field: pa.Field) -> bool:
            t = field.type
            return pa.types.is_integer(t) or pa.types.is_floating(t)

        def _replace_column(tbl: pa.Table, name: str, arr: pa.Array | pa.ChunkedArray) -> pa.Table:
            idx = tbl.column_names.index(name)
            return tbl.set_column(idx, tbl.schema.field(idx), arr)

        # First pass: compute fill values for mean/median/mode if requested
        computed_fill = {}  # col -> scalar
        if method:
            mm = method.lower()
            if mm == "constant":
                if fill_value is None:
                    raise ValueError("fill_value is required for method='constant' (scalar or {col:value}).")
                # Normalize dict
                if isinstance(fill_value, dict):
                    for c in cols_to_fill:
                        if c in fill_value:
                            computed_fill[c] = fill_value[c]
                else:
                    for c in cols_to_fill:
                        computed_fill[c] = fill_value
            elif mm == "mean":
                # Only numeric columns
                targets = [c for c in cols_to_fill if _is_numeric(schema.field(c))]
                sums = {c: 0.0 for c in targets}
                counts = {c: 0 for c in targets}

                def accumulate_means(pf: pq.ParquetFile):
                    for rg_idx in range(pf.num_row_groups):
                        rg = pf.read_row_group(rg_idx, columns=targets, use_threads=True)
                        for c in targets:
                            arr = rg[c]
                            if arr.null_count == arr.length():
                                continue
                            s = pc.sum(arr).as_py()
                            n = arr.length() - arr.null_count
                            if s is not None and n:
                                sums[c] += float(s)
                                counts[c] += int(n)

                if is_single_file:
                    accumulate_means(pf0)
                else:
                    frags = list(self.dataset.get_fragments())
                    for frag in tqdm(frags, desc="Stats pass (mean)", disable=not show_progress):
                        pf = pq.ParquetFile(frag.path)
                        accumulate_means(pf)

                for c in targets:
                    computed_fill[c] = (sums[c] / counts[c]) if counts[c] else None
            elif mm in ("median", "mode"):
                # Approximate via sampling up to sample_limit
                targets = [c for c in cols_to_fill if _is_numeric(schema.field(c)) or pa.types.is_string(schema.field(c).type)]
                samples = {c: [] for c in targets}
                collected = {c: 0 for c in targets}

                def take_samples(pf: pq.ParquetFile):
                    nonlocal samples, collected
                    for rg_idx in range(pf.num_row_groups):
                        # early stop if all reached
                        if all(collected[c] >= sample_limit for c in targets):
                            return
                        rg = pf.read_row_group(rg_idx, columns=targets, use_threads=True)
                        for c in targets:
                            if collected[c] >= sample_limit:
                                continue
                            arr = rg[c]
                            if arr.null_count == arr.length():
                                continue
                            npv = arr.to_numpy(zero_copy_only=False)
                            if npv.size == 0:
                                continue
                            if np.issubdtype(npv.dtype, np.number):
                                npv = npv[np.isfinite(npv)]
                            else:
                                npv = npv[~pd.isna(npv)]
                            if npv.size == 0:
                                continue
                            need = min(npv.size, sample_limit - collected[c])
                            samples[c].append(npv[:need])
                            collected[c] += need

                import pandas as pd  # for robust mode for non-numeric
                if is_single_file:
                    take_samples(pf0)
                else:
                    for frag in tqdm(self.dataset.get_fragments(), desc=f"Stats pass ({mm})", disable=not show_progress):
                        pf = pq.ParquetFile(frag.path)
                        take_samples(pf)

                for c in targets:
                    if not samples[c]:
                        computed_fill[c] = None
                        continue
                    vals = np.concatenate(samples[c], axis=0)
                    if mm == "median":
                        try:
                            computed_fill[c] = float(np.median(vals.astype(float))) if vals.size else None
                        except Exception:
                            # for strings, median undefined
                            computed_fill[c] = None
                    else:
                        # mode; prefer pandas for mixed types
                        s = pd.Series(vals)
                        md = s.mode()
                        computed_fill[c] = md.iloc[0] if not md.empty else None
            else:
                raise ValueError("method must be one of: 'constant', 'mean', 'median', 'mode'.")

        # Second pass: transform & write
        total_rows_in = 0
        total_rows_out = 0
        files_written = 0

        if is_single_file:
            pf = pf0
            writer = None
            rg_iter = tqdm(range(pf.num_row_groups), desc="RowGroups", disable=not show_progress)
            for rg_idx in rg_iter:
                tbl = pf.read_row_group(rg_idx, use_threads=True)
                total_rows_in += tbl.num_rows
                if tbl.num_rows == 0:
                    continue

                # Drop rows with nulls in drop_cols
                if drop_cols:
                    mask = None
                    for dc in drop_cols:
                        if dc not in tbl.column_names:
                            continue
                        m = pc.is_valid(tbl[dc])
                        mask = m if mask is None else pc.and_kleene(mask, m)
                    if mask is not None:
                        tbl = tbl.filter(mask)
                if tbl.num_rows == 0:
                    continue

                # Fill nulls in cols_to_fill (if requested)
                if method:
                    for c in cols_to_fill:
                        if c not in tbl.column_names:
                            continue
                        fill_val = computed_fill.get(c, None)
                        if fill_val is None:
                            continue
                        try:
                            scalar = pa.scalar(fill_val, type=tbl[c].type)
                        except Exception:
                            # try cast scalar to column type if needed
                            scalar = pa.scalar(fill_val)
                            try:
                                scalar = pc.cast(scalar, tbl[c].type)
                            except Exception:
                                continue
                        new_arr = pc.fill_null(tbl[c], scalar)
                        tbl = _replace_column(tbl, c, new_arr)

                if tbl.num_rows == 0:
                    continue

                if writer is None:
                    writer = pq.ParquetWriter(
                        out_file,
                        tbl.schema,
                        compression=compression,
                        use_dictionary=True,
                        write_statistics=True,
                    )
                if row_group_size:
                    n = tbl.num_rows
                    start = 0
                    while start < n:
                        end = min(start + row_group_size, n)
                        writer.write_table(tbl.slice(start, end - start))
                        start = end
                else:
                    writer.write_table(tbl)

                total_rows_out += tbl.num_rows
                rg_iter.set_postfix(rows=f"{total_rows_out:,}")

            if writer is not None:
                writer.close()
            files_written = 1

            return {
                "mode": "single_file",
                "output_file": out_file,
                "rows_in": total_rows_in,
                "rows_out": total_rows_out,
                "dropped_on": drop_cols,
                "filled_columns": [c for c, v in computed_fill.items() if v is not None],
                "compression": compression,
            }

        # Dataset mode
        root_abs = os.path.abspath(self.data_path)
        fragments = list(self.dataset.get_fragments())
        frag_bar = tqdm(fragments, desc="Fragments", disable=not show_progress)

        for fragment in frag_bar:
            frag_rel = fragment.path
            src_abs = frag_rel if os.path.isabs(frag_rel) else os.path.join(root_abs, frag_rel)

            try:
                pf = pq.ParquetFile(src_abs)
            except Exception as e:
                print(f"⚠️ Skip unreadable fragment {frag_rel}: {e}")
                continue

            # Preserve partitions
            if os.path.isabs(frag_rel):
                rel_path = os.path.relpath(frag_rel, root_abs)
            else:
                rel_path = frag_rel
            out_dir = os.path.join(out_root, os.path.dirname(rel_path)) if preserve_partitions else out_root
            os.makedirs(out_dir, exist_ok=True)
            out_file = os.path.join(out_dir, os.path.basename(rel_path))

            # Try to keep source compression if user left default 'zstd'
            frag_compression = compression
            try:
                src_codec = pf.metadata.row_group(0).column(0).compression
                if src_codec and compression == "zstd":
                    frag_compression = src_codec
            except Exception:
                pass

            writer = None
            for rg_idx in range(pf.num_row_groups):
                tbl = pf.read_row_group(rg_idx, use_threads=True)
                total_rows_in += tbl.num_rows
                if tbl.num_rows == 0:
                    continue

                # Drop rows missing in specified columns
                if drop_cols:
                    mask = None
                    for dc in drop_cols:
                        if dc not in tbl.column_names:
                            continue
                        m = pc.is_valid(tbl[dc])
                        mask = m if mask is None else pc.and_kleene(mask, m)
                    if mask is not None:
                        tbl = tbl.filter(mask)
                if tbl.num_rows == 0:
                    continue

                # Fill nulls
                if method:
                    for c in cols_to_fill:
                        if c not in tbl.column_names:
                            continue
                        fill_val = computed_fill.get(c, None)
                        if fill_val is None:
                            continue
                        try:
                            scalar = pa.scalar(fill_val, type=tbl[c].type)
                        except Exception:
                            scalar = pa.scalar(fill_val)
                            try:
                                scalar = pc.cast(scalar, tbl[c].type)
                            except Exception:
                                continue
                        new_arr = pc.fill_null(tbl[c], scalar)
                        tbl = _replace_column(tbl, c, new_arr)

                if tbl.num_rows == 0:
                    continue

                if writer is None:
                    writer = pq.ParquetWriter(
                        out_file,
                        tbl.schema,
                        compression=frag_compression,
                        use_dictionary=True,
                        write_statistics=True,
                    )
                if row_group_size:
                    n = tbl.num_rows
                    start = 0
                    while start < n:
                        end = min(start + row_group_size, n)
                        writer.write_table(tbl.slice(start, end - start))
                        start = end
                else:
                    writer.write_table(tbl)

                total_rows_out += tbl.num_rows

            if writer is not None:
                writer.close()
                files_written += 1
                if show_progress:
                    frag_bar.set_postfix(files=files_written, rows=f"{total_rows_out:,}")

        return {
            "mode": "dataset",
            "output_root": out_root,
            "files_written": files_written,
            "rows_in": total_rows_in,
            "rows_out": total_rows_out,
            "dropped_on": drop_cols,
            "filled_columns": [c for c, v in computed_fill.items() if v is not None],
            "compression": compression,
            "preserve_partitions": preserve_partitions,
        }
    
    
    def missing_data(self, drop_row_if_nan_in_column=None, filling_dict_colmn_val=None, method='ffill',
                    column_to_fill=None, date_column_name=None, verbose=True):
        """
        Handle missing data in the dataframe with multiple strategies.
        
        Parameters:
        -----------
        drop_row_if_nan_in_column : str or list, optional
            Column name(s) to check for NaN values. If 'all', drops rows with NaN in any column.
        filling_dict_colmn_val : dict, optional
            Dictionary with column names as keys and values to fill NaNs with.
        method : str, optional
            Method to fill missing values:
            - 'ffill': Forward fill (use previous value)
            - 'bfill': Backward fill (use next value)
            - 'mean': Fill with column mean
            - 'median': Fill with column median
            - 'mode': Fill with column mode
            - 'constant': Fill with a constant value (specify in filling_dict_colmn_val)
            - 'interpolate': Use pandas interpolation
        column_to_fill : str or list, optional
            Specific column(s) to apply the filling method to.
        date_column_name : str, optional
            Date column name for time-based operations.
        verbose : bool, optional
            Whether to print detailed statistics about missing value handling.
            
        Returns:
        --------
        DataFrame
            The current DataFrame instance with missing values handled
        """
        from tqdm import tqdm
        
        if verbose:
            print(f"🔍 Handling missing data using method: {method}")
        
        # For Parquet datasets, load specific columns into memory first
        if self.data_type == 'parquet' and hasattr(self, 'dataset') and self.dataset is not None:
            # If we're operating on specific columns, load only those
            columns_to_process = column_to_fill if column_to_fill else self.get_columns_names()
            
            if not isinstance(columns_to_process, list):
                columns_to_process = [columns_to_process]
                
            # Only load columns we need to process
            if date_column_name and date_column_name not in columns_to_process:
                columns_to_process.append(date_column_name)
                
            if verbose:
                print(f"Loading columns for processing: {columns_to_process}")
                
            # Load data for the columns we need to process
            self.dataframe = self.dataset.to_table(columns=columns_to_process).to_pandas()
        
        # Drop rows if specified
        if drop_row_if_nan_in_column is not None:
            original_count = len(self.dataframe)
            
            if drop_row_if_nan_in_column == 'all':
                self.dataframe = self.dataframe.dropna()
                if verbose:
                    dropped_count = original_count - len(self.dataframe)
                    print(f"Dropped {dropped_count} rows ({dropped_count/original_count:.2%}) with any missing values")
            else:
                # Handle both single column and list of columns
                drop_columns = [drop_row_if_nan_in_column] if isinstance(drop_row_if_nan_in_column, str) else drop_row_if_nan_in_column
                self.dataframe = self.dataframe.dropna(subset=drop_columns)
                if verbose:
                    dropped_count = original_count - len(self.dataframe)
                    print(f"Dropped {dropped_count} rows ({dropped_count/original_count:.2%}) with missing values in {drop_columns}")
            
            return self
            
        # Fill specific values from dictionary
        if filling_dict_colmn_val is not None:
            self.dataframe.fillna(filling_dict_colmn_val, inplace=True)
            if verbose:
                print(f"Filled missing values with specific values for columns: {list(filling_dict_colmn_val.keys())}")
            return self
        
        # Apply filling method to specified columns or all columns
        columns_to_fill = []
        if column_to_fill:
            if isinstance(column_to_fill, list):
                columns_to_fill = column_to_fill
            else:
                columns_to_fill = [column_to_fill]
        else:
            columns_to_fill = self.get_columns_names()
        
        # If date column is specified, set it as index for time-based operations
        if date_column_name and date_column_name in self.dataframe.columns:
            try:
                self.dataframe[date_column_name] = pd.to_datetime(self.dataframe[date_column_name])
                if method in ['ffill', 'bfill', 'interpolate']:
                    self.dataframe.set_index(date_column_name, inplace=True)
                    if verbose:
                        print(f"Set {date_column_name} as index for time-based imputation")
            except Exception as e:
                if verbose:
                    print(f"Warning: Could not convert {date_column_name} to datetime: {e}")
        
        # Apply the selected method
        for col in tqdm(columns_to_fill, desc="Filling missing values", disable=not verbose):
            if col not in self.dataframe.columns:
                if verbose:
                    print(f"Warning: Column {col} not found in dataframe")
                continue
                
            # Count missing values before filling
            missing_before = self.dataframe[col].isna().sum()
            if missing_before == 0:
                continue
                
            # Apply different methods
            if method == 'ffill':
                self.dataframe[col] = self.dataframe[col].fillna(method='ffill')
                # Check if there are still missing values (e.g., at the beginning of the series)
                if self.dataframe[col].isna().any():
                    self.dataframe[col] = self.dataframe[col].fillna(method='bfill')
                    
            elif method == 'bfill':
                self.dataframe[col] = self.dataframe[col].fillna(method='bfill')
                # Check if there are still missing values (e.g., at the end of the series)
                if self.dataframe[col].isna().any():
                    self.dataframe[col] = self.dataframe[col].fillna(method='ffill')
                    
            elif method == 'interpolate':
                self.dataframe[col] = self.dataframe[col].interpolate(method='linear')
                # Fill any remaining NAs at the edges
                self.dataframe[col] = self.dataframe[col].fillna(method='ffill').fillna(method='bfill')
                
            elif method == 'mean':
                col_mean = self.dataframe[col].mean()
                self.dataframe[col] = self.dataframe[col].fillna(col_mean)
                
            elif method == 'median':
                col_median = self.dataframe[col].median()
                self.dataframe[col] = self.dataframe[col].fillna(col_median)
                
            elif method == 'mode':
                col_mode = self.dataframe[col].mode()[0]
                self.dataframe[col] = self.dataframe[col].fillna(col_mode)
                
            # Count missing values after filling
            missing_after = self.dataframe[col].isna().sum()
            if verbose and missing_before > 0:
                filled_count = missing_before - missing_after
                print(f"Column {col}: Filled {filled_count}/{missing_before} missing values using {method}")
        
        # If we set the index for time-based operations, reset it
        if date_column_name and self.dataframe.index.name == date_column_name:
            self.dataframe.reset_index(inplace=True)
        
        if verbose:
            remaining_missing = self.dataframe.isna().sum().sum()
            if remaining_missing > 0:
                print(f"⚠️ {remaining_missing} missing values remain after processing")
            else:
                print("✅ All missing values have been handled")
        
        return self          
    
    
    def get_row(self, row_index, columns=None, as_dict=False):
        """
        Retrieve a single row (lazy for parquet).

        Parquet: supports only integer positional index (0-based, negatives allowed).
        """
        if self.data_type != 'parquet':
            if isinstance(row_index, int):
                row = self.get_dataframe().iloc[row_index]
            else:
                row = self.get_dataframe().loc[row_index]
            return row.to_dict() if as_dict else row

        import pyarrow.parquet as pq
        import os

        if not isinstance(row_index, int):
            raise TypeError("Parquet get_row only supports integer positional indices.")

        # Collect fragment row counts
        fragments = list(self.dataset.get_fragments())
        frag_infos = []
        total_rows = 0

        dataset_root = os.path.abspath(self.data_path)

        def candidate_paths(raw_path: str):
            # raw_path may be:
            # - relative like year=2020/part-0.parquet
            # - already absolute
            # - (in some pyarrow versions) already contains the root duplicated
            paths = []
            if os.path.isabs(raw_path):
                paths.append(raw_path)
            else:
                paths.append(os.path.join(dataset_root, raw_path))
            # If duplication happened (root appears twice), trim it
            if os.path.sep in raw_path:
                parts = raw_path.split(os.path.sep)
                # remove leading duplicated root segments
                if len(parts) > 1 and parts[0] in dataset_root.replace('\\', '/'):
                    # heuristic: if join produced root/root/...
                    while True:
                        doubled = os.path.join(dataset_root, raw_path)
                        if doubled.count(dataset_root) > 1:
                            # try removing first segment
                            parts = parts[1:]
                            raw_path2 = os.path.sep.join(parts)
                            paths.append(os.path.join(dataset_root, raw_path2))
                            raw_path = raw_path2
                        else:
                            break
            # Deduplicate
            seen = set()
            for p in paths:
                npth = os.path.normpath(p)
                if npth not in seen:
                    seen.add(npth)
                    yield npth

        for frag in fragments:
            raw_path = getattr(frag, "path", None)
            # Some filesystem / partitioning setups may not expose .path; fallback
            if raw_path is None:
                # Try physical files list (pyarrow >=12)
                try:
                    physical = frag.physical_files()
                    if physical:
                        raw_path = physical[0]
                except Exception:
                    raw_path = ""
            nrows = None
            opened = False
            last_error = None
            chosen_path = None
            for cand in candidate_paths(raw_path):
                try:
                    pf = pq.ParquetFile(cand)
                    nrows = pf.metadata.num_rows
                    opened = True
                    chosen_path = cand
                    break
                except Exception as e:
                    last_error = e
                    continue
            if not opened:
                # Fallback (may load full fragment)
                try:
                    tbl = frag.to_table(columns=columns)
                    nrows = tbl.num_rows
                    chosen_path = None  # will use tbl directly later if needed
                    pf = None
                except Exception as e2:
                    raise FileNotFoundError(
                        f"Unable to open fragment path variants for '{raw_path}'. Last error: {last_error or e2}"
                    )
            frag_infos.append((frag, chosen_path, nrows))
            total_rows += nrows

        # Negative index
        if row_index < 0:
            row_index = total_rows + row_index
        if row_index < 0 or row_index >= total_rows:
            return None

        # Locate target fragment
        cumulative = 0
        target = None
        offset_in_fragment = None
        for frag, path_used, nrows in frag_infos:
            if cumulative + nrows > row_index:
                target = (frag, path_used, nrows)
                offset_in_fragment = row_index - cumulative
                break
            cumulative += nrows
        if target is None:
            return None

        frag, frag_path, frag_nrows = target

        # Read only required row group if possible
        if frag_path is not None:
            pf = pq.ParquetFile(frag_path)
            remaining = offset_in_fragment
            selected_table = None
            for rg_idx in range(pf.num_row_groups):
                rg_num_rows = pf.metadata.row_group(rg_idx).num_rows
                if remaining < rg_num_rows:
                    selected_table = pf.read_row_group(rg_idx, columns=columns)
                    break
                remaining -= rg_num_rows
            if selected_table is None:
                return None
            pdf = selected_table.to_pandas()
            if remaining >= len(pdf):
                return None
            row_series = pdf.iloc[remaining]
        else:
            # Fallback: we already loaded full fragment earlier if we got here (avoid re-load)
            tbl = frag.to_table(columns=columns)
            pdf = tbl.to_pandas()
            row_series = pdf.iloc[offset_in_fragment]

        return row_series.to_dict() if as_dict else row_series
    
    def set_row(self, column_name, row_index, new_value):
        if isinstance(row_index, int):
            self.dataframe[column_name].iloc[row_index] = new_value
        self.dataframe[column_name].loc[row_index] = new_value
    
    def replace_column(self, column, pattern, replacement, regex=False, number_of_time=-1, case_sensetivity=False):
        self.set_column(column, self.get_column(column).str.replace(pattern, replacement, regex=regex, n=number_of_time,
                                                                    case=case_sensetivity))

    def replace_num_data(self, val, replacement):
        self.get_dataframe().replace(val, replacement, inplace=True)

    def map_function(self, func, **kwargs):
        self.dataframe = self.get_dataframe().applymap(func, **kwargs)

    def apply_fun_to_column(self, column, func, in_place=True,):
        if in_place is True:
            self.set_column(column, self.get_column(column).apply(func))
        else:
            return self.get_column(column).apply(func)
        
    def add_column_based_on_function(self, column_name, func_accepting_row):
        self.add_column(column_name, self.get_dataframe().apply(func_accepting_row, axis=1))
        
    def convert_column_type(self, column_name, new_type='float64'):
        """Convert the type of the column

        Args:
            column_name (str): Name of the column to convert
            Retruns (dataframe): New dataframe after conversion
        """
        self.set_column(column_name, self.get_column(column_name).astype(new_type))
    
    def convert_dataframe_type(self, new_type='float64', ):
        for p in self.get_columns_names():
            self.convert_column_type(p, new_type)

    def concatinate(self, dataframe, ignore_index=False, join='outer'):
        """conacatenate horizontally two dataframe

        Args:
            dataframe (dataframe): the destination dataframe 
            ignore_index (bool, optional): If True, do not use the index values along the concatenation axis. Defaults to False.
        """
        # 
        self.dataframe = pd.concat([self.get_dataframe(), dataframe], axis=1, ignore_index=ignore_index, join=join)
    
    def append_dataframe(self, dataframe):
        # append dataset contents data_sets must have the same columns names
        self.dataframe = pd.concat([self.dataframe, dataframe], ignore_index=True)
        
    def join(self, dataframe, on_column='index', how='inner'):
        if on_column == 'index':
           self.dataframe = pd.merge(self.get_dataframe(), dataframe, left_index=True, right_index=True, how=how)
        else:
            self.dataframe = pd.merge(self.dataframe, dataframe, on=on_column, how=how)
            
    def interpolate_time_series(self, column_name, freq='d', method='linear', **kwargs):
        
        start_datetime = self.dataframe.index.min()
        end_datetime = self.dataframe.index.max()
        data_temp = DataFrame(self.generate_datetime_range_dataframe(start_datetime, end_datetime, freq=freq), 'df')
        data_temp.join(self.get_dataframe(), how='left')

        # Perform linear interpolation to fill missing values
        self.dataframe = data_temp.dataframe.interpolate(method=method)

    def left_join(self, dataframe, column):
        self.dataframe = pd.merge(self.dataframe, dataframe, on=column, how='left')

    def right_join(self, dataframe, column):
        self.dataframe = pd.merge(self.dataframe, dataframe, on=column, how='right')

    def eliminate_outliers_neighbors(self, n_neighbors=20, contamination=.05):
        outliers = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
        self.dataframe['inlier'] = outliers.fit_predict(self.get_dataframe())
        self.dataframe = self.get_dataframe().loc[self.get_dataframe().inlier == 1,
                                                      self.get_dataframe().columns.tolist()]

    def get_pca(self, new_dim):
        # pca.explained_variance_ratio_ gain d'info pour chaque vecteur
        pca_model = PCA(n_components=new_dim)
        return pca_model.fit_transform(self.get_dataframe())

    def get_centre_reduite(self):
        sc = StandardScaler()
        return sc.fit_transform(X=self.get_dataframe())
    
    def copy(self):
        # Deep copy
        return copy.deepcopy(self)
    
    def column_to_standard_scale(self, column):
        sc = StandardScaler()
        columns_names = self.get_columns_names()
        dataframe_copy = self
        dataframe = DataFrame(sc.fit_transform(X=self.get_dataframe()), columns_names_as_list=columns_names, data_type='matrix')
        self.reindex_dataframe()
        dataframe_copy.set_column(column, dataframe.get_column(column))
        self.set_dataframe(dataframe_copy.get_dataframe())
    
    def s__column_to_min_max_scale(self, column):
        self.set_column(column, minmax_scale(self.get_column(column)))
        
    def column_to_min_max_scale(self, column):
        self.vectorizer = MinMaxScaler() 
        dataframe_copy = self.get_dataframe()
        self.keep_columns([column])
        self.vectorizer.fit(self.get_dataframe())
        scaled_column = self.vectorizer.transform(self.get_dataframe())
        self.set_dataframe(dataframe_copy)
        self.set_column(column, scaled_column)
        return scaled_column
    
    def get_min_max_scaled_columns(self, columns_names_as_list):
        self.vectorizer = MinMaxScaler() 
        dataframe_copy = self.get_dataframe()
        self.keep_columns(columns_names_as_list)
        self.vectorizer.fit(self.get_dataframe())
        scaled_column = self.vectorizer.transform(self.get_dataframe())
        self.set_dataframe(dataframe_copy)
        return scaled_column
    
    def k_get_min_max_scaled_dataframe(self):
        self.vectorizer = MinMaxScaler()
        self.vectorizer.fit(self.get_dataframe())
        scaled_dataframe = DataFrame(self.vectorizer.transform(self.get_dataframe()), 
                                     data_type='matrix',
                                     columns_names_as_list=self.get_columns_names())
        return scaled_dataframe.get_dataframe()
    
    def get_min_max_scaled_dataframe(self):
        self.vectorizer = MinMaxScaler()
        self.vectorizer.fit(self.get_dataframe())
        return self.vectorizer.transform(self.get_dataframe())
        
    def dataframe_to_min_max_scale(self):
        self.vectorizer = MinMaxScaler()
        self.set_dataframe(self.vectorizer.fit_transform(X=self.get_dataframe()))
        
    def get_inverse_transform(self, scaled_list):
        scaled_list = np.reshape(scaled_list, (len(scaled_list), 1))
        return self.vectorizer.inverse_transform(scaled_list)
        
    def get_last_window_for_time_serie_as_list(self, column, window_size=3):
        #print(np.reshape(self.get_column(column).iloc[-window_size:].to_numpy(), (window_size, 1)))
        #print(self.vectorizer.transform([np.array(self.get_column(column).iloc[-window_size:])]))
        return self.vectorizer.transform(np.reshape(self.get_column(column).iloc[-window_size:].to_numpy(), (window_size, 1)))

    def write_column_in_file(self, column, path='data/out.csv'):
        Lib.write_liste_in_file(path, self.get_column(column).apply(str))

    def check_duplicated_rows(self, **kwargs):
        # Count duplicated rows
        duplicate_count = self.get_dataframe().duplicated(**kwargs).sum()
        # Display the count of duplicated rows
        print("Number of duplicated rows:", duplicate_count)
        return duplicate_count

    def check_duplicated_in_column(self, column):
        return any(self.get_column(column).duplicated())

    def write_check_duplicated_column_result_in_file(self, column, path='data/latin_comments.csv'):
        Lib.write_liste_in_file(path, self.get_column(column).duplicated().apply(str))

    def write_files_grouped_by_column(self, column_index, dossier):
        for p in self.get_dataframe().values:
            Lib.write_line_in_file(dossier + str(p[0]).lower() + '.csv', p[column_index])

    
    def estimate_nbr_rows_for_size(self,
                                        target_mb: int = 512,
                                        sample_rows: int = 200_000,
                                        verbose: bool = True):
        """
        Estimate an optimal Parquet row_group_size (rows) to approximate target_mb per row group.

        Parameters
        ----------
        target_mb : int
            Desired uncompressed (in‑memory) size per row group in MB (default 512).
        sample_rows : int
            Number of rows to sample for estimating average row byte size.
        verbose : bool
            Print details.

        Returns
        -------
        dict with:
            target_mb
            sampled_rows
            avg_row_bytes
            est_uncompressed_MB_per_row_group (approx)
            recommended_row_group_rows
        """
        import math

        if self.data_type != 'parquet' or not hasattr(self, 'dataset'):
            if self.dataframe is None or self.dataframe.empty:
                raise ValueError("No data to sample.")
            df = self.dataframe.head(min(sample_rows, len(self.dataframe)))
            total_bytes = df.memory_usage(deep=True).sum()
            rows = len(df)
        else:
            import pyarrow as pa
            # Use dataset.head to pull up to sample_rows (may read multiple fragments)
            tbl = self.dataset.head(sample_rows)
            rows = tbl.num_rows
            if rows == 0:
                raise ValueError("Parquet dataset is empty.")
            # Sum physical buffers
            total_bytes = 0
            for col in tbl.itercolumns():
                # For chunked arrays, sum chunks
                if isinstance(col, pa.ChunkedArray):
                    total_bytes += sum(chunk.nbytes for chunk in col.chunks)
                else:
                    total_bytes += col.nbytes

        avg_row_bytes = total_bytes / rows
        target_bytes = target_mb * 1024 * 1024
        recommended_rows = int(target_bytes / avg_row_bytes)

        # Round to a nice multiple (e.g., nearest 1k)
        if recommended_rows > 1000:
            recommended_rows = int(round(recommended_rows / 1000) * 1000)

        est_uncompressed_MB = (recommended_rows * avg_row_bytes) / (1024 * 1024)

        result = {
            "target_mb": target_mb,
            "sampled_rows": rows,
            "avg_row_bytes": avg_row_bytes,
            "recommended_row_group_rows": recommended_rows,
            "est_uncompressed_MB_per_row_group": est_uncompressed_MB
        }
        if verbose:
            print(f"Sampled {rows} rows; avg_row_bytes ≈ {avg_row_bytes:,.1f}")
            print(f"Recommended row_group_size (rows) ≈ {recommended_rows} "
                  f"(~{est_uncompressed_MB:.1f} MB uncompressed)")
        return result
    
    
    def filter_dataframe_v1(self,
                         column_name,
                         decision_function,
                         in_place=False,
                         output_folder=None,
                         overwrite=False,
                         preserve_partitions=True,
                         columns=None,
                         batch_rows_threshold=9_000_000,
                         compression="zstd",
                         show_progress=True,
                         *args
                         ):
        """
        Filter rows by applying decision_function to column_name.

        Non-parquet: keeps original in-memory behavior.

        Parquet:
          - Streams source dataset fragment by fragment / row group by row group.
          - Applies decision_function(value, *args) per value of column_name.
          - Writes filtered rows to a new parquet dataset rooted at output_folder.
          - Preserves partition folder structure (e.g. year=YYYY/month=MM) if preserve_partitions=True.

        Parameters
        ----------
        column_name : str
            Column to evaluate.
        decision_function : callable
            Function returning True/False given (value, *args).
        in_place : bool
            If True (parquet) re-point this DataFrame instance to the filtered dataset root.
        output_folder : str
            Destination parquet root (required for parquet).
        overwrite : bool
            Overwrite output folder if exists.
        preserve_partitions : bool
            Keep original partition directory hierarchy.
        columns : list | None
            Subset of columns to retain (default: all). Filter column is auto-added.
        batch_rows_threshold : int
            Flush buffered filtered rows per partition after this many accumulated rows.
        compression : str
            Parquet compression codec.
        show_progress : bool
            Show tqdm progress bars.
        """
        # ---------- Standard pandas path ----------
        if self.data_type != 'parquet' or not hasattr(self, 'dataset'):
            if in_place:
                if len(args) == 2:
                    self.set_dataframe(
                        self.get_dataframe().loc[self.get_column(column_name)
                                                 .apply(decision_function, args=(args[0], args[1]))])
                else:
                    self.set_dataframe(
                        self.get_dataframe().loc[self.get_column(column_name)
                                                 .apply(decision_function)])
                return self.get_dataframe()
            else:
                if len(args) == 2:
                    return self.get_dataframe().loc[self.get_column(column_name)
                                                    .apply(decision_function, args=(args[0], args[1]))]
                else:
                    return self.get_dataframe().loc[self.get_column(column_name)
                                                    .apply(decision_function)]

        # ---------- Parquet path (dataset or single file) ----------
        import os, shutil, inspect, re
        import pyarrow as pa, pyarrow.parquet as pq, pyarrow.compute as pc
        from tqdm import tqdm

        if output_folder is None:
            raise ValueError("output_folder is required for parquet filtering.")
        output_folder = os.path.abspath(output_folder)

        # Detect single parquet file vs directory dataset
        is_single_file = os.path.isfile(self.data_path) and self.data_path.lower().endswith(".parquet")

        if os.path.exists(output_folder):
            if not overwrite:
                raise FileExistsError(f"{output_folder} exists (overwrite=False).")
            # For single-file mode, if output is a directory we still allow overwrite removal
            

        # SINGLE PARQUET FILE MODE
        if is_single_file:
            print("🔍 Single parquet file detected.")
            input_file = self.data_path
            pf = pq.ParquetFile(input_file)
            schema = pf.schema_arrow
            if column_name not in schema.names:
                print(f"⚠️ Column '{column_name}' not found in parquet file.")
                return {"output_file": output_folder, "rows_in": 0, "rows_out": 0, "files": []}

            # Columns to keep
            if columns is None:
                keep_cols = list(schema.names)
            else:
                keep_cols = [c for c in columns if c in schema.names]
            if column_name not in keep_cols:
                keep_cols = [column_name] + keep_cols

            # Fast predicate detection (same logic as dataset mode)
            fast_mode = False
            fast_predicate = None
            if len(args) == 0:
                try:
                    src = inspect.getsource(decision_function).strip()
                    src = re.sub(r"\s+", " ", src)
                    if src.startswith("lambda"):
                        m = re.match(r"lambda\s+([A-Za-z_]\w*)\s*:\s*(.+)", src)
                        if m:
                            var, expr = m.group(1), m.group(2)
                            pat = re.compile(rf"^{var}\s*(==|!=|>=|<=|>|<)\s*([-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)$")
                            pat_rev = re.compile(rf"^([-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)\s*(==|!=|>=|<=|>|<)\s*{var}$")
                            mv = pat.match(expr)
                            if mv:
                                op = mv.group(1); val = float(mv.group(2)); fast_mode = True
                            else:
                                mr = pat_rev.match(expr)
                                if mr:
                                    rev = {"<": ">", ">": "<", "<=": ">=", ">=": "<=", "==": "==", "!=": "!="}
                                    op = rev[mr.group(3)]; val = float(mr.group(1)); fast_mode = True
                            if fast_mode:
                                fn_map = {
                                    ">": pc.greater,
                                    "<": pc.less,
                                    ">=": pc.greater_equal,
                                    "<=": pc.less_equal,
                                    "==": pc.equal,
                                    "!=": pc.not_equal
                                }
                                fast_predicate = (fn_map[op], pa.scalar(val))
                                print(f"⚡ Fast predicate (single file): {column_name} {op} {val}")
                except Exception:
                    pass
            if not fast_mode:
                print("ℹ️  Single file: row-group loop (no simple predicate recognized).")

            total_in = 0
            total_out = 0
            writer = None
            row_groups_written = 0

            pbar = tqdm(range(pf.num_row_groups), desc="RowGroups", disable=not show_progress)
            for rg_idx in pbar:
                rg_tbl = pf.read_row_group(rg_idx, columns=keep_cols, use_threads=True)
                nrg = rg_tbl.num_rows
                total_in += nrg
                if nrg == 0:
                    continue

                if fast_mode:
                    col = rg_tbl[column_name]
                    fn, cst = fast_predicate
                    mask = fn(col, cst)
                    if mask.null_count:
                        mask = pc.fill_null(mask, False)
                    if not pc.any(mask).as_py():
                        continue
                    filtered = rg_tbl.filter(mask)
                else:
                    col_arr = rg_tbl[column_name].to_numpy(zero_copy_only=False)
                    import numpy as np
                    if len(args) == 2:
                        mask_np = np.fromiter((bool(decision_function(v, args[0], args[1])) for v in col_arr),
                                              dtype=bool, count=col_arr.shape[0])
                    elif len(args) == 1:
                        mask_np = np.fromiter((bool(decision_function(v, args[0])) for v in col_arr),
                                              dtype=bool, count=col_arr.shape[0])
                    else:
                        # try vectorized
                        try:
                            vec = decision_function(col_arr)
                            mask_np = np.asarray(vec, dtype=bool)
                            if mask_np.shape[0] != col_arr.shape[0]:
                                raise ValueError
                        except Exception:
                            mask_np = np.fromiter((bool(decision_function(v)) for v in col_arr),
                                                  dtype=bool, count=col_arr.shape[0])
                    if not mask_np.any():
                        continue
                    filtered = rg_tbl.filter(pa.array(mask_np))

                kept = filtered.num_rows
                if kept == 0:
                    continue
                total_out += kept

                if writer is None:
                    # If output_folder points to a directory, create file inside; else treat as file path
                    if output_folder.lower().endswith(".parquet"):
                        out_file = output_folder
                        out_dir = os.path.dirname(out_file) or "."
                        os.makedirs(out_dir, exist_ok=True)
                    else:
                        # output_folder is a directory path
                        os.makedirs(output_folder, exist_ok=True)
                        out_file = os.path.join(output_folder, "filtered.parquet")
                    writer = pq.ParquetWriter(out_file,
                                              filtered.schema,
                                              compression=compression,
                                              use_dictionary=False,
                                              write_statistics=True)
                    out_path_final = out_file

                writer.write_table(filtered)
                row_groups_written += 1
                if show_progress and total_in:
                    pbar.set_postfix(out_rows=f"{total_out:,}")

            if writer is not None:
                writer.close()
            else:
                # No rows matched -> write empty file
                if output_folder.lower().endswith(".parquet"):
                    out_path_final = output_folder
                else:
                    os.makedirs(output_folder, exist_ok=True)
                    out_path_final = os.path.join(output_folder, "filtered_empty.parquet")
                empty_tbl = pa.table({c: pa.array([], type=schema.field(c).type) for c in keep_cols})
                pq.write_table(empty_tbl, out_path_final, compression=compression)
                print("⚠️ No rows matched; wrote empty parquet.")

            print(f"✅ Single file filter complete | In rows: {total_in:,} | Out rows: {total_out:,} | Row groups written: {row_groups_written}")

            if in_place:
                self.data_path = out_path_final
                # Rebuild dataset wrapper for consistency
                try:
                    self.dataset = ds.dataset(self.data_path, format="parquet")
                except Exception:
                    pass

            return {
                "output_file": out_path_final,
                "rows_in": total_in,
                "rows_out": total_out,
                "fast_mode": fast_mode,
                "files": [out_path_final]
            }

        # -------- ORIGINAL MULTI-FILE DATASET MODE (unchanged below, continues with fragments logic) --------
        schema = self.dataset.schema
        if column_name not in schema.names:
            print(f"⚠️ Column '{column_name}' not found in parquet schema.")
            return {"output_folder": output_folder, "files": [], "rows_in": 0, "rows_out": 0}

        # Columns to keep
        if columns is None:
            keep_cols = list(schema.names)
        else:
            keep_cols = [c for c in columns if c in schema.names]
        if column_name not in keep_cols:
            keep_cols = [column_name] + keep_cols


        # Attempt simple predicate detection
        fast_mode = False
        fast_predicate = None
        if len(args) == 0:
            try:
                src = inspect.getsource(decision_function).strip()
                src = re.sub(r"\s+", " ", src)
                if src.startswith("lambda"):
                    m = re.match(r"lambda\s+([A-Za-z_]\w*)\s*:\s*(.+)", src)
                    if m:
                        var, expr = m.group(1), m.group(2)
                        pat = re.compile(rf"^{var}\s*(==|!=|>=|<=|>|<)\s*([-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)$")
                        pat_rev = re.compile(rf"^([-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)\s*(==|!=|>=|<=|>|<)\s*{var}$")
                        mv = pat.match(expr)
                        if mv:
                            op = mv.group(1); val = float(mv.group(2))
                            fast_mode = True
                        else:
                            mr = pat_rev.match(expr)
                            if mr:
                                rev = {"<": ">", ">": "<", "<=": ">=", ">=": "<=", "==": "==", "!=": "!="}
                                op = rev[mr.group(3)]; val = float(mr.group(1)); fast_mode = True
                        if fast_mode:
                            fn_map = {
                                ">": pc.greater,
                                "<": pc.less,
                                ">=": pc.greater_equal,
                                "<=": pc.less_equal,
                                "==": pc.equal,
                                "!=": pc.not_equal
                            }
                            fast_predicate = (fn_map[op], pa.scalar(val))
                            print(f"⚡ Fast predicate: {column_name} {op} {val}")
            except Exception:
                pass
        if not fast_mode:
            print("ℹ️  Falling back to row-group loop (no simple predicate recognized).")

        fragments = list(self.dataset.get_fragments())
        root_abs = os.path.abspath(self.data_path)
        frag_pbar = tqdm(fragments, desc="Fragments", disable=not show_progress)

        total_rows_in = 0
        total_rows_out = 0
        files_written = 0
        buffers = {}
        written_files = []

        def _flush_partition(part_key):
            nonlocal files_written
            tbls = buffers.get(part_key, [])
            if not tbls:
                return
            big = pa.concat_tables(tbls)
            out_dir = os.path.join(output_folder, part_key) if part_key else output_folder
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"part-{files_written:05d}.parquet")
            pq.write_table(big, out_path, compression=compression)
            written_files.append(out_path)
            files_written += 1
            buffers[part_key] = []

        for fragment in frag_pbar:
            frag_rel = fragment.path
            
            # Force relative path for partition preservation so output_folder is honored
            if os.path.isabs(frag_rel):
                rel_path = os.path.relpath(frag_rel, root_abs)
            else:
                rel_path = frag_rel

            partition_key = os.path.dirname(rel_path) if preserve_partitions else ""

            frag_abs = frag_rel if os.path.isabs(frag_rel) else os.path.join(root_abs, frag_rel)
            try:
                pf = pq.ParquetFile(frag_abs)
            except Exception as e:
                print(f"⚠️ Could not open {frag_rel}: {e}")
                continue
            
            
            for rg_idx in range(pf.num_row_groups):
                rg_tbl = pf.read_row_group(rg_idx, columns=keep_cols, use_threads=True)
                nrg = rg_tbl.num_rows
                total_rows_in += nrg
                if nrg == 0:
                    continue

                if fast_mode:
                    col = rg_tbl[column_name]
                    fn, cst = fast_predicate
                    mask = fn(col, cst)
                    if mask.null_count:
                        mask = pc.fill_null(mask, False)
                    if not pc.any(mask).as_py():
                        continue
                    filtered = rg_tbl.filter(mask)
                else:
                    col_arr = rg_tbl[column_name].to_numpy(zero_copy_only=False)
                    import numpy as np
                    if len(args) == 2:
                        mask_np = np.fromiter((bool(decision_function(v, args[0], args[1])) for v in col_arr),
                                              dtype=bool, count=col_arr.shape[0])
                    elif len(args) == 1:
                        mask_np = np.fromiter((bool(decision_function(v, args[0])) for v in col_arr),
                                              dtype=bool, count=col_arr.shape[0])
                    else:
                        # try vectorized
                        try:
                            vec = decision_function(col_arr)
                            mask_np = np.asarray(vec, dtype=bool)
                            if mask_np.shape[0] != col_arr.shape[0]:
                                raise ValueError
                        except Exception:
                            mask_np = np.fromiter((bool(decision_function(v)) for v in col_arr),
                                                  dtype=bool, count=col_arr.shape[0])
                    if not mask_np.any():
                        continue
                    filtered = rg_tbl.filter(pa.array(mask_np))

                kept = filtered.num_rows
                if kept == 0:
                    continue
                total_rows_out += kept
                buffers.setdefault(partition_key, []).append(filtered)
                if sum(t.num_rows for t in buffers[partition_key]) >= batch_rows_threshold:
                    _flush_partition(partition_key)

        for pk in list(buffers.keys()):
            _flush_partition(pk)

        if total_rows_out == 0:
            empty_tbl = pa.table({c: pa.array([], type=schema.field(c).type) for c in keep_cols})
            empty_path = os.path.join(output_folder, "empty.parquet")
            pq.write_table(empty_tbl, empty_path, compression=compression)
            written_files.append(empty_path)
            print("⚠️ No rows matched; wrote empty placeholder.")

        print(f"✅ Filter complete | In rows: {total_rows_in:,} | Out rows: {total_rows_out:,} "
              f"| Files: {len(written_files)} | Fast mode: {fast_mode}")
        return {
            "output_folder": output_folder,
            "files": written_files,
            "rows_in": total_rows_in,
            "rows_out": total_rows_out,
            "fast_mode": fast_mode
        }
    
    
    # Add this method to DataFrame class to extract original settings
    def get_parquet_metadata(self):
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(self.data_path)
        metadata = pf.metadata
        first_rg = metadata.row_group(0)
        compression = first_rg.column(0).compression
        return {
            'compression': compression,
            'row_group_size_bytes': metadata.row_group(0).total_byte_size,
            'rows_per_group': metadata.row_group(0).num_rows
        }   
    
    
    
    def show_parquet_metadata(
        self,
        detailed: bool = False,
        include_stats: bool = False,
        max_files: int = 25,
    ) -> dict:
        """
        Show metadata for Parquet input (single file or folder dataset).

        Args:
            detailed: If True, include per-file details (up to max_files).
            include_stats: If True, aggregate min/max/null_count per column from metadata.
            max_files: Max files to inspect when dataset has many files.

        Returns:
            dict summary of metadata.
        """
        if self.data_type != "parquet" or not hasattr(self, "dataset"):
            print("Not a Parquet dataset. Nothing to show.")
            return {}

        import os
        import pyarrow.parquet as pq
        import pyarrow as pa
        import pyarrow.dataset as ds
        from collections import Counter, defaultdict

        root_path = os.path.abspath(self.data_path)
        is_single_file = os.path.isfile(root_path) and root_path.lower().endswith(".parquet")

        def col_types_from_schema(schema: pa.Schema):
            return {f.name: str(f.type) for f in schema}

        summary = {
            "type": "single_file" if is_single_file else "dataset",
            "path": root_path,
            "files": 0,
            "total_rows": 0,
            "total_row_groups": 0,
            "columns": [],
            "dtypes": {},
            "compression_codecs": [],
            "created_by": [],
            "key_value_metadata": {},
        }
        per_file = []
        codec_counter = Counter()
        created_by_set = set()

        def file_info(parquet_path: str):
            info = {
                "path": parquet_path,
                "rows": 0,
                "row_groups": 0,
                "rows_per_row_group": 0,
                "codecs": {},
                "created_by": None,
            }
            try:
                pf = pq.ParquetFile(parquet_path)
                md = pf.metadata
                info["rows"] = md.num_rows
                info["row_groups"] = md.num_row_groups
                info["rows_per_row_group"] = (md.num_rows // md.num_row_groups) if md.num_row_groups else 0
                # Collect codecs across row-groups/columns
                c_counter = Counter()
                for rg in range(md.num_row_groups):
                    rg_md = md.row_group(rg)
                    for c in range(rg_md.num_columns):
                        try:
                            c_counter[rg_md.column(c).compression] += 1
                        except Exception:
                            pass
                info["codecs"] = dict(c_counter)
                info["created_by"] = md.created_by
                # KV metadata (bytes->bytes); only add from first file
                if not summary["key_value_metadata"] and md.metadata is not None:
                    # Convert to str
                    kv = {}
                    for k, v in md.metadata.items():
                        try:
                            kv[k.decode("utf-8", errors="ignore")] = v.decode("utf-8", errors="ignore")
                        except Exception:
                            kv[str(k)] = str(v)
                    summary["key_value_metadata"] = kv
                return info, pf
            except Exception as e:
                print(f"⚠️ Could not read parquet metadata: {parquet_path} ({e})")
                return info, None

        def aggregate_column_stats(pf: pq.ParquetFile, agg_store: dict):
            md = pf.metadata
            for rg in range(md.num_row_groups):
                rg_md = md.row_group(rg)
                for c in range(rg_md.num_columns):
                    col_md = rg_md.column(c)
                    stats = getattr(col_md, "statistics", None)
                    if stats is None:
                        continue
                    name = col_md.path_in_schema
                    entry = agg_store.setdefault(name, {"min": None, "max": None, "null_count": 0})
                    # Only numeric/ordered types have min/max in metadata
                    try:
                        # stats.min and stats.max may be bytes; rely on Arrow to format
                        vmin = stats.min
                        vmax = stats.max
                        if vmin is not None:
                            if entry["min"] is None or vmin < entry["min"]:
                                entry["min"] = vmin
                        if vmax is not None:
                            if entry["max"] is None or vmax > entry["max"]:
                                entry["max"] = vmax
                        if stats.null_count is not None:
                            entry["null_count"] += int(stats.null_count)
                    except Exception:
                        pass

        if is_single_file:
            info, pf = file_info(root_path)
            summary["files"] = 1
            summary["total_rows"] = info["rows"]
            summary["total_row_groups"] = info["row_groups"]
            summary["rows_per_row_group"] = info["rows_per_row_group"]
            summary["compression_codecs"] = sorted(list(info["codecs"].keys()))
            if pf is not None:
                schema = pf.schema_arrow
                summary["columns"] = list(schema.names)
                summary["dtypes"] = col_types_from_schema(schema)
            if info["created_by"]:
                created_by_set.add(info["created_by"])
            if detailed:
                per_file.append(info)
            if include_stats and pf is not None:
                agg_stats = {}
                aggregate_column_stats(pf, agg_stats)
                summary["column_stats"] = agg_stats
        else:
            fragments = list(self.dataset.get_fragments())
            summary["files"] = len(fragments)
            # Use dataset schema for dtypes
            schema = self.dataset.schema
            summary["columns"] = list(schema.names)
            summary["dtypes"] = col_types_from_schema(schema)

            inspected = 0
            agg_stats = {}
            for frag in fragments:
                if inspected >= max_files and not detailed and not include_stats:
                    break
                fpath = frag.path if os.path.isabs(frag.path) else os.path.join(root_path, frag.path)
                info, pf = file_info(fpath)
                summary["total_rows"] += info["rows"]
                summary["total_row_groups"] += info["row_groups"]
                codec_counter.update(info["codecs"])
                if info["created_by"]:
                    created_by_set.add(info["created_by"])
                if detailed and inspected < max_files:
                    per_file.append(info)
                if include_stats and pf is not None and inspected < max_files:
                    aggregate_column_stats(pf, agg_stats)
                inspected += 1

            summary["compression_codecs"] = sorted(list(codec_counter.keys()))
            summary["created_by"] = sorted(list(created_by_set))
            if include_stats:
                summary["column_stats"] = agg_stats
            if detailed:
                summary["per_file"] = per_file
                if summary["files"] > max_files:
                    summary["per_file_truncated"] = True

        # Pretty print
        print(f"Parquet metadata [{summary['type']}]: {summary['path']}")
        print(f"- Files: {summary['files']:,}")
        print(f"- Rows: {summary['total_rows']:,}")
        print(f"- Row groups: {summary['total_row_groups']:,}")
        print(f"- Rows per row group: {summary['rows_per_row_group']:,}")
        print(f"- Columns: {len(summary['columns'])} -> {summary['columns'][:10]}{' ...' if len(summary['columns'])>10 else ''}")
        print(f"- Codecs: {', '.join(summary['compression_codecs']) or 'n/a'}")
        if summary["created_by"]:
            print(f"- Created by: {', '.join(summary['created_by'])}")
        if summary["key_value_metadata"]:
            print(f"- Key/Value metadata: {len(summary['key_value_metadata'])} entries")
        if detailed and "per_file" in summary:
            shown = len(summary["per_file"])
            print(f"- Per-file details shown: {shown}{' (truncated)' if summary.get('per_file_truncated') else ''}")
        if include_stats and "column_stats" in summary:
            # Show a tiny preview of stats
            keys = list(summary["column_stats"].keys())[:5]
            print(f"- Column stats (preview): {keys}")

        return summary
    
    
    def filter_dataframe(self,
                     column_name,
                     decision_function,
                     inplace=True,
                     output_folder_or_file_path=None,
                     overwrite=False,
                     preserve_partitions=True,
                     columns=None,
                     batch_rows_threshold='auto',
                     compression="auto",
                     show_progress=True,
                     chunking=True,
                     row_wise: bool = False,   # <<< NEW: apply predicate on whole row instead of a single column value
                     *args
                     ):
        """
        Filter rows by applying decision_function.

        Modes:
        - Default (row_wise=False): decision_function(value, *args) receives each value from column_name
          (backward compatible original behavior, with fast predicate optimization when possible).
        - Row-wise (row_wise=True): decision_function(row) receives a full row (pandas Series for in-memory,
          pandas DataFrame row for parquet chunks). Must return True/False per row.
          Fast predicate detection is disabled in this mode.

        When parquet:
        - Streams fragment/row-group.
        - If row_wise=True row group is converted to a pandas DataFrame slice for masking.
          (For very large row groups this is still memory-friendly vs loading the whole dataset.)

        Args:
            column_name : str
                Kept for backward compatibility; ignored for predicate evaluation when row_wise=True
                (still auto-added to output columns list if not present).
            decision_function : callable
                Column mode: fn(value, *args) -> bool  OR vectorized fn(np.ndarray) -> bool mask.
                Row mode: fn(row) -> bool  OR vectorized fn(DataFrame) -> 1D bool array/Series.
            row_wise : bool
                If True evaluate predicate on whole rows.

        Other parameters unchanged.
        """
        # ---------- Standard pandas path ----------
        if self.data_type != 'parquet' or not hasattr(self, 'dataset'):
            if row_wise:
                df = self.get_dataframe()
                # Try vectorized first (function returns boolean array/Series of same length)
                try:
                    vec_res = decision_function(df)
                    import numpy as np, pandas as pd
                    if isinstance(vec_res, (pd.Series, np.ndarray)) and len(vec_res) == len(df):
                        mask = np.asarray(vec_res, dtype=bool)
                    else:
                        raise ValueError
                except Exception:
                    mask = df.apply(lambda r: decision_function(r), axis=1)
                if inplace:
                    self.set_dataframe(self.get_dataframe().loc[mask])
                    return self.get_dataframe()
                return self.get_dataframe().loc[mask]

            # original column-based logic kept
            if inplace:
                if len(args) == 2:
                    self.set_dataframe(
                        self.get_dataframe().loc[self.get_column(column_name)
                                                 .apply(decision_function, args=(args[0], args[1]))])
                elif len(args) == 1:
                    self.set_dataframe(
                        self.get_dataframe().loc[self.get_column(column_name)
                                                 .apply(decision_function, args=(args[0],))])
                else:
                    # vectorized attempt
                    try:
                        vec = decision_function(self.get_column(column_name).to_numpy())
                        import numpy as np
                        vec_bool = np.asarray(vec, dtype=bool)
                        if vec_bool.shape[0] != self.get_shape()[0]:
                            raise ValueError
                        self.set_dataframe(self.get_dataframe().loc[vec_bool])
                    except Exception:
                        self.set_dataframe(
                            self.get_dataframe().loc[self.get_column(column_name)
                                                     .apply(decision_function)])
                return self.get_dataframe()
            else:
                if len(args) == 2:
                    return self.get_dataframe().loc[self.get_column(column_name)
                                                    .apply(decision_function, args=(args[0], args[1]))]
                elif len(args) == 1:
                    return self.get_dataframe().loc[self.get_column(column_name)
                                                    .apply(decision_function, args=(args[0],))]
                else:
                    # vectorized attempt
                    try:
                        vec = decision_function(self.get_column(column_name).to_numpy())
                        import numpy as np
                        vec_bool = np.asarray(vec, dtype=bool)
                        if vec_bool.shape[0] != self.get_shape()[0]:
                            raise ValueError
                        return self.get_dataframe().loc[vec_bool]
                    except Exception:
                        return self.get_dataframe().loc[self.get_column(column_name)
                                                        .apply(decision_function)]

        # ---------- Parquet path ----------
        import os, inspect, re
        import pyarrow as pa, pyarrow.parquet as pq, pyarrow.compute as pc
        from tqdm import tqdm
        import numpy as np
        import pandas as pd

        if output_folder_or_file_path is None:
            raise ValueError("output_folder is required for parquet filtering.")
        output_folder = os.path.abspath(output_folder_or_file_path)

        is_single_file = os.path.isfile(self.data_path) and self.data_path.lower().endswith(".parquet")

        try:
            original_metadata = self.get_parquet_metadata()
            if compression == 'auto':
                compression = original_metadata['compression']
                print(f"🔄 Using source compression codec: {compression}")
            if batch_rows_threshold == 'auto':
                batch_rows_threshold = original_metadata['rows_per_group']
                print(f"🔄 Using source rows per group: {batch_rows_threshold:,}")
        except Exception:
            if compression == 'auto':
                compression = "zstd"
            if batch_rows_threshold == 'auto':
                batch_rows_threshold = 9_000_000
        
        # Normalize desired output row group size (None means "don't enforce")
        row_group_size = None
        try:
            b = int(batch_rows_threshold)
            if b > 0:
                row_group_size = b
        except Exception:
            row_group_size = None

        if os.path.exists(output_folder_or_file_path):
            if not overwrite:
                raise FileExistsError(f"{output_folder_or_file_path} exists (overwrite=False).")

        # Helper: evaluate row-wise mask for a pandas DataFrame chunk
        def apply_row_wise(df_chunk: pd.DataFrame):
            # Try vectorized first
            try:
                res = decision_function(df_chunk)
                if isinstance(res, (pd.Series, np.ndarray)) and len(res) == len(df_chunk):
                    return np.asarray(res, dtype=bool)
            except Exception:
                pass
            # Fallback: per-row apply
            return df_chunk.apply(lambda r: decision_function(r), axis=1).to_numpy(dtype=bool)

        # FAST predicate detection only if NOT row_wise
        def detect_fast(decision_function):
            if row_wise:
                return False, None
            try:
                src = inspect.getsource(decision_function).strip()
                src = re.sub(r"\s+", " ", src)
                if src.startswith("lambda"):
                    m = re.match(r"lambda\s+([A-Za-z_]\w*)\s*:\s*(.+)", src)
                    if m:
                        var, expr = m.group(1), m.group(2)
                        pat = re.compile(rf"^{var}\s*(==|!=|>=|<=|>|<)\s*([-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)$")
                        pat_rev = re.compile(rf"^([-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)\s*(==|!=|>=|<=|>|<)\s*{var}$")
                        mv = pat.match(expr)
                        if mv:
                            op = mv.group(1); val = float(mv.group(2))
                        else:
                            mr = pat_rev.match(expr)
                            if not mr:
                                return False, None
                            rev = {"<": ">", ">": "<", "<=": ">=", ">=": "<=", "==": "==", "!=": "!="}
                            op = rev[mr.group(3)]; val = float(mr.group(1))
                        fn_map = {
                            ">": pc.greater,
                            "<": pc.less,
                            ">=": pc.greater_equal,
                            "<=": pc.less_equal,
                            "==": pc.equal,
                            "!=": pc.not_equal
                        }
                        return True, (fn_map[op], pa.scalar(val))
            except Exception:
                return False, None
            return False, None

        # ---------- SINGLE FILE MODE ----------
        if is_single_file:
            output_is_file = output_folder_or_file_path.lower().endswith('.parquet')
            if not chunking and not output_is_file:
                os.makedirs(output_folder_or_file_path, exist_ok=True)
                output_file = os.path.join(output_folder_or_file_path, "filtered.parquet")
            elif output_is_file:
                os.makedirs(os.path.dirname(output_folder_or_file_path) or ".", exist_ok=True)
                output_file = output_folder_or_file_path
            else:
                os.makedirs(output_folder_or_file_path, exist_ok=True)
                output_file = os.path.join(output_folder_or_file_path, "filtered.parquet")

            pf = pq.ParquetFile(self.data_path)
            schema = pf.schema_arrow

            # Columns to keep
            if columns is None:
                keep_cols = list(schema.names)
            else:
                keep_cols = [c for c in columns if c in schema.names]
            if column_name not in keep_cols:
                keep_cols = [column_name] + keep_cols if column_name in schema.names else keep_cols

            fast_mode, fast_predicate = detect_fast(decision_function)
            if fast_mode:
                print(f"⚡ Fast predicate (single file): {column_name}")
            elif not row_wise:
                print("ℹ️  Single file: row-group loop (no simple predicate recognized).")
            else:
                print("ℹ️  Row-wise evaluation (no fast predicate).")

            total_in = 0
            total_out = 0
            writer = None
            row_groups_written = 0

            pbar = tqdm(range(pf.num_row_groups), desc="RowGroups", disable=not show_progress)
            for rg_idx in pbar:
                rg_tbl = pf.read_row_group(rg_idx, columns=keep_cols, use_threads=True)
                nrg = rg_tbl.num_rows
                total_in += nrg
                if nrg == 0:
                    continue

                if row_wise:
                    pdf = rg_tbl.to_pandas()
                    mask_np = apply_row_wise(pdf)
                    if not mask_np.any():
                        continue
                    filtered = rg_tbl.filter(pa.array(mask_np.tolist()))
                else:
                    if fast_mode:
                        col = rg_tbl[column_name]
                        fn, cst = fast_predicate
                        mask = fn(col, cst)
                        if mask.null_count:
                            mask = pc.fill_null(mask, False)
                        if not pc.any(mask).as_py():
                            continue
                        filtered = rg_tbl.filter(mask)
                    else:
                        col_arr = rg_tbl[column_name].to_numpy(zero_copy_only=False)
                        if len(args) == 2:
                            mask_np = np.fromiter((bool(decision_function(v, args[0], args[1])) for v in col_arr),
                                                  dtype=bool, count=col_arr.shape[0])
                        elif len(args) == 1:
                            mask_np = np.fromiter((bool(decision_function(v, args[0])) for v in col_arr),
                                                  dtype=bool, count=col_arr.shape[0])
                        else:
                            # vectorized try
                            try:
                                vec = decision_function(col_arr)
                                mask_np = np.asarray(vec, dtype=bool)
                                if mask_np.shape[0] != col_arr.shape[0]:
                                    raise ValueError
                            except Exception:
                                mask_np = np.fromiter((bool(decision_function(v)) for v in col_arr),
                                                      dtype=bool, count=col_arr.shape[0])
                        if not mask_np.any():
                            continue
                        filtered = rg_tbl.filter(pa.array(mask_np))

                kept = filtered.num_rows
                if kept == 0:
                    continue
                if writer is None:
                    writer = pq.ParquetWriter(
                        output_file,
                        filtered.schema,
                        compression=compression,
                        use_dictionary=True,
                        write_statistics=True
                    )
                # Enforce output row group size if provided
                if row_group_size:
                    n_rows = filtered.num_rows
                    start = 0
                    while start < n_rows:
                        end = min(start + row_group_size, n_rows)
                        writer.write_table(filtered.slice(start, end - start))
                        start = end
                else:
                    writer.write_table(filtered)    
                
                row_groups_written += 1
                total_out += kept
                if show_progress and total_in:
                    pbar.set_postfix(out_rows=f"{total_out:,}")

            if writer is not None:
                writer.close()
                print(f"✅ Single file filter complete | In rows: {total_in:,} | Out rows: {total_out:,}")
                print(f"   Output file: {output_file} | Row groups: {row_groups_written}")
            else:
                empty_tbl = pa.table({c: pa.array([], type=schema.field(c).type) for c in keep_cols})
                pq.write_table(empty_tbl, output_file, compression=compression)
                print("⚠️ No rows matched; wrote empty parquet.")

            if inplace:
                self.data_path = output_file
                try:
                    self.dataset = ds.dataset(self.data_path, format="parquet")
                except Exception:
                    pass

            return {
                "output_file": output_file,
                "rows_in": total_in,
                "rows_out": total_out,
                "fast_mode": fast_mode and not row_wise,
                "row_wise": row_wise,
                "files": [output_file]
            }

        # ---------- MULTI-FILE DATASET ----------
        schema = self.dataset.schema
        if column_name not in schema.names and not row_wise:
            print(f"⚠️ Column '{column_name}' not found in parquet schema.")
            return {"output_folder": output_folder_or_file_path, "files": [], "rows_in": 0, "rows_out": 0}

        os.makedirs(output_folder_or_file_path, exist_ok=True)

        if columns is None:
            keep_cols = list(schema.names)
        else:
            keep_cols = [c for c in columns if c in schema.names]
        if column_name not in keep_cols and column_name in schema.names:
            keep_cols = [column_name] + keep_cols

        fast_mode, fast_predicate = detect_fast(decision_function)
        if fast_mode:
            print(f"⚡ Fast predicate: {column_name}")
        elif row_wise:
            print("ℹ️  Row-wise evaluation (no fast predicate).")
        else:
            print("ℹ️  Falling back to row-group loop (no simple predicate recognized).")

        import pyarrow.dataset as ds
        fragments = list(self.dataset.get_fragments())
        root_abs = os.path.abspath(self.data_path)
        frag_pbar = tqdm(fragments, desc="Fragments", disable=not show_progress)

        # Non-chunking single-output-file mode
        if not chunking:
            output_file = os.path.join(output_folder_or_file_path, "filtered.parquet")
            writer = None
            total_rows_in = 0
            total_rows_out = 0

            for fragment in frag_pbar:
                frag_rel = fragment.path
                frag_abs = frag_rel if os.path.isabs(frag_rel) else os.path.join(root_abs, frag_rel)
                try:
                    pf = pq.ParquetFile(frag_abs)
                except Exception as e:
                    print(f"⚠️ Could not open {frag_rel}: {e}")
                    continue

                for rg_idx in range(pf.num_row_groups):
                    rg_tbl = pf.read_row_group(rg_idx, columns=keep_cols, use_threads=True)
                    nrg = rg_tbl.num_rows
                    total_rows_in += nrg
                    if nrg == 0:
                        continue

                    if row_wise:
                        pdf = rg_tbl.to_pandas()
                        mask_np = apply_row_wise(pdf)
                        if not mask_np.any():
                            continue
                        filtered = rg_tbl.filter(pa.array(mask_np.tolist()))
                    else:
                        if fast_mode:
                            col = rg_tbl[column_name]
                            fn, cst = fast_predicate
                            mask = fn(col, cst)
                            if mask.null_count:
                                mask = pc.fill_null(mask, False)
                            if not pc.any(mask).as_py():
                                continue
                            filtered = rg_tbl.filter(mask)
                        else:
                            col_arr = rg_tbl[column_name].to_numpy(zero_copy_only=False)
                            if len(args) == 2:
                                mask_np = np.fromiter((bool(decision_function(v, args[0], args[1])) for v in col_arr),
                                                      dtype=bool, count=col_arr.shape[0])
                            elif len(args) == 1:
                                mask_np = np.fromiter((bool(decision_function(v, args[0])) for v in col_arr),
                                                      dtype=bool, count=col_arr.shape[0])
                            else:
                                try:
                                    vec = decision_function(col_arr)
                                    mask_np = np.asarray(vec, dtype=bool)
                                    if mask_np.shape[0] != col_arr.shape[0]:
                                        raise ValueError
                                except Exception:
                                    mask_np = np.fromiter((bool(decision_function(v)) for v in col_arr),
                                                          dtype=bool, count=col_arr.shape[0])
                            if not mask_np.any():
                                continue
                            filtered = rg_tbl.filter(pa.array(mask_np))

                    kept = filtered.num_rows
                    if kept == 0:
                        continue
                    if writer is None:
                        writer = pq.ParquetWriter(
                            output_file,
                            filtered.schema,
                            compression=compression,
                            use_dictionary=True,
                            write_statistics=True
                        )
                    # Enforce output row group size if provided
                    if row_group_size:
                        n_rows = filtered.num_rows
                        start = 0
                        while start < n_rows:
                            end = min(start + row_group_size, n_rows)
                            writer.write_table(filtered.slice(start, end - start))
                            start = end
                    else:
                        writer.write_table(filtered)
                    
                    total_rows_out += kept
                    if show_progress:
                        frag_pbar.set_postfix(out_rows=f"{total_rows_out:,}")

            if writer is not None:
                writer.close()
                print(f"✅ Multi-file dataset filter (single output) | In rows: {total_rows_in:,} | Out rows: {total_rows_out:,}")
            else:
                empty_tbl = pa.table({c: pa.array([], type=schema.field(c).type) for c in keep_cols})
                pq.write_table(empty_tbl, output_file, compression=compression)
                print("⚠️ No rows matched; wrote empty parquet.")

            if inplace:
                self.data_path = output_file
                try:
                    self.dataset = ds.dataset(self.data_path, format="parquet")
                except Exception:
                    pass

            return {
                "output_file": output_file,
                "rows_in": total_in,
                "rows_out": total_out,
                "fast_mode": fast_mode and not row_wise,
                "row_wise": row_wise,
                "files": [output_file]
            }

        # Chunking mode with partition preservation
        total_rows_in = 0
        total_rows_out = 0
        files_written = 0
        buffers = {}
        written_files = []

        def _flush_partition(part_key):
            nonlocal files_written
            tbls = buffers.get(part_key, [])
            if not tbls:
                return
            big = pa.concat_tables(tbls)
            out_dir = os.path.join(output_folder_or_file_path, part_key) if part_key else output_folder_or_file_path
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"part-{files_written:05d}.parquet")
            # Use row_group_size to control row groups in the output
            pq.write_table(big, out_path, compression=compression, row_group_size=row_group_size)
            written_files.append(out_path)
            files_written += 1
            buffers[part_key] = []

        for fragment in frag_pbar:
            frag_rel = fragment.path
            rel_path = frag_rel if os.path.isabs(frag_rel) else os.path.relpath(
                os.path.join(root_abs, frag_rel), root_abs)
            partition_key = os.path.dirname(rel_path) if preserve_partitions else ""
            frag_abs = frag_rel if os.path.isabs(frag_rel) else os.path.join(root_abs, frag_rel)

            try:
                pf = pq.ParquetFile(frag_abs)
            except Exception as e:
                print(f"⚠️ Could not open {frag_rel}: {e}")
                continue

            for rg_idx in range(pf.num_row_groups):
                rg_tbl = pf.read_row_group(rg_idx, columns=keep_cols, use_threads=True)
                nrg = rg_tbl.num_rows
                total_rows_in += nrg
                if nrg == 0:
                    continue

                if row_wise:
                    pdf = rg_tbl.to_pandas()
                    mask_np = apply_row_wise(pdf)
                    if not mask_np.any():
                        continue
                    filtered = rg_tbl.filter(pa.array(mask_np.tolist()))
                else:
                    if fast_mode:
                        col = rg_tbl[column_name]
                        fn, cst = fast_predicate
                        mask = fn(col, cst)
                        if mask.null_count:
                            mask = pc.fill_null(mask, False)
                        if not pc.any(mask).as_py():
                            continue
                        filtered = rg_tbl.filter(mask)
                    else:
                        col_arr = rg_tbl[column_name].to_numpy(zero_copy_only=False)
                        if len(args) == 2:
                            mask_np = np.fromiter((bool(decision_function(v, args[0], args[1])) for v in col_arr),
                                                  dtype=bool, count=col_arr.shape[0])
                        elif len(args) == 1:
                            mask_np = np.fromiter((bool(decision_function(v, args[0])) for v in col_arr),
                                                  dtype=bool, count=col_arr.shape[0])
                        else:
                            try:
                                vec = decision_function(col_arr)
                                mask_np = np.asarray(vec, dtype=bool)
                                if mask_np.shape[0] != col_arr.shape[0]:
                                    raise ValueError
                            except Exception:
                                mask_np = np.fromiter((bool(decision_function(v)) for v in col_arr),
                                                      dtype=bool, count=col_arr.shape[0])
                        if not mask_np.any():
                            continue
                        filtered = rg_tbl.filter(pa.array(mask_np))

                kept = filtered.num_rows
                if kept == 0:
                    continue
                total_rows_out += kept
                buffers.setdefault(partition_key, []).append(filtered)
                if sum(t.num_rows for t in buffers[partition_key]) >= batch_rows_threshold:
                    _flush_partition(partition_key)

        for pk in list(buffers.keys()):
            _flush_partition(pk)

        if total_rows_out == 0:
            empty_tbl = pa.table({c: pa.array([], type=schema.field(c).type) for c in keep_cols})
            empty_path = os.path.join(output_folder_or_file_path, "empty.parquet")
            pq.write_table(empty_tbl, empty_path, compression=compression)
            written_files.append(empty_path)
            print("⚠️ No rows matched; wrote empty placeholder.")

        print(f"✅ Filter complete | In rows: {total_rows_in:,} | Out rows: {total_rows_out:,} "
              f"| Files: {len(written_files)} | Fast mode: {fast_mode and not row_wise} | Row-wise: {row_wise}")

        if inplace:
            self.data_path = output_folder_or_file_path
            try:
                self.dataset = ds.dataset(self.data_path, format="parquet")
            except Exception:
                pass

        return {
            "output_folder": output_folder_or_file_path,
            "files": written_files,
            "rows_in": total_rows_in,
            "rows_out": total_rows_out,
            "fast_mode": fast_mode and not row_wise,
            "row_wise": row_wise
        }
    
    
    
    def filter_dataframe_one_column(self,
                     column_name,
                     decision_function,
                     in_place=False,
                     output_folder=None,
                     overwrite=False,
                     preserve_partitions=True,
                     columns=None,
                     batch_rows_threshold='auto',
                     compression="auto",
                     show_progress=True,
                     chunking=True,
                     *args
                     ):
        """
        Filter rows by applying decision_function to column_name.

        Non-parquet: keeps original in-memory behavior.

        Parquet:
        - Streams source dataset fragment by fragment / row group by row group.
        - Applies decision_function(value, *args) per value of column_name.
        - Writes filtered rows to a new parquet dataset rooted at output_folder.
        - Preserves partition folder structure (e.g. year=YYYY/month=MM) if preserve_partitions=True.
        - When chunking=False and input is a single file, writes a single output file.

        Parameters
        ----------
        column_name : str
            Column to evaluate.
        decision_function : callable
            Function returning True/False given (value, *args).
        in_place : bool
            If True (parquet) re-point this DataFrame instance to the filtered dataset root.
        output_folder : str
            Destination parquet root (required for parquet).
        overwrite : bool
            Overwrite output folder if exists.
        preserve_partitions : bool
            Keep original partition directory hierarchy.
        columns : list | None
            Subset of columns to retain (default: all). Filter column is auto-added.
        batch_rows_threshold : int or 'auto'
            Flush buffered filtered rows after this many accumulated rows.
            If 'auto', uses rows_per_group from source parquet metadata.
        compression : str or 'auto'
            Parquet compression codec.
            If 'auto', uses compression codec from source parquet.
        show_progress : bool
            Show tqdm progress bars.
        chunking : bool
            When False and input is a single file, writes a single output file.
        """
        # ---------- Standard pandas path ----------
        if self.data_type != 'parquet' or not hasattr(self, 'dataset'):
            if in_place:
                if len(args) == 2:
                    self.set_dataframe(
                        self.get_dataframe().loc[self.get_column(column_name)
                                                .apply(decision_function, args=(args[0], args[1]))])
                else:
                    self.set_dataframe(
                        self.get_dataframe().loc[self.get_column(column_name)
                                                .apply(decision_function)])
                return self.get_dataframe()
            else:
                if len(args) == 2:
                    return self.get_dataframe().loc[self.get_column(column_name)
                                                    .apply(decision_function, args=(args[0], args[1]))]
                else:
                    return self.get_dataframe().loc[self.get_column(column_name)
                                                    .apply(decision_function)]

        # ---------- Parquet path ----------
        import os, shutil, inspect, re
        import pyarrow as pa, pyarrow.parquet as pq, pyarrow.compute as pc
        from tqdm import tqdm

        if output_folder is None:
            raise ValueError("output_folder is required for parquet filtering.")
        output_folder = os.path.abspath(output_folder)

        # Detect single parquet file vs directory dataset
        is_single_file = os.path.isfile(self.data_path) and self.data_path.lower().endswith(".parquet")
        
        
        # Auto-configure settings from source parquet metadata
        if compression == 'auto' or batch_rows_threshold == 'auto':
            try:
                original_metadata = self.get_parquet_metadata()
                
                if compression == 'auto':
                    compression = original_metadata['compression']
                    print(f"🔄 Using source compression codec: {compression}")
                    
                if batch_rows_threshold == 'auto':
                    batch_rows_threshold = original_metadata['rows_per_group']
                    print(f"🔄 Using source rows per group: {batch_rows_threshold:,}")
                    
            except Exception as e:
                print(f"⚠️ Could not extract metadata from source parquet: {e}")
                # Fallback to defaults
                if compression == 'auto':
                    compression = "zstd"
                    print(f"⚠️ Falling back to default compression: {compression}")
                    
                if batch_rows_threshold == 'auto':
                    batch_rows_threshold = 9_000_000
                    print(f"⚠️ Falling back to default batch size: {batch_rows_threshold:,}")


        # Handle existing output path
        if os.path.exists(output_folder):
            if not overwrite:
                raise FileExistsError(f"{output_folder} exists (overwrite=False).")

        # SINGLE PARQUET FILE MODE
        if is_single_file:
            # Determine if output_folder is a file or directory path
            output_is_file = output_folder.lower().endswith('.parquet')
            
            # For non-chunking mode, ensure we write directly to a single file
            if not chunking and not output_is_file:
                # Create directory and set output file path
                os.makedirs(output_folder, exist_ok=True)
                output_file = os.path.join(output_folder, "filtered.parquet")
            elif output_is_file:
                # Output is already a file path
                os.makedirs(os.path.dirname(output_folder) or ".", exist_ok=True)
                output_file = output_folder
            else:
                # Chunking mode with directory output - create directory
                os.makedirs(output_folder, exist_ok=True)
                output_file = os.path.join(output_folder, "filtered.parquet")

            input_file = self.data_path
            pf = pq.ParquetFile(input_file)
            schema = pf.schema_arrow
            if column_name not in schema.names:
                print(f"⚠️ Column '{column_name}' not found in parquet file.")
                return {"output_file": output_file, "rows_in": 0, "rows_out": 0, "files": []}

            # Columns to keep
            if columns is None:
                keep_cols = list(schema.names)
            else:
                keep_cols = [c for c in columns if c in schema.names]
            if column_name not in keep_cols:
                keep_cols = [column_name] + keep_cols

            # Fast predicate detection
            fast_mode = False
            fast_predicate = None
            if len(args) == 0:
                try:
                    src = inspect.getsource(decision_function).strip()
                    src = re.sub(r"\s+", " ", src)
                    if src.startswith("lambda"):
                        m = re.match(r"lambda\s+([A-Za-z_]\w*)\s*:\s*(.+)", src)
                        if m:
                            var, expr = m.group(1), m.group(2)
                            pat = re.compile(rf"^{var}\s*(==|!=|>=|<=|>|<)\s*([-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)$")
                            pat_rev = re.compile(rf"^([-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)\s*(==|!=|>=|<=|>|<)\s*{var}$")
                            mv = pat.match(expr)
                            if mv:
                                op = mv.group(1); val = float(mv.group(2)); fast_mode = True
                            else:
                                mr = pat_rev.match(expr)
                                if mr:
                                    rev = {"<": ">", ">": "<", "<=": ">=", ">=": "<=", "==": "==", "!=": "!="}
                                    op = rev[mr.group(3)]; val = float(mr.group(1)); fast_mode = True
                            if fast_mode:
                                fn_map = {
                                    ">": pc.greater,
                                    "<": pc.less,
                                    ">=": pc.greater_equal,
                                    "<=": pc.less_equal,
                                    "==": pc.equal,
                                    "!=": pc.not_equal
                                }
                                fast_predicate = (fn_map[op], pa.scalar(val))
                                print(f"⚡ Fast predicate (single file): {column_name} {op} {val}")
                except Exception:
                    pass
            if not fast_mode:
                print("ℹ️  Single file: row-group loop (no simple predicate recognized).")

            total_in = 0
            total_out = 0
            writer = None  # Initialize writer here for direct streaming
            row_groups_written = 0

            # Process all row groups and write directly to output
            pbar = tqdm(range(pf.num_row_groups), desc="RowGroups", disable=not show_progress)
            for rg_idx in pbar:
                rg_tbl = pf.read_row_group(rg_idx, columns=keep_cols, use_threads=True)
                nrg = rg_tbl.num_rows
                total_in += nrg
                if nrg == 0:
                    continue

                if fast_mode:
                    col = rg_tbl[column_name]
                    fn, cst = fast_predicate
                    mask = fn(col, cst)
                    if mask.null_count:
                        mask = pc.fill_null(mask, False)
                    if not pc.any(mask).as_py():
                        continue
                    filtered = rg_tbl.filter(mask)
                else:
                    col_arr = rg_tbl[column_name].to_numpy(zero_copy_only=False)
                    import numpy as np
                    if len(args) == 2:
                        mask_np = np.fromiter((bool(decision_function(v, args[0], args[1])) for v in col_arr),
                                            dtype=bool, count=col_arr.shape[0])
                    elif len(args) == 1:
                        mask_np = np.fromiter((bool(decision_function(v, args[0])) for v in col_arr),
                                            dtype=bool, count=col_arr.shape[0])
                    else:
                        # try vectorized
                        try:
                            vec = decision_function(col_arr)
                            mask_np = np.asarray(vec, dtype=bool)
                            if mask_np.shape[0] != col_arr.shape[0]:
                                raise ValueError
                        except Exception:
                            mask_np = np.fromiter((bool(decision_function(v)) for v in col_arr),
                                                dtype=bool, count=col_arr.shape[0])
                    if not mask_np.any():
                        continue
                    filtered = rg_tbl.filter(pa.array(mask_np))

                kept = filtered.num_rows
                if kept == 0:
                    continue
                
                # Write directly to file instead of collecting in memory
                if writer is None:
                    writer = pq.ParquetWriter(
                        output_file, 
                        filtered.schema,
                        compression=compression,
                        use_dictionary=True,
                        write_statistics=True
                    )
                
                writer.write_table(filtered)
                row_groups_written += 1
                total_out += kept
                
                if show_progress and total_in:
                    pbar.set_postfix(out_rows=f"{total_out:,}")

            if writer is not None:
                writer.close()
                print(f"✅ Single file filter complete | In rows: {total_in:,} | Out rows: {total_out:,} | Row groups written: {row_groups_written}")
            else:
                # No rows matched -> write empty file
                empty_tbl = pa.table({c: pa.array([], type=schema.field(c).type) for c in keep_cols})
                pq.write_table(empty_tbl, output_file, compression=compression)
                print("⚠️ No rows matched; wrote empty parquet.")

            if in_place:
                self.data_path = output_file
                # Rebuild dataset wrapper for consistency
                try:
                    self.dataset = ds.dataset(self.data_path, format="parquet")
                except Exception:
                    pass

            return {
                "output_file": output_file,
                "rows_in": total_in,
                "rows_out": total_out,
                "fast_mode": fast_mode,
                "files": [output_file]
            }

        # -------- MULTI-FILE DATASET MODE --------
        schema = self.dataset.schema
        if column_name not in schema.names:
            print(f"⚠️ Column '{column_name}' not found in parquet schema.")
            return {"output_folder": output_folder, "files": [], "rows_in": 0, "rows_out": 0}
        
        # Create output directory for dataset
        os.makedirs(output_folder, exist_ok=True)

        # Columns to keep
        if columns is None:
            keep_cols = list(schema.names)
        else:
            keep_cols = [c for c in columns if c in schema.names]
        if column_name not in keep_cols:
            keep_cols = [column_name] + keep_cols

        # Attempt simple predicate detection (same as single file mode)
        fast_mode = False
        fast_predicate = None
        if len(args) == 0:
            try:
                src = inspect.getsource(decision_function).strip()
                src = re.sub(r"\s+", " ", src)
                if src.startswith("lambda"):
                    m = re.match(r"lambda\s+([A-Za-z_]\w*)\s*:\s*(.+)", src)
                    if m:
                        var, expr = m.group(1), m.group(2)
                        pat = re.compile(rf"^{var}\s*(==|!=|>=|<=|>|<)\s*([-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)$")
                        pat_rev = re.compile(rf"^([-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)\s*(==|!=|>=|<=|>|<)\s*{var}$")
                        mv = pat.match(expr)
                        if mv:
                            op = mv.group(1); val = float(mv.group(2))
                            fast_mode = True
                        else:
                            mr = pat_rev.match(expr)
                            if mr:
                                rev = {"<": ">", ">": "<", "<=": ">=", ">=": "<=", "==": "==", "!=": "!="}
                                op = rev[mr.group(3)]; val = float(mr.group(1)); fast_mode = True
                        if fast_mode:
                            fn_map = {
                                ">": pc.greater,
                                "<": pc.less,
                                ">=": pc.greater_equal,
                                "<=": pc.less_equal,
                                "==": pc.equal,
                                "!=": pc.not_equal
                            }
                            fast_predicate = (fn_map[op], pa.scalar(val))
                            print(f"⚡ Fast predicate: {column_name} {op} {val}")
            except Exception:
                pass
        if not fast_mode:
            print("ℹ️  Falling back to row-group loop (no simple predicate recognized).")

        fragments = list(self.dataset.get_fragments())
        root_abs = os.path.abspath(self.data_path)
        frag_pbar = tqdm(fragments, desc="Fragments", disable=not show_progress)

        # For non-chunking mode with multi-file dataset, stream data directly to a single file
        if not chunking:
            output_file = os.path.join(output_folder, "filtered.parquet")
            writer = None
            total_rows_in = 0
            total_rows_out = 0
            
            for fragment in frag_pbar:
                frag_rel = fragment.path
                
                # Get fragment path
                if os.path.isabs(frag_rel):
                    rel_path = os.path.relpath(frag_rel, root_abs)
                else:
                    rel_path = frag_rel

                frag_abs = frag_rel if os.path.isabs(frag_rel) else os.path.join(root_abs, frag_rel)
                try:
                    pf = pq.ParquetFile(frag_abs)
                except Exception as e:
                    print(f"⚠️ Could not open {frag_rel}: {e}")
                    continue
                
                for rg_idx in range(pf.num_row_groups):
                    rg_tbl = pf.read_row_group(rg_idx, columns=keep_cols, use_threads=True)
                    nrg = rg_tbl.num_rows
                    total_rows_in += nrg
                    if nrg == 0:
                        continue

                    if fast_mode:
                        col = rg_tbl[column_name]
                        fn, cst = fast_predicate
                        mask = fn(col, cst)
                        if mask.null_count:
                            mask = pc.fill_null(mask, False)
                        if not pc.any(mask).as_py():
                            continue
                        filtered = rg_tbl.filter(mask)
                    else:
                        col_arr = rg_tbl[column_name].to_numpy(zero_copy_only=False)
                        import numpy as np
                        if len(args) == 2:
                            mask_np = np.fromiter((bool(decision_function(v, args[0], args[1])) for v in col_arr),
                                                dtype=bool, count=col_arr.shape[0])
                        elif len(args) == 1:
                            mask_np = np.fromiter((bool(decision_function(v, args[0])) for v in col_arr),
                                                dtype=bool, count=col_arr.shape[0])
                        else:
                            # try vectorized
                            try:
                                vec = decision_function(col_arr)
                                mask_np = np.asarray(vec, dtype=bool)
                                if mask_np.shape[0] != col_arr.shape[0]:
                                    raise ValueError
                            except Exception:
                                mask_np = np.fromiter((bool(decision_function(v)) for v in col_arr),
                                                    dtype=bool, count=col_arr.shape[0])
                        if not mask_np.any():
                            continue
                        filtered = rg_tbl.filter(pa.array(mask_np))

                    kept = filtered.num_rows
                    if kept == 0:
                        continue
                    
                    # Initialize writer with first filtered result's schema
                    if writer is None:
                        writer = pq.ParquetWriter(
                            output_file, 
                            filtered.schema,
                            compression=compression,
                            use_dictionary=True,
                            write_statistics=True
                        )
                    
                    # Write directly to the output file
                    writer.write_table(filtered)
                    total_rows_out += kept
                    
                    # Update progress
                    frag_pbar.set_postfix(out_rows=f"{total_rows_out:,}")
            
            if writer is not None:
                writer.close()
                print(f"✅ Multi-file dataset filter (non-chunking) | In rows: {total_rows_in:,} | Out rows: {total_rows_out:,}")
                
                if in_place:
                    self.data_path = output_file
                    try:
                        self.dataset = ds.dataset(self.data_path, format="parquet")
                    except Exception:
                        pass
                    
                return {
                    "output_file": output_file,
                    "rows_in": total_rows_in,
                    "rows_out": total_rows_out,
                    "fast_mode": fast_mode,
                    "files": [output_file]
                }
            else:
                # No rows matched
                empty_tbl = pa.table({c: pa.array([], type=schema.field(c).type) for c in keep_cols})
                empty_path = os.path.join(output_folder, "empty.parquet")
                pq.write_table(empty_tbl, empty_path, compression=compression)
                print("⚠️ No rows matched; wrote empty placeholder.")
                
                return {
                    "output_file": empty_path,
                    "rows_in": total_rows_in,
                    "rows_out": 0,
                    "fast_mode": fast_mode,
                    "files": [empty_path]
                }

        # -------- CHUNKING MODE (ORIGINAL MULTI-FILE DATASET LOGIC) --------
        total_rows_in = 0
        total_rows_out = 0
        files_written = 0
        buffers = {}
        written_files = []

        def _flush_partition(part_key):
            nonlocal files_written
            tbls = buffers.get(part_key, [])
            if not tbls:
                return
            big = pa.concat_tables(tbls)
            out_dir = os.path.join(output_folder, part_key) if part_key else output_folder
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"part-{files_written:05d}.parquet")
            pq.write_table(big, out_path, compression=compression)
            written_files.append(out_path)
            files_written += 1
            buffers[part_key] = []

        for fragment in frag_pbar:
            frag_rel = fragment.path
            
            # Force relative path for partition preservation so output_folder is honored
            if os.path.isabs(frag_rel):
                rel_path = os.path.relpath(frag_rel, root_abs)
            else:
                rel_path = frag_rel

            partition_key = os.path.dirname(rel_path) if preserve_partitions else ""

            frag_abs = frag_rel if os.path.isabs(frag_rel) else os.path.join(root_abs, frag_rel)
            try:
                pf = pq.ParquetFile(frag_abs)
            except Exception as e:
                print(f"⚠️ Could not open {frag_rel}: {e}")
                continue
            
            for rg_idx in range(pf.num_row_groups):
                rg_tbl = pf.read_row_group(rg_idx, columns=keep_cols, use_threads=True)
                nrg = rg_tbl.num_rows
                total_rows_in += nrg
                if nrg == 0:
                    continue

                if fast_mode:
                    col = rg_tbl[column_name]
                    fn, cst = fast_predicate
                    mask = fn(col, cst)
                    if mask.null_count:
                        mask = pc.fill_null(mask, False)
                    if not pc.any(mask).as_py():
                        continue
                    filtered = rg_tbl.filter(mask)
                else:
                    col_arr = rg_tbl[column_name].to_numpy(zero_copy_only=False)
                    import numpy as np
                    if len(args) == 2:
                        mask_np = np.fromiter((bool(decision_function(v, args[0], args[1])) for v in col_arr),
                                            dtype=bool, count=col_arr.shape[0])
                    elif len(args) == 1:
                        mask_np = np.fromiter((bool(decision_function(v, args[0])) for v in col_arr),
                                            dtype=bool, count=col_arr.shape[0])
                    else:
                        # try vectorized
                        try:
                            vec = decision_function(col_arr)
                            mask_np = np.asarray(vec, dtype=bool)
                            if mask_np.shape[0] != col_arr.shape[0]:
                                raise ValueError
                        except Exception:
                            mask_np = np.fromiter((bool(decision_function(v)) for v in col_arr),
                                                dtype=bool, count=col_arr.shape[0])
                    if not mask_np.any():
                        continue
                    filtered = rg_tbl.filter(pa.array(mask_np))

                kept = filtered.num_rows
                if kept == 0:
                    continue
                total_rows_out += kept
                buffers.setdefault(partition_key, []).append(filtered)
                if sum(t.num_rows for t in buffers[partition_key]) >= batch_rows_threshold:
                    _flush_partition(partition_key)

        # Flush any remaining buffers
        for pk in list(buffers.keys()):
            _flush_partition(pk)

        # Handle empty result
        if total_rows_out == 0:
            empty_tbl = pa.table({c: pa.array([], type=schema.field(c).type) for c in keep_cols})
            empty_path = os.path.join(output_folder, "empty.parquet")
            pq.write_table(empty_tbl, empty_path, compression=compression)
            written_files.append(empty_path)
            print("⚠️ No rows matched; wrote empty placeholder.")

        print(f"✅ Filter complete | In rows: {total_rows_in:,} | Out rows: {total_rows_out:,} "
            f"| Files: {len(written_files)} | Fast mode: {fast_mode}")
        
        # Update data_path if in_place=True
        if in_place:
            self.data_path = output_folder
            try:
                self.dataset = ds.dataset(self.data_path, format="parquet")
            except Exception:
                pass
                
        return {
            "output_folder": output_folder,
            "files": written_files,
            "rows_in": total_rows_in,
            "rows_out": total_rows_out,
            "fast_mode": fast_mode
        }  
    
    def filter_dataframe_old(self,
                         column_name,
                         decision_function,
                         in_place=False,
                         output_folder=None,
                         overwrite=False,
                         preserve_partitions=True,
                         columns=None,
                         batch_rows_threshold=9_000_000,
                         compression="zstd",
                         show_progress=True,
                         *args
                         ):
        """
        Filter rows by applying decision_function to column_name.

        Non-parquet: keeps original in-memory behavior.

        Parquet:
          - Streams source dataset fragment by fragment / row group by row group.
          - Applies decision_function(value, *args) per value of column_name.
          - Writes filtered rows to a new parquet dataset rooted at output_folder.
          - Preserves partition folder structure (e.g. year=YYYY/month=MM) if preserve_partitions=True.

        Parameters
        ----------
        column_name : str
            Column to evaluate.
        decision_function : callable
            Function returning True/False given (value, *args).
        in_place : bool
            If True (parquet) re-point this DataFrame instance to the filtered dataset root.
        output_folder : str
            Destination parquet root (required for parquet).
        overwrite : bool
            Overwrite output folder if exists.
        preserve_partitions : bool
            Keep original partition directory hierarchy.
        columns : list | None
            Subset of columns to retain (default: all). Filter column is auto-added.
        batch_rows_threshold : int
            Flush buffered filtered rows per partition after this many accumulated rows.
        compression : str
            Parquet compression codec.
        show_progress : bool
            Show tqdm progress bars.
        """
        # ---------- Standard pandas path ----------
        if self.data_type != 'parquet' or not hasattr(self, 'dataset'):
            if in_place:
                if len(args) == 2:
                    self.set_dataframe(
                        self.get_dataframe().loc[self.get_column(column_name)
                                                 .apply(decision_function, args=(args[0], args[1]))])
                else:
                    self.set_dataframe(
                        self.get_dataframe().loc[self.get_column(column_name)
                                                 .apply(decision_function)])
                return self.get_dataframe()
            else:
                if len(args) == 2:
                    return self.get_dataframe().loc[self.get_column(column_name)
                                                    .apply(decision_function, args=(args[0], args[1]))]
                else:
                    return self.get_dataframe().loc[self.get_column(column_name)
                                                    .apply(decision_function)]

        # ---------- Parquet path (in_place IGNORED) ----------
        import os, shutil, inspect, re
        import pyarrow as pa, pyarrow.parquet as pq, pyarrow.compute as pc
        from tqdm import tqdm

        if output_folder is None:
            raise ValueError("output_folder is required for parquet filtering.")
        output_folder = os.path.abspath(output_folder)
        if os.path.exists(output_folder):
            if not overwrite:
                raise FileExistsError(f"{output_folder} exists (overwrite=False).")
        os.makedirs(output_folder, exist_ok=True)

        
        schema = self.dataset.schema
        if column_name not in schema.names:
            print(f"⚠️ Column '{column_name}' not found in parquet schema.")
            return {"output_folder": output_folder, "files": [], "rows_in": 0, "rows_out": 0}

        # Columns to keep
        if columns is None:
            keep_cols = list(schema.names)
        else:
            keep_cols = [c for c in columns if c in schema.names]
        if column_name not in keep_cols:
            keep_cols = [column_name] + keep_cols

        # Attempt simple predicate detection
        fast_mode = False
        fast_predicate = None
        if len(args) == 0:
            try:
                src = inspect.getsource(decision_function).strip()
                src = re.sub(r"\s+", " ", src)
                if src.startswith("lambda"):
                    m = re.match(r"lambda\s+([A-Za-z_]\w*)\s*:\s*(.+)", src)
                    if m:
                        var, expr = m.group(1), m.group(2)
                        pat = re.compile(rf"^{var}\s*(==|!=|>=|<=|>|<)\s*([-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)$")
                        pat_rev = re.compile(rf"^([-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)\s*(==|!=|>=|<=|>|<)\s*{var}$")
                        mv = pat.match(expr)
                        if mv:
                            op = mv.group(1); val = float(mv.group(2))
                            fast_mode = True
                        else:
                            mr = pat_rev.match(expr)
                            if mr:
                                rev = {"<": ">", ">": "<", "<=": ">=", ">=": "<=", "==": "==", "!=": "!="}
                                op = rev[mr.group(3)]; val = float(mr.group(1)); fast_mode = True
                        if fast_mode:
                            fn_map = {
                                ">": pc.greater,
                                "<": pc.less,
                                ">=": pc.greater_equal,
                                "<=": pc.less_equal,
                                "==": pc.equal,
                                "!=": pc.not_equal
                            }
                            fast_predicate = (fn_map[op], pa.scalar(val))
                            print(f"⚡ Fast predicate: {column_name} {op} {val}")
            except Exception:
                pass
        if not fast_mode:
            print("ℹ️  Falling back to row-group loop (no simple predicate recognized).")

        fragments = list(self.dataset.get_fragments())
        root_abs = os.path.abspath(self.data_path)
        frag_pbar = tqdm(fragments, desc="Fragments", disable=not show_progress)

        total_rows_in = 0
        total_rows_out = 0
        files_written = 0
        buffers = {}
        written_files = []

        def _flush_partition(part_key):
            nonlocal files_written
            tbls = buffers.get(part_key, [])
            if not tbls:
                return
            big = pa.concat_tables(tbls)
            out_dir = os.path.join(output_folder, part_key) if part_key else output_folder
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"part-{files_written:05d}.parquet")
            pq.write_table(big, out_path, compression=compression)
            written_files.append(out_path)
            files_written += 1
            buffers[part_key] = []

        for fragment in frag_pbar:
            frag_rel = fragment.path
            
            # Force relative path for partition preservation so output_folder is honored
            if os.path.isabs(frag_rel):
                rel_path = os.path.relpath(frag_rel, root_abs)
            else:
                rel_path = frag_rel

            partition_key = os.path.dirname(rel_path) if preserve_partitions else ""

            frag_abs = frag_rel if os.path.isabs(frag_rel) else os.path.join(root_abs, frag_rel)
            try:
                pf = pq.ParquetFile(frag_abs)
            except Exception as e:
                print(f"⚠️ Could not open {frag_rel}: {e}")
                continue
            
            
            for rg_idx in range(pf.num_row_groups):
                rg_tbl = pf.read_row_group(rg_idx, columns=keep_cols, use_threads=True)
                nrg = rg_tbl.num_rows
                total_rows_in += nrg
                if nrg == 0:
                    continue

                if fast_mode:
                    col = rg_tbl[column_name]
                    fn, cst = fast_predicate
                    mask = fn(col, cst)
                    if mask.null_count:
                        mask = pc.fill_null(mask, False)
                    if not pc.any(mask).as_py():
                        continue
                    filtered = rg_tbl.filter(mask)
                else:
                    col_arr = rg_tbl[column_name].to_numpy(zero_copy_only=False)
                    import numpy as np
                    if len(args) == 2:
                        mask_np = np.fromiter((bool(decision_function(v, args[0], args[1])) for v in col_arr),
                                              dtype=bool, count=col_arr.shape[0])
                    elif len(args) == 1:
                        mask_np = np.fromiter((bool(decision_function(v, args[0])) for v in col_arr),
                                              dtype=bool, count=col_arr.shape[0])
                    else:
                        # try vectorized
                        try:
                            vec = decision_function(col_arr)
                            mask_np = np.asarray(vec, dtype=bool)
                            if mask_np.shape[0] != col_arr.shape[0]:
                                raise ValueError
                        except Exception:
                            mask_np = np.fromiter((bool(decision_function(v)) for v in col_arr),
                                                  dtype=bool, count=col_arr.shape[0])
                    if not mask_np.any():
                        continue
                    filtered = rg_tbl.filter(pa.array(mask_np))

                kept = filtered.num_rows
                if kept == 0:
                    continue
                total_rows_out += kept
                buffers.setdefault(partition_key, []).append(filtered)
                if sum(t.num_rows for t in buffers[partition_key]) >= batch_rows_threshold:
                    _flush_partition(partition_key)

        for pk in list(buffers.keys()):
            _flush_partition(pk)

        if total_rows_out == 0:
            empty_tbl = pa.table({c: pa.array([], type=schema.field(c).type) for c in keep_cols})
            empty_path = os.path.join(output_folder, "empty.parquet")
            pq.write_table(empty_tbl, empty_path, compression=compression)
            written_files.append(empty_path)
            print("⚠️ No rows matched; wrote empty placeholder.")

        print(f"✅ Filter complete | In rows: {total_rows_in:,} | Out rows: {total_rows_out:,} "
              f"| Files: {len(written_files)} | Fast mode: {fast_mode}")
        return {
            "output_folder": output_folder,
            "files": written_files,
            "rows_in": total_rows_in,
            "rows_out": total_rows_out,
            "fast_mode": fast_mode
        }
    
    
    def select_datetime_range_parquet(
        self,
        datetime_column: str,
        start_datetime="2020-01-01T00:00:00",
        end_datetime="2024-01-01T00:00:00",
        *,
        output_path: str | None = None,          # if None: auto next to source with suffix "_range"
        overwrite: bool = False,
        preserve_partitions: bool = True,
        show_progress: bool = True,
        in_place: bool = False,
        timezone: str | None = None,             # optional: assume/localize if source is tz-naive strings
    ) -> dict:
        """
        Stream-filter a Parquet input to keep rows within [start_datetime, end_datetime] in datetime_column.

        - Single parquet file: writes a single output file.
        - Parquet dataset (folder): writes a new dataset, preserving partitions if requested.
        - Non-parquet (pandas): filters in memory.

        Returns summary dict. Optionally points this DataFrame to the result when in_place=True.
        """
        import os
        import pandas as pd
        import pyarrow as pa
        import pyarrow.parquet as pq
        from tqdm import tqdm

        # Fallback: in-memory pandas
        if self.data_type != "parquet" or not hasattr(self, "dataset"):
            if datetime_column not in self.dataframe.columns:
                raise ValueError(f"Column '{datetime_column}' not found in DataFrame.")
            s = pd.to_datetime(self.dataframe[datetime_column], errors="coerce")
            s0 = pd.to_datetime(start_datetime)
            s1 = pd.to_datetime(end_datetime)
            mask = s.between(s0, s1, inclusive="both")
            self.dataframe = self.dataframe.loc[mask].copy()
            return {
                "mode": "pandas",
                "rows_out": len(self.dataframe),
                "start": str(s0),
                "end": str(s1),
            }

        # Parquet path
        root_abs = os.path.abspath(self.data_path)
        is_single_file = os.path.isfile(root_abs) and root_abs.lower().endswith(".parquet")

        # Validate column in schema
        schema = None
        try:
            if is_single_file:
                pf0 = pq.ParquetFile(root_abs)
                schema = pf0.schema_arrow
            else:
                schema = self.dataset.schema
        except Exception as e:
            raise RuntimeError(f"Could not read parquet schema: {e}") from e

        if datetime_column not in (schema.names if schema is not None else []):
            raise ValueError(f"Column '{datetime_column}' not found in parquet schema.")

        # Parse range
        dt_start = pd.to_datetime(start_datetime)
        dt_end = pd.to_datetime(end_datetime)

        # Resolve output_path (auto if None)
        def _auto_out_for_single(src_file: str) -> str:
            base, _ = os.path.splitext(src_file)
            return f"{base}_range.parquet"

        def _ensure_dir(path: str):
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        if is_single_file:
            if output_path is None:
                output_path = _auto_out_for_single(root_abs)
            out_is_file = output_path.lower().endswith(".parquet")
            if not out_is_file:
                os.makedirs(output_path, exist_ok=True)
                base = os.path.splitext(os.path.basename(root_abs))[0]
                output_path = os.path.join(output_path, f"{base}_range.parquet")
            if os.path.exists(output_path) and not overwrite:
                raise FileExistsError(f"{output_path} exists (overwrite=False).")
            _ensure_dir(output_path)
        else:
            if output_path is None:
                # create sibling directory with suffix
                parent = os.path.dirname(root_abs.rstrip("\\/"))
                name = os.path.basename(root_abs.rstrip("\\/")) or "dataset"
                output_path = os.path.join(parent, f"{name}_range")
            if os.path.exists(output_path) and not overwrite:
                raise FileExistsError(f"{output_path} exists (overwrite=False).")
            os.makedirs(output_path, exist_ok=True)

        total_rows_in = 0
        total_rows_out = 0
        files_written = 0

        if is_single_file:
            pf = pq.ParquetFile(root_abs)
            # Try to keep source compression
            try:
                src_codec = pf.metadata.row_group(0).column(0).compression
            except Exception:
                src_codec = "zstd"
            writer = None
            rg_iter = tqdm(range(pf.num_row_groups), desc="RowGroups", disable=not show_progress)
            for rg_idx in rg_iter:
                tbl = pf.read_row_group(rg_idx, use_threads=True)
                if tbl.num_rows == 0:
                    continue
                total_rows_in += tbl.num_rows
                # Build boolean mask via pandas for robustness
                ser = tbl[datetime_column].to_pandas()
                ser = pd.to_datetime(ser, errors="coerce", utc=False)
                if timezone and getattr(ser.dt.tz, "zone", None) is None:
                    try:
                        ser = ser.dt.tz_localize(timezone)
                    except Exception:
                        pass
                # Compare with naive targets; if tz-aware, convert to naive for compare
                if getattr(ser.dt, "tz", None) is not None:
                    ser_cmp = ser.dt.tz_convert(None)
                else:
                    ser_cmp = ser
                mask = ser_cmp.between(dt_start, dt_end, inclusive="both")
                if not mask.any():
                    continue
                kept = tbl.filter(pa.array(mask.to_numpy(dtype=bool).tolist()))
                if kept.num_rows == 0:
                    continue
                total_rows_out += kept.num_rows
                if writer is None:
                    writer = pq.ParquetWriter(
                        output_path, kept.schema, compression=src_codec, use_dictionary=True, write_statistics=True
                    )
                writer.write_table(kept)
                if show_progress:
                    rg_iter.set_postfix(out=f"{total_rows_out:,}")
            if writer is not None:
                writer.close()
                files_written = 1
            else:
                # write empty file with same schema
                empty_tbl = pa.table({n: pa.array([], type=schema.field(n).type) for n in schema.names})
                pq.write_table(empty_tbl, output_path, compression=src_codec)
                files_written = 1

            if in_place:
                self.data_path = output_path
                try:
                    import pyarrow.dataset as ds
                    self.dataset = ds.dataset(self.data_path, format="parquet")
                except Exception:
                    pass
            
            print(f"✅ Single-file filter | In rows: {total_rows_in:,} | Out rows: {total_rows_out:,}")
            # file saved to 
            print(f"File saved to: {output_path}")
            return {
                "mode": "single_file",
                "output_file": output_path,
                "rows_in": total_rows_in,
                "rows_out": total_rows_out,
                "files_written": files_written,
                "start": str(dt_start),
                "end": str(dt_end),
            }

        # Dataset mode
        import pyarrow.dataset as ds
        fragments = list(self.dataset.get_fragments())
        frag_bar = tqdm(fragments, desc="Fragments", disable=not show_progress)
        for fragment in frag_bar:
            frag_rel = fragment.path
            frag_abs = frag_rel if os.path.isabs(frag_rel) else os.path.join(root_abs, frag_rel)
            try:
                pf = pq.ParquetFile(frag_abs)
            except Exception as e:
                print(f"⚠️ Skip unreadable fragment {frag_rel}: {e}")
                continue

            # Determine output file path
            if os.path.isabs(frag_rel):
                rel_path = os.path.relpath(frag_rel, root_abs)
            else:
                rel_path = frag_rel
            out_dir = os.path.join(output_path, os.path.dirname(rel_path)) if preserve_partitions else output_path
            os.makedirs(out_dir, exist_ok=True)
            out_file = os.path.join(out_dir, os.path.basename(rel_path))

            # Keep source compression if possible
            try:
                src_codec = pf.metadata.row_group(0).column(0).compression
            except Exception:
                src_codec = "zstd"

            writer = None
            for rg_idx in range(pf.num_row_groups):
                tbl = pf.read_row_group(rg_idx, use_threads=True)
                if tbl.num_rows == 0:
                    continue
                total_rows_in += tbl.num_rows

                ser = tbl[datetime_column].to_pandas()
                ser = pd.to_datetime(ser, errors="coerce", utc=False)
                if timezone and getattr(ser.dt.tz, "zone", None) is None:
                    try:
                        ser = ser.dt.tz_localize(timezone)
                    except Exception:
                        pass
                if getattr(ser.dt, "tz", None) is not None:
                    ser_cmp = ser.dt.tz_convert(None)
                else:
                    ser_cmp = ser
                mask = ser_cmp.between(dt_start, dt_end, inclusive="both")
                if not mask.any():
                    continue

                kept = tbl.filter(pa.array(mask.to_numpy(dtype=bool).tolist()))
                if kept.num_rows == 0:
                    continue

                if writer is None:
                    writer = pq.ParquetWriter(
                        out_file, kept.schema, compression=src_codec, use_dictionary=True, write_statistics=True
                    )
                writer.write_table(kept)
                total_rows_out += kept.num_rows

            if writer is not None:
                writer.close()
                files_written += 1
                if show_progress:
                    frag_bar.set_postfix(files=files_written, out=f"{total_rows_out:,}")

        if in_place:
            self.data_path = output_path
            try:
                self.dataset = ds.dataset(self.data_path, format="parquet")
            except Exception:
                pass

        return {
            "mode": "dataset",
            "output_root": output_path,
            "files_written": files_written,
            "rows_in": total_rows_in,
            "rows_out": total_rows_out,
            "start": str(dt_start),
            "end": str(dt_end),
            "preserve_partitions": preserve_partitions,
        }
    
    
    def select_datetime_range(self, start_datetime, end_datetime, datetime_format='%Y-%m-%d %H:%M:%S', in_place=True, **kwargs):
        #start_datetime = datetime.strptime(start_datetime, datetime_format)
        #end_datetime = datetime.strptime(end_datetime, datetime_format)
        if in_place is True:
            self.dataframe = self.dataframe[(self.dataframe.index >= start_datetime) & (self.dataframe.index <= end_datetime)]
        else:
            return self.dataframe[(self.dataframe.index >= start_datetime) & (self.dataframe.index <= end_datetime)]
    
    def ignore_time_in_datetime(self, datetime_column_name='datetime', in_place=True, *args):
        self.transform_column(datetime_column_name, lambda o: datetime.strptime(o.date().isoformat(), "%Y-%m-%d"), in_place=in_place)
        
    def ignore_day_in_date(self, datetime_column_name='datetime', in_place=True, *args):
        # Extract year and month from the 'date' column
        self.dataframe['year_month'] = self.dataframe[datetime_column_name].dt.to_period('M')
        self.drop_column(datetime_column_name)
        self.rename_columns({'year_month': datetime_column_name})

    def transform_column(self, column_name, transformation_func, in_place=True, *args):
        """_summary_

        Args:
            column_to_trsform (_type_): column to transform
            column_src (_type_): Column to use as a source for the transformation
            fun_de_trasformation (_type_): The function of transformation, if it has multiple arguments pass them as args:
            example: data.transform_column(column, column, Lib.remove_stopwords, True, stopwords)
            in_place (bool, optional): If true the changes will affect the original dataframe. Defaults to True.

        Returns:
            _type_: _description_
        """
        if in_place is True:
            if (len(args) != 0):
                self.set_column(column_name, self.get_column(column_name).apply(transformation_func, args=(args[0],)))
            else:
                self.set_column(column_name, self.get_column(column_name).apply(transformation_func))
        else:
            if (len(args) != 0):
                return self.get_column(column_name).apply(transformation_func, args=(args[0],))
            else:
                return self.get_column(column_name).apply(transformation_func)
            
    def column_to_no_accent(self, column):
        self.transform_column(column, Lib.no_accent)
        self.set_column(column, self.get_column(column))
        
    def column_keep_only_alphanumeric(self, column_name):
        from re import sub
        pattern = r'[^a-zA-Z0-9\s]'
        self.transform_column(column_name, lambda o: sub(pattern, '', o))

    def write_dataframe_in_file(self, out_file='data/out.csv', delimiter=','):
        Lib.write_liste_csv(self.get_dataframe().values, out_file, delimiter)

    def sort(self, by_column_name_list=None, ascending=True, **kwargs):
        if by_column_name_list is None:
            self.dataframe.sort_index(ascending=ascending, inplace=True,**kwargs)
        else:
            self.set_dataframe(self.get_dataframe().sort_values(by=by_column_name_list, ascending=ascending, **kwargs))

    def count_occurence_of_each_row(self, column):
        return self.get_dataframe().pivot_table(index=[column], aggfunc='size')
    
    def get_distinct_values_as_list(self, column):
        return list(self.get_dataframe().pivot_table(index=[column], aggfunc='size').index)
    
    def encode_textual_column(self, column):
        mapping = list(self.get_dataframe().pivot_table(index=[column], aggfunc='size').index)
        self.transform_column(column, lambda o : mapping.index(o))
        mapping_dict = {index: value for index, value in enumerate(mapping)}
        return mapping_dict
    
    def reverse_column_from_numerical_values(self, column, maping):
        self.trasform_column(column, column, lambda o : maping[int(o)])

    def count_occurence_of_row_as_count_column(self, column):
        column_name = 'count'
        self.set_column(column_name, self.get_column(column).value_counts())
        self.transform_column(column_name, column, lambda x:self.get_column(column).value_counts().get(x))
        return self.get_dataframe()
    
    def get_count_number_of_all_words(self, column):
        self.apply_fun_to_column(column, lambda x: len(x.split(' ')))
        return self.get_column(column).sum()
    
    def get_count_occurrence_of_value(self, column, value, case_sensitive=True):
        
        if case_sensitive:
            self.apply_fun_to_column(column, lambda x: x.split(' ').count(value))
            return self.get_column(column).sum()
        else:
            self.apply_fun_to_column(column, lambda x: list(map(str.lower, x.split(' '))).count(value))
            return self.get_column(column).sum()

    def count_true_decision_function_rows(self, column, decision_function):
        self.filter_dataframe(column, decision_function)
        
    def add_artificial_missing_data(self, column_name, nbr_missing_data=31, method='continious'):
        """
        Fill a randomly selected period with NaN values in a specified column of a pandas dataframe.

        Parameters:
            df (pandas.DataFrame): The input dataframe.
            column (str): The name of the column to fill with NaN values.
            period (str): The length of the period to fill with NaN values, in pandas frequency string format (e.g., 'D' for day, 'W' for week, 'M' for month).

        Returns:
            None
        """
        
        filled_indices = []
        if method == 'continious':
            random_index = self.dataframe.sample().index[0]
            #self.dataframe[random_index:random_index+nbr_missing_data][column_name] = np.nan 
            previous_data = self.dataframe.loc[(self.dataframe.index >= random_index) & (self.dataframe.index < random_index + nbr_missing_data), column_name]
            self.dataframe.loc[(self.dataframe.index >= random_index) & (self.dataframe.index < random_index + nbr_missing_data), column_name] = np.nan
            # Fill the selected period with NaN values in the specified column
            #df.loc[(df.index >= random_period) & (df.index < random_period + pd.Timedelta(period)), column] = np.nan
            filled_indices = range(random_index, random_index + nbr_missing_data) 
        else:
            for  i in range(nbr_missing_data):
                random_index = self.dataframe.sample().index[0]
                self.dataframe.loc[self.dataframe.index == random_index, column_name] = np.nan
                filled_indices.append(random_index)
        return previous_data
            
    def plot_column(self, column_name, x_column_name='index', 
                    x_label='Date & time',
                    y_label=None,
                    x_label_rotation=0,
                    y_label_rotation=0,
                    save_fig=False,
                    savefig_path='out.png',
                    date_format_x_axis=None
                    ):
        """_summary_

        Args:
            column_name (_type_): _description_
            x_column_name (str, optional): _description_. Defaults to 'index'.
            x_label (str, optional): _description_. Defaults to 'Date & time'.
            y_label (_type_, optional): _description_. Defaults to None.
            x_label_rotation (int, optional): _description_. Defaults to 0.
            y_label_rotation (int, optional): _description_. Defaults to 0.
            save_fig (bool, optional): _description_. Defaults to False.
            savefig_path (str, optional): _description_. Defaults to 'out.png'.
            date_format_x_axis (str, optional): _description_. Example to '%m-%d'.
        """
        fig, ax = plt.subplots()
        
        if date_format_x_axis is not None:
            import matplotlib.dates as mdate
            # format the x-axis tick labels
            date_format = mdate.DateFormatter(date_format_x_axis)
            ax.xaxis.set_major_formatter(date_format)
            
        if x_column_name == 'index':
            ax.plot(self.get_index(), self.get_column(column_name))
        else:
            ax.plot(self.get_column(x_column_name), self.get_column(column_name))
        # set the axis labels and title
        ax.set_xlabel(x_label)
        if y_label is None:
            y_label = column_name
        ax.set_ylabel(y_label)
        ax.tick_params(axis='x', rotation=x_label_rotation)
        ax.tick_params(axis='y', rotation=y_label_rotation)
        plt.tight_layout()
        if save_fig is True:
            import matplotlib as mpl
            mpl.rcParams['agg.path.chunksize'] = 10000
            fig.savefig(savefig_path, dpi=720)
        plt.show()
        
    def split_export(self, percentage=0.8, train_out_file="train.csv", test_out_file="test.csv"):
        train = self.dataframe.iloc[:int(percentage*self.get_shape()[0]), :]
        test = self.dataframe.iloc[int(percentage*self.get_shape()[0]):, :]
        train.to_csv(train_out_file, index=False)
        test.to_csv(test_out_file, index=False) 
        
    def show_wordcloud(self, column):
        wordcloud = WordCloud(
            background_color='white',
            max_words=100,
            max_font_size=30,
            scale=3,
            random_state=1)
   
        wordcloud = wordcloud.generate(self.get_column_as_joined_text(column))
        fig = plt.figure(1, figsize=(12, 12))
        plt.axis('off')
        plt.imshow(wordcloud)
        plt.show()
    
    
    def export_rows_as_parquet(
        self,
        output_path: str = "data/subset.parquet",
        percentage: float | None = None,
        n_rows: int | None = None,
        columns: list[str] | None = None,
        overwrite: bool = False,
        compression: str = "zstd",
        show_progress: bool = True,
    ) -> dict:
        """
        Export a subset of rows to a new Parquet file.

        Use exactly one of n_rows or percentage (percentage in [0,1] or [0,100]).
        Works for:
          - standard pandas DataFrame
          - parquet dataset (streams row groups, writes a single output file)

        Args:
            output_path: Destination parquet file path ('.parquet'). If a directory is provided,
                         a file named 'subset.parquet' is created inside it.
            n_rows: Number of rows to export.
            percentage: Fraction (0-1) or percent (0-100) of total rows to export.
            columns: Optional subset of columns to export.
            overwrite: Overwrite existing output_path if it exists.
            compression: Parquet compression codec (e.g., 'zstd', 'snappy', 'gzip').
            show_progress: Show progress for parquet datasets.

        Returns:
            dict with {'output_file', 'rows_requested', 'rows_written', 'columns'}
        """
        import os
        import math
        import pandas as pd

        # Validate selection arguments
        if (n_rows is None and percentage is None) or (n_rows is not None and percentage is not None):
            raise ValueError("Provide exactly one of n_rows or percentage.")

        # Normalize output path to a file
        out_path = output_path
        out_is_dir = os.path.isdir(out_path) or (not out_path.lower().endswith(".parquet"))
        if out_is_dir:
            os.makedirs(out_path, exist_ok=True)
            out_path = os.path.join(out_path, "subset.parquet")

        if os.path.exists(out_path) and not overwrite:
            raise FileExistsError(f"{out_path} exists. Set overwrite=True to overwrite.")

        # Compute rows to export
        if self.data_type != 'parquet' or not hasattr(self, 'dataset'):
            total_rows = len(self.get_dataframe())
        else:
            # Efficient total rows estimation for parquet dataset
            total_rows = self.get_shape()[0]

        if percentage is not None:
            pct = float(percentage)
            if pct > 1.0:
                pct = pct / 100.0
            if not (0.0 < pct <= 1.0):
                raise ValueError("percentage must be in (0,1] or (0,100].")
            target_rows = int(math.ceil(total_rows * pct))
        else:
            target_rows = int(n_rows)
            if target_rows <= 0:
                raise ValueError("n_rows must be > 0.")

        # Pandas path
        if self.data_type != 'parquet' or not hasattr(self, 'dataset'):
            df = self.get_dataframe()
            if columns is not None:
                missing = [c for c in columns if c not in df.columns]
                if missing:
                    raise ValueError(f"Columns not found: {missing}")
                df = df[columns]
            subset = df.head(target_rows)
            subset.to_parquet(out_path, index=False, compression=compression)
            return {
                "output_file": out_path,
                "rows_requested": target_rows,
                "rows_written": len(subset),
                "columns": list(subset.columns),
            }

        # Parquet dataset path (single output file)
        import pyarrow as pa
        import pyarrow.parquet as pq
        import os
        from tqdm import tqdm

        schema = self.dataset.schema
        # Validate/choose columns
        if columns is None:
            keep_cols = list(schema.names)
        else:
            keep_cols = [c for c in columns if c in schema.names]
            missing = [c for c in (columns or []) if c not in schema.names]
            if missing:
                raise ValueError(f"Columns not found in dataset schema: {missing}")

        root_abs = os.path.abspath(self.data_path)
        fragments = list(self.dataset.get_fragments())
        remaining = target_rows
        writer = None
        rows_written = 0

        # Progress over files/row-groups
        frag_iter = tqdm(fragments, desc="Fragments", disable=not show_progress)
        try:
            for fragment in frag_iter:
                if remaining <= 0:
                    break
                frag_rel = getattr(fragment, "path", None) or ""
                frag_abs = frag_rel if os.path.isabs(frag_rel) else os.path.join(root_abs, frag_rel)
                try:
                    pf = pq.ParquetFile(frag_abs)
                except Exception as e:
                    # Fallback: try reading as a table from the fragment
                    try:
                        tbl = fragment.to_table(columns=keep_cols)
                        if remaining < tbl.num_rows:
                            tbl = tbl.slice(0, remaining)
                        if writer is None:
                            writer = pq.ParquetWriter(out_path, tbl.schema, compression=compression, use_dictionary=True, write_statistics=True)
                        writer.write_table(tbl)
                        rows_written += tbl.num_rows
                        remaining -= tbl.num_rows
                        frag_iter.set_postfix(written=rows_written)
                        continue
                    except Exception as ee:
                        raise RuntimeError(f"Failed to read fragment: {frag_rel}. Error: {ee}") from ee

                rg_iter = range(pf.num_row_groups)
                if show_progress:
                    rg_iter = tqdm(rg_iter, desc="RowGroups", leave=False)
                for rg_idx in rg_iter:
                    if remaining <= 0:
                        break
                    rg_tbl = pf.read_row_group(rg_idx, columns=keep_cols, use_threads=True)
                    if rg_tbl.num_rows == 0:
                        continue
                    to_write = rg_tbl if remaining >= rg_tbl.num_rows else rg_tbl.slice(0, remaining)
                    if writer is None:
                        writer = pq.ParquetWriter(out_path, to_write.schema, compression=compression, use_dictionary=True, write_statistics=True)
                    writer.write_table(to_write)
                    rows_written += to_write.num_rows
                    remaining -= to_write.num_rows
                    if show_progress:
                        frag_iter.set_postfix(written=rows_written)
        finally:
            if writer is not None:
                writer.close()

        # Handle empty result (e.g., empty dataset)
        if rows_written == 0:
            empty_tbl = pa.table({c: pa.array([], type=schema.field(c).type) for c in keep_cols})
            pq.write_table(empty_tbl, out_path, compression=compression)

        print(f"✅ Exported {rows_written} rows to {out_path} (requested {target_rows}).")
        return {
            "output_file": out_path,
            "rows_requested": target_rows,
            "rows_written": rows_written,
            "columns": keep_cols,
        }
    
    
    
    def reindex_dataframe(self, index_as_column_name=None, index_as_liste=None):
        if index_as_liste is not None:
            new_index = new_index = index_as_liste
            self.get_dataframe().index = new_index
        if index_as_column_name is not None:
            self.dataframe.set_index(index_as_column_name, inplace=True)
        if index_as_column_name is None and index_as_liste is None:
            new_index = pd.Series(np.arange(self.get_shape()[0]))
            self.get_dataframe().index = new_index
    
    def get_columns_names(self):
        """
        Get the column names of the dataframe.
        
        For Parquet datasets, retrieves column names directly from the dataset schema.
        For regular pandas DataFrames, gets column names from the dataframe.
        
        Returns:
        --------
        list
            List of column names
        """
        if self.data_type == 'parquet' and hasattr(self, 'dataset') and self.dataset is not None:
            try:
                # Get column names from PyArrow dataset schema
                return self.dataset.schema.names
            except Exception as e:
                print(f"Warning: Could not get column names from Parquet dataset: {e}")
                # Fall back to pandas dataframe if available
                if hasattr(self, 'dataframe') and self.dataframe is not None:
                    return list(self.dataframe.columns)
                return []
        else:
            # Regular pandas dataframe
            return list(self.dataframe.columns)
    
    def export_column(self, column_name, out_file='out.csv'):
        self.get_column(column_name).to_csv(out_file, index=False)
    
    def export(self, destination_path='data/json_dataframe.csv', type='csv', index=True):
        if type == 'json':
            destination_path = 'data/json_dataframe.json'
            self.get_dataframe().to_json(destination_path)
        elif type == 'csv':
            import os
            # Parquet source: stream to CSV
            if self.data_type == 'parquet' and hasattr(self, 'dataset') and self.dataset is not None:
                import pyarrow.parquet as pq

                is_single_file = os.path.isfile(self.data_path) and self.data_path.lower().endswith(".parquet")

                # If destination is a single CSV file: append row-groups/fragments into one file
                if destination_path.lower().endswith(".csv"):
                    out_file = destination_path
                    os.makedirs(os.path.dirname(out_file) or ".", exist_ok=True)
                    # Write header only once
                    write_header = True

                    def write_table_to_csv(tbl):
                        nonlocal write_header
                        df_chunk = tbl.to_pandas()
                        df_chunk.to_csv(out_file, mode=("w" if write_header else "a"), header=write_header, index=index)
                        write_header = False

                    if is_single_file:
                        pf = pq.ParquetFile(self.data_path)
                        for rg_idx in range(pf.num_row_groups):
                            tbl = pf.read_row_group(rg_idx, use_threads=True)
                            if tbl.num_rows:
                                write_table_to_csv(tbl)
                    else:
                        # Dataset: iterate fragments then row-groups
                        root_abs = os.path.abspath(self.data_path)
                        for frag in self.dataset.get_fragments():
                            frag_rel = getattr(frag, "path", "")
                            frag_abs = frag_rel if os.path.isabs(frag_rel) else os.path.join(root_abs, frag_rel)
                            try:
                                pf = pq.ParquetFile(frag_abs)
                                for rg_idx in range(pf.num_row_groups):
                                    tbl = pf.read_row_group(rg_idx, use_threads=True)
                                    if tbl.num_rows:
                                        write_table_to_csv(tbl)
                            except Exception:
                                # Fallback: whole fragment
                                tbl = frag.to_table()
                                if tbl.num_rows:
                                    write_table_to_csv(tbl)

                else:
                    # Destination is a directory: mirror files -> CSVs (keeps partition structure for dataset)
                    out_root = destination_path
                    os.makedirs(out_root, exist_ok=True)

                    if is_single_file:
                        base = os.path.splitext(os.path.basename(self.data_path))[0]
                        out_file = os.path.join(out_root, f"{base}.csv")
                        pf = pq.ParquetFile(self.data_path)
                        write_header = True
                        for rg_idx in range(pf.num_row_groups):
                            tbl = pf.read_row_group(rg_idx, use_threads=True)
                            if tbl.num_rows:
                                df_chunk = tbl.to_pandas()
                                df_chunk.to_csv(out_file, mode=("w" if write_header else "a"), header=write_header, index=index)
                                write_header = False
                    else:
                        root_abs = os.path.abspath(self.data_path)
                        for frag in self.dataset.get_fragments():
                            frag_rel = getattr(frag, "path", "")
                            # Mirror partition dirs and change extension to .csv
                            rel_path = frag_rel if os.path.isabs(frag_rel) else os.path.relpath(
                                os.path.join(root_abs, frag_rel), root_abs
                            )
                            rel_csv = os.path.splitext(rel_path)[0] + ".csv"
                            out_file = os.path.join(out_root, rel_csv)
                            os.makedirs(os.path.dirname(out_file), exist_ok=True)

                            write_header = True
                            try:
                                pf = pq.ParquetFile(os.path.join(root_abs, frag_rel) if not os.path.isabs(frag_rel) else frag_rel)
                                for rg_idx in range(pf.num_row_groups):
                                    tbl = pf.read_row_group(rg_idx, use_threads=True)
                                    if tbl.num_rows:
                                        df_chunk = tbl.to_pandas()
                                        df_chunk.to_csv(out_file, mode=("w" if write_header else "a"),
                                                        header=write_header, index=index)
                                        write_header = False
                            except Exception:
                                # Fallback: write full fragment
                                tbl = frag.to_table()
                                if tbl.num_rows:
                                    df_chunk = tbl.to_pandas()
                                    df_chunk.to_csv(out_file, mode=("w" if write_header else "a"),
                                                    header=write_header, index=index)
                                    write_header = False
            else:
                # Standard pandas path
                self.get_dataframe().to_csv(destination_path, index=index)
        elif type == 'pkl':
            self.get_dataframe().to_pickle(destination_path)
        elif type == 'parquet':
            import os
            try:
                import pyarrow as pa
                import pyarrow.parquet as pq
            except Exception:
                pa = None
                pq = None
            # If source is a parquet dataset/file, stream out accordingly
            if self.data_type == 'parquet' and hasattr(self, 'dataset') and self.dataset is not None and pq is not None:
                # If destination is a single file, merge all fragments/row-groups into that file
                if destination_path.lower().endswith(".parquet"):
                    out_file = destination_path
                    os.makedirs(os.path.dirname(out_file) or ".", exist_ok=True)
                    writer = None
                    root_abs = os.path.abspath(self.data_path)
                    fragments = list(self.dataset.get_fragments())
                    for frag in fragments:
                        frag_rel = getattr(frag, "path", "")
                        frag_abs = frag_rel if os.path.isabs(frag_rel) else os.path.join(root_abs, frag_rel)
                        try:
                            pf = pq.ParquetFile(frag_abs)
                        except Exception:
                            # Fallback: read whole fragment
                            try:
                                tbl = frag.to_table()
                                if writer is None:
                                    writer = pq.ParquetWriter(
                                        out_file, tbl.schema, compression="zstd", use_dictionary=True, write_statistics=True
                                    )
                                writer.write_table(tbl)
                            except Exception:
                                continue
                            continue
                        for rg_idx in range(pf.num_row_groups):
                            rg_tbl = pf.read_row_group(rg_idx, use_threads=True)
                            if writer is None:
                                writer = pq.ParquetWriter(
                                    out_file, rg_tbl.schema, compression="zstd", use_dictionary=True, write_statistics=True
                                )
                            writer.write_table(rg_tbl)
                    if writer is not None:
                        writer.close()
                else:
                    # Destination is a directory -> mirror dataset into that folder (preserve partitions)
                    out_root = destination_path
                    os.makedirs(out_root, exist_ok=True)
                    root_abs = os.path.abspath(self.data_path)
                    fragments = list(self.dataset.get_fragments())
                    for frag in fragments:
                        frag_rel = getattr(frag, "path", "")
                        rel_path = frag_rel if os.path.isabs(frag_rel) else os.path.relpath(
                            os.path.join(root_abs, frag_rel), root_abs
                        )
                        src_abs = frag_rel if os.path.isabs(frag_rel) else os.path.join(root_abs, frag_rel)
                        out_file = os.path.join(out_root, rel_path)
                        os.makedirs(os.path.dirname(out_file), exist_ok=True)
                        try:
                            pf = pq.ParquetFile(src_abs)
                        except Exception:
                            # Fallback: read whole fragment and rewrite
                            try:
                                tbl = frag.to_table()
                                pq.write_table(tbl, out_file, compression="zstd")
                            except Exception:
                                continue
                            continue
                        writer = None
                        for rg_idx in range(pf.num_row_groups):
                            rg_tbl = pf.read_row_group(rg_idx, use_threads=True)
                            if writer is None:
                                writer = pq.ParquetWriter(
                                    out_file, rg_tbl.schema, compression="zstd", use_dictionary=True, write_statistics=True
                                )
                            writer.write_table(rg_tbl)
                        if writer is not None:
                            writer.close()
            else:
                # In-memory DataFrame -> write directly to parquet
                os.makedirs(os.path.dirname(destination_path) or ".", exist_ok=True)
                self.get_dataframe().to_parquet(destination_path, index=index)
        print('DataFrame exported successfully to ' + destination_path)
   
    def sample(self, n=10, frac=None):
        if frac is not None:
            return self.get_dataframe().sample(n=frac)
        return self.get_dataframe().sample(n=n)

    def show(self, number_of_row=None):
        """
        Display rows of the underlying data.

        For parquet datasets (lazy):
        - number_of_row is None  -> show first 10 rows (dataset head)
        - number_of_row > 0      -> show that many rows from the start
        - number_of_row < 0      -> attempt to show last |n| rows (may require larger read)
        """
        if self.data_type == 'parquet' and hasattr(self, 'dataset'):
            import pyarrow.parquet as pq

            # Default head size
            if number_of_row is None:
                n = 10
                table = self.dataset.head(n)
                df = table.to_pandas()
                print(df)
                return df

            if number_of_row >= 0:
                n = number_of_row
                if n == 0:
                    print("⚠️ number_of_row=0 -> nothing to show.")
                    return pd.DataFrame()
                table = self.dataset.head(n)
                df = table.to_pandas()
                print(df)
                return df
            else:
                # Tail logic
                tail_n = abs(number_of_row)

                # Estimate total rows (metadata only)
                total_rows_est = 0
                try:
                    for fragment in self.dataset.get_fragments():
                        try:
                            pf = pq.ParquetFile(fragment.path)
                            total_rows_est += pf.metadata.num_rows
                        except Exception:
                            # Fallback (may be expensive)
                            total_rows_est += fragment.to_table().num_rows
                except Exception:
                    total_rows_est = None

                if total_rows_est is not None and tail_n >= total_rows_est:
                    # Just return full dataset (warn if large)
                    if total_rows_est > 200_000:
                        print(f"⚠️ Reading full dataset ({total_rows_est:,} rows) to satisfy tail request.")
                    full_tbl = self.dataset.to_table()
                    df_full = full_tbl.to_pandas()
                    print(df_full.tail(tail_n))
                    return df_full.tail(tail_n)

                # If dataset is small enough, read all; else fallback to last row groups heuristic
                read_all = (total_rows_est is not None and total_rows_est <= 500_000)
                if read_all:
                    full_tbl = self.dataset.to_table()
                    df_full = full_tbl.to_pandas()
                    tail_df = df_full.tail(tail_n)
                    print(tail_df)
                    return tail_df
                else:
                    # Heuristic: read last row groups of last fragment(s) until enough rows collected
                    collected = []
                    remaining = tail_n
                    fragments = list(self.dataset.get_fragments())
                    # Iterate fragments in reverse
                    for fragment in reversed(fragments):
                        try:
                            pf = pq.ParquetFile(fragment.path)
                            # Iterate row groups in reverse
                            for rg_idx in reversed(range(pf.num_row_groups)):
                                rg_tbl = pf.read_row_group(rg_idx)
                                df_rg = rg_tbl.to_pandas()
                                collected.append(df_rg)
                                remaining -= len(df_rg)
                                if remaining <= 0:
                                    break
                            if remaining <= 0:
                                break
                        except Exception:
                            # Fallback: load entire fragment
                            try:
                                tbl = fragment.to_table()
                                df_rg = tbl.to_pandas()
                                collected.append(df_rg)
                                remaining -= len(df_rg)
                                if remaining <= 0:
                                    break
                            except Exception:
                                continue
                    if not collected:
                        print("⚠️ Could not collect tail rows; returning empty DataFrame.")
                        return pd.DataFrame()
                    tail_df = pd.concat(reversed(collected), ignore_index=True).tail(tail_n)
                    print(tail_df)
                    return tail_df

        # -------- Standard pandas path --------
        if number_of_row is None:
            print(self.get_dataframe())
            return self.get_dataframe()
        elif number_of_row < 0:
            print(self.get_dataframe().tail(abs(number_of_row)))
            return self.get_dataframe().tail(abs(number_of_row))
        else:
            print(self.get_dataframe().head(number_of_row))
            return self.get_dataframe().head(number_of_row)

    
    def set_dataframe_from_parquet_sample(
        self,
        ratio: float | None = None,
        columns: list[str] | None = None,
        n_rows: int | None = None,
        random: bool = False,
        random_state: int = 42,
        show_progress: bool = True,
    ):
        """
        Load a sample from a parquet file into self.dataframe.
        - If random=True: sample uniformly across the file.
        - If random=False: read sequentially from the beginning.

        Args:
            parquet_path: path to the parquet file.
            ratio: fraction of total rows to sample (mutually exclusive with n_rows).
            columns: optional list of columns to read.
            n_rows: exact number of rows to sample (mutually exclusive with ratio).
            random: random vs sequential sampling.
            random_state: seed for random sampling.
            show_progress: show tqdm progress.

        Returns:
            dict: {'rows': int, 'total_rows': int, 'path': str}
        """
        import pyarrow.parquet as pq
        import pandas as pd
        import numpy as np
        from math import ceil
        from tqdm import tqdm
        
        parquet_path = self.data_path

        pf = pq.ParquetFile(parquet_path)
        num_rgs = pf.metadata.num_row_groups
        rg_rows = [pf.metadata.row_group(i).num_rows for i in range(num_rgs)]
        total_rows = int(sum(rg_rows))

        if total_rows == 0:
            self.set_dataframe(pd.DataFrame(), data_type='df')
            self.data_path = parquet_path
            self.data_type = 'df'
            return {'rows': 0, 'total_rows': 0, 'path': parquet_path}

        target = n_rows if n_rows is not None else int(ceil((ratio or 0.0) * total_rows))
        target = max(0, min(total_rows, target if target else 0))

        # Early exit
        if target == 0:
            self.set_dataframe(pd.DataFrame(), data_type='df')
            self.data_path = parquet_path
            self.data_type = 'df'
            return {'rows': 0, 'total_rows': total_rows, 'path': parquet_path}

        # Build cumulative row-group boundaries
        cum = np.cumsum([0] + rg_rows)  # length num_rgs+1, cum[i] is start of rg i

        dfs = []
        if random:
            rng = np.random.default_rng(random_state)
            global_idx = np.sort(rng.choice(total_rows, size=target, replace=False))

            # Map global indices to per-row-group local indices
            # For each rg, gather indices that fall into [cum[i], cum[i+1])
            if show_progress:
                pbar = tqdm(total=target, desc="Sampling rows (random)")
            pos = 0
            for i in range(num_rgs):
                start, end = cum[i], cum[i+1]
                # slice global_idx in this interval
                while pos < len(global_idx) and global_idx[pos] < start:
                    pos += 1
                j = pos
                while j < len(global_idx) and global_idx[j] < end:
                    j += 1
                if j > pos:
                    local = global_idx[pos:j] - start
                    table = pf.read_row_group(i, columns=columns)
                    df_rg = table.to_pandas()
                    dfs.append(df_rg.iloc[local])
                    if show_progress:
                        pbar.update(j - pos)
                pos = j
            if show_progress:
                pbar.close()
        else:
            remaining = target
            it = range(num_rgs)
            if show_progress:
                it = tqdm(it, desc="Reading rows (sequential)")
            for i in it:
                to_take = min(remaining, rg_rows[i])
                if to_take <= 0:
                    break
                table = pf.read_row_group(i, columns=columns)
                df_rg = table.to_pandas()
                dfs.append(df_rg.head(to_take))
                remaining -= to_take

        df_out = pd.concat(dfs, axis=0, ignore_index=True) if dfs else pd.DataFrame()
        self.set_dataframe(df_out, data_type='df')
        self.data_path = parquet_path
        self.data_type = 'df'
        return {'rows': int(len(df_out)), 'total_rows': total_rows, 'path': parquet_path}
    
    
    def get_sliced_dataframe(self, start_row=None, end_row=None, start_column=None, end_column=None):
        return self.get_dataframe().loc[start_row:end_row, start_column: end_column]
    
    
    
    def anomaly_filter_parquet(
        self,
        column_name: str,
        method: str | None = "zscore",
        *,
        threshold: float = 3.0,          # zscore / robust_zscore
        iqr_factor: float = 1.5,         # iqr fences
        mad_thresh: float = 3.5,         # mad / robust_zscore
        output_folder: str = "parquet_no_outliers",
        overwrite: bool = False,
        show_progress: bool = True,
        sample_limit: int = 1_000_000_000    # for stats estimation
    ):
        """
        Stream/filter OUTlier rows from a (large) parquet dataset (or in‑memory df) and
        write a new parquet dataset/file WITHOUT anomalies.

        Keeps only non‑anomalous rows (inverse of detect_anomaly_parquet output).

        Supported methods (same logic as detect_anomaly_parquet):
            'zscore'        : |(x-mean)/std| > threshold
            'robust_zscore' : |(x-median)/(1.4826*MAD)| > threshold
            'iqr'           : outside [Q1 - iqr_factor*IQR, Q3 + iqr_factor*IQR]
            'mad'           : |x - median| / (1.4826 * MAD) > mad_thresh
            callable        : custom(values: np.ndarray)-> bool mask (True = anomaly)

        Params:
            column_name   : numeric column to analyze
            method        : str or callable
            threshold     : for zscore / robust_zscore
            iqr_factor    : for iqr method
            mad_thresh    : for mad / robust_zscore
            output_folder : destination root (dir or single .parquet file if source is single file)
            overwrite     : allow overwrite
            show_progress : tqdm bars
            sample_limit  : max values to sample for stats estimation (iqr/mad/robust)
        Returns:
            dict summary
        """
        import os, numpy as np, pandas as pd
        from tqdm import tqdm

        # Helper: build anomaly mask for a values array given meta stats
        def mask_anomalies(values: np.ndarray, meta: dict):
            if callable(method):
                m = method(values)
                if not isinstance(m, np.ndarray) or m.dtype != bool or m.shape != values.shape:
                    raise ValueError("Custom method must return boolean np.ndarray same shape (True=outlier).")
                return m
            mm = str(method).lower()
            if mm == "zscore":
                mean, std = meta.get("mean"), meta.get("std")
                if std in (0, None) or np.isnan(std):
                    return np.zeros_like(values, dtype=bool)
                z = (values - mean) / std
                return np.abs(z) > threshold
            if mm in ("robust_zscore", "mad"):
                med, mad = meta.get("median"), meta.get("mad")
                if mad in (0, None) or np.isnan(mad):
                    return np.zeros_like(values, dtype=bool)
                rz = (values - med) / (1.4826 * mad)
                limit = threshold if mm == "robust_zscore" else mad_thresh
                return np.abs(rz) > limit
            if mm == "iqr":
                q1, q3 = meta.get("q1"), meta.get("q3")
                if q1 is None or q3 is None:
                    return np.zeros_like(values, dtype=bool)
                iqr = q3 - q1
                lower = q1 - iqr_factor * iqr
                upper = q3 + iqr_factor * iqr
                return (values < lower) | (values > upper)
            raise ValueError(f"Unsupported method {method}")

        # ---------- In‑memory pandas path ----------
        if self.data_type != 'parquet' or not hasattr(self, 'dataset'):
            if column_name not in self.dataframe.columns:
                raise ValueError(f"Column '{column_name}' not found.")
            ser = pd.to_numeric(self.dataframe[column_name], errors="coerce")
            vals = ser.to_numpy(dtype=float)

            meta = {}
            if not callable(method):
                m = str(method).lower()
                if m == "zscore":
                    meta["mean"] = float(np.nanmean(vals))
                    meta["std"] = float(np.nanstd(vals, ddof=0))
                elif m in ("mad", "robust_zscore"):
                    med = float(np.nanmedian(vals))
                    mad = float(np.nanmedian(np.abs(vals - med)))
                    meta.update(median=med, mad=mad)
                elif m == "iqr":
                    q1, q3 = np.nanpercentile(vals, [25, 75])
                    meta.update(q1=float(q1), q3=float(q3))
                else:
                    raise ValueError(f"Unsupported method: {method}")

            anomaly_mask = mask_anomalies(vals, meta)
            anomaly_mask = np.where(np.isnan(vals), False, anomaly_mask)
            keep_mask = ~anomaly_mask

            filtered_df = self.dataframe.loc[keep_mask].copy()

            # Write parquet (single file) if requested
            os.makedirs(output_folder, exist_ok=True)
            out_file = os.path.join(output_folder, "filtered_no_outliers.parquet")
            filtered_df.to_parquet(out_file, index=False)

            return {
                "mode": "pandas",
                "rows_in": int(len(self.dataframe)),
                "rows_out": int(len(filtered_df)),
                "removed": int(anomaly_mask.sum()),
                "output_root": output_folder,
                "output_file": out_file,
                "method": "custom" if callable(method) else str(method)
            }

        # ---------- Parquet (stream) path ----------
        import pyarrow.parquet as pq, pyarrow as pa

        schema = self.dataset.schema
        if column_name not in schema.names:
            raise ValueError(f"Column '{column_name}' not found in parquet schema.")

        is_single_file = os.path.isfile(self.data_path) and self.data_path.lower().endswith(".parquet")

        # Handle output path
        output_folder = os.path.abspath(output_folder)
        if os.path.exists(output_folder):
            if not overwrite:
                raise FileExistsError(f"{output_folder} exists. Set overwrite=True.")

        if is_single_file and output_folder.lower().endswith(".parquet"):
            # Treat output_folder as file path
            out_is_file = True
            out_file_single = output_folder
            os.makedirs(os.path.dirname(out_file_single) or ".", exist_ok=True)
        elif is_single_file and output_folder.lower().endswith(".parquet") is False:
            out_is_file = True
            os.makedirs(output_folder, exist_ok=True)
            base = os.path.splitext(os.path.basename(self.data_path))[0]
            out_file_single = os.path.join(output_folder, f"{base}_no_outliers.parquet")
        else:
            out_is_file = False
            os.makedirs(output_folder, exist_ok=True)

        # -------- Stats estimation (first pass if needed) --------
        meta = {}
        if not callable(method):
            mm = str(method).lower()
            fragments = list(self.dataset.get_fragments())
            if mm == "zscore":
                n = 0
                mean = 0.0
                M2 = 0.0
                frag_bar = tqdm(fragments, desc="Stats pass (zscore)", disable=not show_progress)
                for frag in frag_bar:
                    pf = pq.ParquetFile(frag.path)
                    for rg_idx in range(pf.num_row_groups):
                        arr = pf.read_row_group(rg_idx, columns=[column_name])[column_name]
                        if arr.null_count == arr.length():
                            continue
                        vals = arr.to_numpy(zero_copy_only=False)
                        vals = vals[~np.isnan(vals)]
                        if not vals.size:
                            continue
                        for x in vals:
                            n += 1
                            d = x - mean
                            mean += d / n
                            d2 = x - mean
                            M2 += d * d2
                std = float(np.sqrt(M2 / n)) if n else float("nan")
                meta.update(mean=float(mean), std=std)
            elif mm in ("mad", "robust_zscore"):
                collected = 0
                samples = []
                frag_bar = tqdm(self.dataset.get_fragments(), desc="Stats pass (mad)", disable=not show_progress)
                for frag in frag_bar:
                    if collected >= sample_limit:
                        break
                    pf = pq.ParquetFile(frag.path)
                    for rg_idx in range(pf.num_row_groups):
                        if collected >= sample_limit:
                            break
                        arr = pf.read_row_group(rg_idx, columns=[column_name])[column_name]
                        if arr.null_count == arr.length():
                            continue
                        vals = arr.to_numpy(zero_copy_only=False)
                        vals = vals[~np.isnan(vals)]
                        if not vals.size:
                            continue
                        take = min(vals.size, sample_limit - collected)
                        samples.append(vals[:take])
                        collected += take
                if collected:
                    samp = np.concatenate(samples, axis=0)
                    med = float(np.median(samp))
                    mad = float(np.median(np.abs(samp - med)))
                    meta.update(median=med, mad=mad)
                else:
                    meta.update(median=float("nan"), mad=float("nan"))
            elif mm == "iqr":
                collected = 0
                samples = []
                frag_bar = tqdm(self.dataset.get_fragments(), desc="Stats pass (iqr)", disable=not show_progress)
                for frag in frag_bar:
                    if collected >= sample_limit:
                        break
                    pf = pq.ParquetFile(frag.path)
                    for rg_idx in range(pf.num_row_groups):
                        if collected >= sample_limit:
                            break
                        arr = pf.read_row_group(rg_idx, columns=[column_name])[column_name]
                        if arr.null_count == arr.length():
                            continue
                        vals = arr.to_numpy(zero_copy_only=False)
                        vals = vals[~np.isnan(vals)]
                        if not vals.size:
                            continue
                        take = min(vals.size, sample_limit - collected)
                        samples.append(vals[:take])
                        collected += take
                if collected:
                    samp = np.concatenate(samples, axis=0)
                    q1, q3 = np.percentile(samp, [25, 75])
                    meta.update(q1=float(q1), q3=float(q3))
                else:
                    meta.update(q1=float("nan"), q3=float("nan"))
            else:
                raise ValueError(f"Unsupported method {method}")

        # -------- Second pass: filter & write --------
        total_in = 0
        total_out = 0
        total_removed = 0

        if out_is_file:
            writer_single = None
        else:
            writer_single = None  # not used; per-fragment writers

        fragments2 = list(self.dataset.get_fragments())
        frag_bar2 = tqdm(fragments2, desc="Filtering", disable=not show_progress)
        for frag in frag_bar2:
            pf = pq.ParquetFile(frag.path)

            # Prepare output per fragment (if dataset mode)
            if not out_is_file:
                root_abs = os.path.abspath(self.data_path)
                frag_rel = frag.path if os.path.isabs(frag.path) else os.path.relpath(frag.path, root_abs)
                out_dir = os.path.join(output_folder, os.path.dirname(frag_rel))
                os.makedirs(out_dir, exist_ok=True)
                out_file = os.path.join(out_dir, os.path.basename(frag_rel))
                writer_fragment = None

            rg_iter = tqdm(
                range(pf.num_row_groups),
                desc=f"RowGroups:{os.path.basename(frag.path)}",
                leave=False,
                disable=not show_progress
            ) if show_progress else range(pf.num_row_groups)

            for rg_idx in rg_iter:
                rg_tbl = pf.read_row_group(rg_idx, use_threads=True)
                if column_name not in rg_tbl.column_names:
                    continue
                col_arr = rg_tbl[column_name].to_numpy(zero_copy_only=False).astype(float)
                nrg = col_arr.shape[0]
                total_in += nrg
                if nrg == 0:
                    continue

                mask_out = mask_anomalies(col_arr, meta)
                mask_out = np.where(np.isnan(col_arr), False, mask_out)

                if not mask_out.any():
                    kept_tbl = rg_tbl
                elif mask_out.all():
                    total_removed += nrg
                    continue
                else:
                    keep_mask = ~mask_out
                    total_removed += int(mask_out.sum())
                    kept_tbl = rg_tbl.filter(pa.array(keep_mask.tolist()))

                if kept_tbl.num_rows == 0:
                    continue

                total_out += kept_tbl.num_rows

                if out_is_file:
                    if writer_single is None:
                        writer_single = pq.ParquetWriter(
                            out_file_single,
                            kept_tbl.schema,
                            compression="zstd",
                            use_dictionary=True,
                            write_statistics=True
                        )
                    writer_single.write_table(kept_tbl)
                else:
                    if writer_fragment is None:
                        writer_fragment = pq.ParquetWriter(
                            out_file,
                            kept_tbl.schema,
                            compression="zstd",
                            use_dictionary=True,
                            write_statistics=True
                        )
                    writer_fragment.write_table(kept_tbl)

            if not out_is_file and 'writer_fragment' in locals() and writer_fragment is not None:
                writer_fragment.close()

            if show_progress and total_in:
                frag_bar2.set_postfix(kept=total_out, removed=total_removed)

        if writer_single is not None:
            writer_single.close()

        if total_out == 0:
            # write empty placeholder
            empty_tbl = pa.table({c: pa.array([], type=schema.field(c).type) for c in schema.names})
            if out_is_file:
                pq.write_table(empty_tbl, out_file_single, compression="zstd")
            else:
                pq.write_table(empty_tbl, os.path.join(output_folder, "empty.parquet"), compression="zstd")
                
        print(f"✅ Filtered dataset written to {output_folder} (in: {total_in:,}, out: {total_out:,}, removed: {total_removed:,}).")

        return {
            "mode": "parquet_single_file" if is_single_file else "parquet_dataset",
            "rows_in": int(total_in),
            "rows_out": int(total_out),
            "removed": int(total_removed),
            "removed_pct": round((total_removed / total_in * 100) if total_in else 0.0, 3),
            "output_root": output_folder if not out_is_file else os.path.dirname(out_file_single),
            "output_file": out_file_single if out_is_file else None,
            "method": "custom" if callable(method) else str(method),
            "stats_meta": meta
        }
    
    
    
    def drop_columns_parquet(
        self,
        columns_to_drop: list[str],
        output_path: str,
        overwrite: bool = False,
        compression: str = "zstd",
        preserve_partitions: bool = True,
        show_progress: bool = True,
        row_group_size: int | None = None,
    ):
        """
        Drop multiple columns from a parquet dataset or single parquet file.

        Args:
            columns_to_drop : list of column names to remove.
            output_path     : If source is a single file:
                                - If endswith .parquet -> exact output file.
                                - Else treated as directory; file written as <name>_dropped.parquet
                              If source is a dataset directory:
                                - Should be a (new) directory root for rewritten dataset.
            overwrite       : Overwrite existing output_path (file or directory).
            compression     : Parquet compression codec.
            preserve_partitions : Keep original partition folder structure (dataset mode).
            show_progress   : tqdm progress bars.
            row_group_size  : Optional row group size (rows) for output (None = write incoming RGs as-is).

        Returns:
            dict summary
        """
        import os
        import pyarrow as pa
        import pyarrow.parquet as pq
        from tqdm import tqdm

        if self.data_type != "parquet" or not hasattr(self, "dataset"):
            raise ValueError("drop_colmns_parquet only supports parquet datasets / files.")

        if not columns_to_drop:
            raise ValueError("columns_to_drop must be a non-empty list.")

        # Detect single file vs dataset
        is_single_file = os.path.isfile(self.data_path) and self.data_path.lower().endswith(".parquet")

        # Resolve schema
        if is_single_file:
            pf0 = pq.ParquetFile(self.data_path)
            schema = pf0.schema_arrow
            source_columns = list(schema.names)
        else:
            schema = self.dataset.schema
            source_columns = list(schema.names)

        missing = [c for c in columns_to_drop if c not in source_columns]
        if missing:
            print(f"⚠️ Columns not found (will skip): {missing}")
        drop_set = set([c for c in columns_to_drop if c in source_columns])

        keep_cols = [c for c in source_columns if c not in drop_set]
        if not keep_cols:
            raise ValueError("Cannot drop all columns; at least one column must remain.")

        # Handle output paths
        if is_single_file:
            if output_path.lower().endswith(".parquet"):
                out_file = output_path
                out_dir = os.path.dirname(out_file) or "."
            else:
                # Directory provided
                os.makedirs(output_path, exist_ok=True)
                base = os.path.splitext(os.path.basename(self.data_path))[0]
                out_file = os.path.join(output_path, f"{base}_dropped.parquet")
                out_dir = output_path
            if os.path.exists(out_file) and not overwrite:
                raise FileExistsError(f"{out_file} exists (overwrite=False).")
            os.makedirs(out_dir, exist_ok=True)
        else:
            # Dataset
            out_root = output_path
            if os.path.exists(out_root):
                if not overwrite:
                    raise FileExistsError(f"{out_root} exists (overwrite=False).")
            os.makedirs(out_root, exist_ok=True)

        total_rows_in = 0
        total_rows_out = 0
        total_files = 0
        total_row_groups = 0

        if is_single_file:
            pf = pq.ParquetFile(self.data_path)
            writer = None
            rg_iter = tqdm(range(pf.num_row_groups), desc="RowGroups", disable=not show_progress)
            for rg_idx in rg_iter:
                rg_tbl = pf.read_row_group(rg_idx, columns=keep_cols, use_threads=True)
                if writer is None:
                    writer = pq.ParquetWriter(
                        out_file,
                        rg_tbl.schema,
                        compression=compression,
                        use_dictionary=True,
                        write_statistics=True
                    )
                if row_group_size:
                    n = rg_tbl.num_rows
                    start = 0
                    while start < n:
                        end = min(start + row_group_size, n)
                        writer.write_table(rg_tbl.slice(start, end - start))
                        start = end
                else:
                    writer.write_table(rg_tbl)
                total_rows_in += pf.metadata.row_group(rg_idx).num_rows
                total_rows_out += rg_tbl.num_rows
                total_row_groups += 1
                if show_progress:
                    rg_iter.set_postfix(rows=f"{total_rows_out:,}")
            if writer is not None:
                writer.close()
            total_files = 1
            summary = {
                "mode": "single_file",
                "output_file": out_file,
                "rows_in": total_rows_in,
                "rows_out": total_rows_out,
                "dropped_columns": list(drop_set),
                "kept_columns": keep_cols,
                "row_groups_written": total_row_groups,
                "compression": compression
            }
        else:
            # Dataset mode
            root_abs = os.path.abspath(self.data_path)
            fragments = list(self.dataset.get_fragments())
            frag_bar = tqdm(fragments, desc="Fragments", disable=not show_progress)

            for fragment in frag_bar:
                frag_rel = fragment.path
                # Determine output file path
                if os.path.isabs(frag_rel):
                    rel_path = os.path.relpath(frag_rel, root_abs)
                else:
                    rel_path = frag_rel
                if preserve_partitions:
                    rel_dir = os.path.dirname(rel_path)
                    out_dir = os.path.join(out_root, rel_dir)
                else:
                    out_dir = out_root
                os.makedirs(out_dir, exist_ok=True)

                src_abs = frag_rel if os.path.isabs(frag_rel) else os.path.join(root_abs, frag_rel)
                try:
                    pf = pq.ParquetFile(src_abs)
                except Exception as e:
                    print(f"⚠️ Skip unreadable fragment {frag_rel}: {e}")
                    continue

                out_file = os.path.join(out_dir, os.path.basename(rel_path))
                writer = None
                for rg_idx in range(pf.num_row_groups):
                    rg_tbl = pf.read_row_group(rg_idx, columns=keep_cols, use_threads=True)
                    if writer is None:
                        writer = pq.ParquetWriter(
                            out_file,
                            rg_tbl.schema,
                            compression=compression,
                            use_dictionary=True,
                            write_statistics=True
                        )
                    if row_group_size:
                        n = rg_tbl.num_rows
                        start = 0
                        while start < n:
                            end = min(start + row_group_size, n)
                            writer.write_table(rg_tbl.slice(start, end - start))
                            start = end
                    else:
                        writer.write_table(rg_tbl)

                    total_rows_in += pf.metadata.row_group(rg_idx).num_rows
                    total_rows_out += rg_tbl.num_rows
                    total_row_groups += 1

                if writer is not None:
                    writer.close()
                    total_files += 1

                if show_progress:
                    frag_bar.set_postfix(files=total_files, rows=f"{total_rows_out:,}")

            summary = {
                "mode": "dataset",
                "output_root": out_root,
                "files_written": total_files,
                "rows_in": total_rows_in,
                "rows_out": total_rows_out,
                "dropped_columns": list(drop_set),
                "kept_columns": keep_cols,
                "row_groups_written": total_row_groups,
                "compression": compression,
                "preserve_partitions": preserve_partitions
            }

        # If user wants to point current object to new data (optional pattern):
        # self.data_path = summary.get("output_file", summary.get("output_root", self.data_path))
        # try: self.dataset = ds.dataset(self.data_path, format="parquet")
        # except: pass

        print(f"✅ drop_colmns_parquet complete | Dropped {len(drop_set)} | Kept {len(keep_cols)} | Rows: {total_rows_out:,}")
        print(f"   Output: {out_file if is_single_file else out_root}")
        return summary
    
    
    def detect_anomaly_parquet(
        self,
        column_name: str,
        method: str | None = "zscore",
        *,
        threshold: float = 3.0,        # for zscore / robust_zscore
        iqr_factor: float = 1.5,       # for iqr
        mad_thresh: float = 3.5,       # for mad (robust z)
        save: bool = False,
        save_path: str = "anomalies.parquet",
        overwrite: bool = False,
        return_df: bool = True,
        show_progress: bool = True,
        sample_limit: int = 1_000_000_000, # iqr/mad/robust_zscore sampling
        max_return_rows: int = 1_000_000_000
    ):
        """
        Detect anomalies in a column and collect full rows where anomalies occur.
        Works with pandas (in-memory) and parquet datasets (streaming).

        method:
            'zscore'        : |(x-mean)/std| > threshold
            'robust_zscore' : |(x-median)/(1.4826*MAD)| > threshold   (new)
            'iqr'           : outside [Q1 - iqr_factor*IQR, Q3 + iqr_factor*IQR]
            'mad'           : |x - median| / (1.4826 * MAD) > mad_thresh
            callable        : custom(values: np.ndarray)-> bool mask

        Saves to .parquet or .csv depending on save_path extension.
        """

        # Helper for anomaly mask
        def make_mask(values: np.ndarray, meta: dict):
            nonlocal method, threshold, iqr_factor, mad_thresh
            if callable(method):
                return method(values)
            m = str(method).lower()
            if m == "zscore":
                mean = meta.get("mean"); std = meta.get("std")
                if std in (None, 0) or np.isnan(std):
                    return np.zeros_like(values, dtype=bool)
                z = (values - mean) / std
                return np.abs(z) > threshold
            if m == "robust_zscore":
                med = meta.get("median"); mad = meta.get("mad")
                if mad in (None, 0) or np.isnan(mad):
                    return np.zeros_like(values, dtype=bool)
                rz = (values - med) / (1.4826 * mad)
                return np.abs(rz) > threshold
            if m == "iqr":
                q1 = meta.get("q1"); q3 = meta.get("q3")
                iqr = (q3 - q1) if (q1 is not None and q3 is not None) else np.nan
                if np.isnan(iqr):
                    return np.zeros_like(values, dtype=bool)
                lower = q1 - iqr_factor * iqr
                upper = q3 + iqr_factor * iqr
                return (values < lower) | (values > upper)
            if m == "mad":
                med = meta.get("median"); mad = meta.get("mad")
                if mad in (None, 0) or np.isnan(mad):
                    return np.zeros_like(values, dtype=bool)
                robust_z = np.abs(values - med) / (1.4826 * mad)
                return robust_z > mad_thresh
            if m == "isolation_forest":
                model = meta.get("iforest")
                if model is None:
                    return np.zeros_like(values, dtype=bool)
                finite = np.isfinite(values)
                out = np.zeros_like(values, dtype=bool)
                if finite.any():
                    preds = model.predict(values[finite].reshape(-1, 1))  # -1 = outlier
                    out[finite] = (preds == -1)
                return out
            
            raise ValueError(f"Unsupported method: {method}")

        # In-memory pandas path
        if self.data_type != 'parquet' or not hasattr(self, 'dataset'):
            if column_name not in self.dataframe.columns:
                raise ValueError(f"Column '{column_name}' not found.")
            ser = pd.to_numeric(self.dataframe[column_name], errors="coerce")
            num = ser.to_numpy(dtype=float)

            if callable(method):
                mask = method(num)
                if not isinstance(mask, np.ndarray) or mask.dtype != bool or mask.shape[0] != num.shape[0]:
                    raise ValueError("Custom method must return a boolean mask of same length.")
            else:
                m = str(method).lower()
                if m == "zscore":
                    meta = dict(mean=float(np.nanmean(num)), std=float(np.nanstd(num, ddof=0)))
                elif m in ("mad", "robust_zscore"):
                    med = float(np.nanmedian(num))
                    mad = float(np.nanmedian(np.abs(num - med)))
                    meta = dict(median=med, mad=mad)
                elif m == "iqr":
                    q1, q3 = np.nanpercentile(num, [25, 75])
                    meta = dict(q1=float(q1), q3=float(q3))
                elif m == "isolation_forest":
                    try:
                        from sklearn.ensemble import IsolationForest
                    except ImportError:
                        raise ImportError("scikit-learn is required for method='isolation_forest' (pip install scikit-learn).")
                    finite = np.isfinite(num)
                    contamination = threshold if 0 < threshold < 0.5 else 0.01
                    if finite.any():
                        iforest = IsolationForest(
                            n_estimators=200,
                            contamination=contamination,
                            random_state=42,
                            n_jobs=-1,
                            verbose=0
                        ).fit(num[finite].reshape(-1, 1))
                        meta = {"iforest": iforest, "contamination": contamination}
                    else:
                        meta = {"iforest": None, "contamination": contamination}
                else:
                    raise ValueError(f"Unsupported method: {method}")
                mask = make_mask(num, meta)
                mask = np.where(np.isnan(num), False, mask)

            anomalies_df = self.dataframe.loc[mask].copy()

            if save:
                ext = os.path.splitext(save_path)[1].lower()
                if os.path.exists(save_path) and not overwrite:
                    raise FileExistsError(f"{save_path} exists. Set overwrite=True.")
                if ext == ".csv":
                    anomalies_df.to_csv(save_path, index=False)
                else:
                    anomalies_df.to_parquet(save_path, index=False)

            return anomalies_df if return_df else {
                "rows_scanned": int(len(self.dataframe)),
                "rows_flagged": int(mask.sum()),
                "method": "custom" if callable(method) else str(method),
                "output_file": save_path if save else None
            }

        # Parquet streaming path
        schema = self.dataset.schema
        if column_name not in schema.names:
            raise ValueError(f"Column '{column_name}' not found in parquet schema.")

        if save:
            out_dir = os.path.dirname(save_path) or "."
            os.makedirs(out_dir, exist_ok=True)
            if os.path.exists(save_path) and not overwrite:
                raise FileExistsError(f"{save_path} exists. Set overwrite=True.")
        ext = os.path.splitext(save_path)[1].lower() if save else ".parquet"
        writer = None
        csv_header_written = False

        meta = {}
        m = method if callable(method) else str(method).lower()
        needs_stats = (not callable(method)) and m in {"zscore", "iqr", "mad", "robust_zscore", "isolation_forest"}

        fragments = list(self.dataset.get_fragments())
        if needs_stats:
            if m == "zscore":
                n = 0
                mean = 0.0
                M2 = 0.0
                frag_bar = tqdm(fragments, desc="Pass 1/2 (zscore): fragments", disable=not show_progress)
                for fragment in frag_bar:
                    pf = pq.ParquetFile(fragment.path)
                    rg_iter = tqdm(range(pf.num_row_groups), desc="RowGroups", leave=False, disable=not show_progress)
                    for rg_idx in rg_iter:
                        arr = pf.read_row_group(rg_idx, columns=[column_name])[column_name]
                        if arr.null_count == arr.length():
                            continue
                        vals = arr.to_numpy(zero_copy_only=False)
                        vals = vals[~np.isnan(vals)]
                        if vals.size == 0:
                            continue
                        for x in vals:
                            n += 1
                            delta = x - mean
                            mean += delta / n
                            delta2 = x - mean
                            M2 += delta * delta2
                        rg_iter.set_postfix(n=n)
                std = float(np.sqrt(M2 / n)) if n > 0 else float("nan")
                meta.update(mean=float(mean), std=std)
            elif m in {"mad", "robust_zscore"}:
                collected = 0
                samples = []
                frag_bar = tqdm(fragments, desc=f"Pass 1/2 ({m}): fragments", disable=not show_progress)
                for fragment in frag_bar:
                    if collected >= sample_limit:
                        break
                    pf = pq.ParquetFile(fragment.path)
                    rg_iter = tqdm(range(pf.num_row_groups), desc="RowGroups", leave=False, disable=not show_progress)
                    for rg_idx in rg_iter:
                        if collected >= sample_limit:
                            break
                        arr = pf.read_row_group(rg_idx, columns=[column_name])[column_name]
                        if arr.null_count == arr.length():
                            continue
                        vals = arr.to_numpy(zero_copy_only=False)
                        vals = vals[~np.isnan(vals)]
                        if vals.size == 0:
                            continue
                        take = min(vals.size, sample_limit - collected)
                        samples.append(vals[:take])
                        collected += take
                        rg_iter.set_postfix(collected=collected)
                    frag_bar.set_postfix(collected=collected)
                if collected == 0:
                    meta.update(median=float("nan"), mad=float("nan"))
                else:
                    samp = np.concatenate(samples, axis=0)
                    med = float(np.median(samp))
                    mad = float(np.median(np.abs(samp - med)))
                    meta.update(median=med, mad=mad)
            
            elif m == "isolation_forest":
                try:
                    from sklearn.ensemble import IsolationForest
                except ImportError:
                    raise ImportError("scikit-learn is required for method='isolation_forest' (pip install scikit-learn).")

                # Hard cap to prevent accidental OOM
                hard_cap = 5_000_000
                if sample_limit is None or sample_limit <= 0:
                    sample_limit = 2_000_000
                if sample_limit > hard_cap:
                    print(f"⚠️ sample_limit={sample_limit:,} too large; capping to {hard_cap:,} to avoid OOM.")
                    sample_limit = hard_cap

                collected = 0
                samples = []
                frag_bar = tqdm(fragments, desc="Pass 1/2 (IsolationForest): fragments", disable=not show_progress)
                for fragment in frag_bar:
                    if collected >= sample_limit:
                        break
                    pf = pq.ParquetFile(fragment.path)
                    rg_iter = tqdm(range(pf.num_row_groups), desc="RowGroups", leave=False, disable=not show_progress)
                    for rg_idx in rg_iter:
                        if collected >= sample_limit:
                            break
                        arr = pf.read_row_group(rg_idx, columns=[column_name])[column_name]
                        if arr.null_count == arr.length():
                            continue
                        # Use float32 to halve memory footprint
                        vals = arr.to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
                        vals = vals[np.isfinite(vals)]
                        if vals.size == 0:
                            continue
                        # Only take up to remaining quota from this row-group
                        take = min(vals.size, sample_limit - collected)
                        if take < vals.size:
                            # Randomly subsample this chunk if it is larger than remaining limit
                            idx = np.random.default_rng(42).choice(vals.size, size=take, replace=False)
                            chunk = vals[idx]
                        else:
                            chunk = vals
                        samples.append(chunk)
                        collected += chunk.size
                        rg_iter.set_postfix(collected=collected)
                    frag_bar.set_postfix(collected=collected)

                contamination = threshold if 0 < threshold < 0.5 else 0.01
                if collected == 0:
                    meta.update(iforest=None, contamination=contamination)
                else:
                    train_vals = np.concatenate(samples, axis=0).astype(np.float32, copy=False).reshape(-1, 1)
                    # Limit max_samples to keep training memory reasonable
                    max_samples = int(min(train_vals.shape[0], 200_000))
                    try:
                        iforest = IsolationForest(
                            n_estimators=200,
                            contamination=contamination,
                            max_samples=max_samples,   # control memory/time
                            random_state=42,
                            n_jobs=-1,
                            verbose=0
                        ).fit(train_vals)
                    except (MemoryError, np.core._exceptions._ArrayMemoryError):
                        # Fallback: downsample further and retry
                        print("⚠️ IsolationForest fit ran out of memory; retrying with smaller sample.")
                        if train_vals.shape[0] > 100_000:
                            idx = np.random.default_rng(42).choice(train_vals.shape[0], size=100_000, replace=False)
                            train_vals = train_vals[idx]
                            iforest = IsolationForest(
                                n_estimators=200,
                                contamination=contamination,
                                max_samples=min(100_000, train_vals.shape[0]),
                                random_state=42,
                                n_jobs=-1,
                                verbose=0
                            ).fit(train_vals)
                        else:
                            # If even this fails, disable IF and continue
                            print("⚠️ IsolationForest disabled due to memory constraints.")
                            iforest = None
                    print(f"   IsolationForest trained on {train_vals.shape[0]:,} samples (float32) with contamination={contamination}.")
                    meta.update(iforest=iforest, contamination=contamination)
            
            else:  # iqr
                collected = 0
                samples = []
                frag_bar = tqdm(fragments, desc="Pass 1/2 (iqr): fragments", disable=not show_progress)
                for fragment in frag_bar:
                    if collected >= sample_limit:
                        break
                    pf = pq.ParquetFile(fragment.path)
                    rg_iter = tqdm(range(pf.num_row_groups), desc="RowGroups", leave=False, disable=not show_progress)
                    for rg_idx in rg_iter:
                        if collected >= sample_limit:
                            break
                        arr = pf.read_row_group(rg_idx, columns=[column_name])[column_name]
                        if arr.null_count == arr.length():
                            continue
                        vals = arr.to_numpy(zero_copy_only=False)
                        vals = vals[~np.isnan(vals)]
                        if vals.size == 0:
                            continue
                        take = min(vals.size, sample_limit - collected)
                        samples.append(vals[:take])
                        collected += take
                        rg_iter.set_postfix(collected=collected)
                    frag_bar.set_postfix(collected=collected)
                if collected == 0:
                    meta.update(q1=float("nan"), q3=float("nan"))
                else:
                    samp = np.concatenate(samples, axis=0)
                    q1, q3 = np.percentile(samp, [25, 75])
                    meta.update(q1=float(q1), q3=float(q3))

        # Second pass unchanged except robust_zscore handled in make_mask
        anomalies_collected = []
        total_scanned = 0
        total_flagged = 0

        frag_bar2 = tqdm(fragments, desc="Pass 2/2: scan fragments", disable=not show_progress)
        for fragment in frag_bar2:
            pf = pq.ParquetFile(fragment.path)
            rg_iter = tqdm(range(pf.num_row_groups), desc="RowGroups", leave=False, disable=not show_progress)
            for rg_idx in rg_iter:
                col_tbl = pf.read_row_group(rg_idx, columns=[column_name], use_threads=True)
                col_arr = col_tbl[column_name]
                nrg = col_arr.length()
                if nrg == 0:
                    continue
                total_scanned += nrg

                vals = col_arr.to_numpy(zero_copy_only=False).astype(float)
                if callable(method):
                    mask = method(vals)
                    if not isinstance(mask, np.ndarray) or mask.dtype != bool or mask.shape[0] != vals.shape[0]:
                        raise ValueError("Custom method must return a boolean mask of same length.")
                else:
                    mask = make_mask(vals, meta)
                mask = np.where(np.isnan(vals), False, mask)

                if not mask.any():
                    rg_iter.set_postfix(scanned=total_scanned, flagged=total_flagged)
                    continue

                full_tbl = pf.read_row_group(rg_idx, use_threads=True)
                filtered = full_tbl.filter(pa.array(mask.tolist()))
                kept = filtered.num_rows
                total_flagged += kept

                if save:
                    if ext == ".csv":
                        df_chunk = filtered.to_pandas()
                        mode = "a" if csv_header_written else "w"
                        header = not csv_header_written
                        df_chunk.to_csv(save_path, index=False, mode=mode, header=header)
                        csv_header_written = True
                    else:
                        if writer is None:
                            writer = pq.ParquetWriter(save_path, filtered.schema, compression="zstd",
                                                      use_dictionary=True, write_statistics=True)
                        writer.write_table(filtered)

                if return_df and total_flagged <= max_return_rows:
                    anomalies_collected.append(filtered.to_pandas())

                rg_iter.set_postfix(scanned=total_scanned, flagged=total_flagged)
            frag_bar2.set_postfix(scanned=total_scanned, flagged=total_flagged)

        if writer is not None:
            writer.close()

        if return_df:
            if anomalies_collected:
                out_df = pd.concat(anomalies_collected, ignore_index=True)
            else:
                out_df = pd.DataFrame(columns=schema.names)
            if total_flagged > max_return_rows and save:
                print(f"Returned only first {max_return_rows:,} anomalies; full set saved to {save_path}.")
            return out_df

        print(f"✅ Scanned {total_scanned:,} rows, flagged {total_flagged:,} anomalies. Saved: {save_path if save else 'N/A'}")

        return {
            "rows_scanned": int(total_scanned),
            "rows_flagged": int(total_flagged),
            "method": "custom" if callable(method) else str(method),
            "output_file": save_path if save else None
        }
    
    def anomaly_checking(
        self,
        column_name: str | None = None,
        columns: list[str] | None = None,
        method: str = "iqr",          # 'iqr' | 'zscore'
        iqr_factor: float = 1.5,
        zscore_threshold: float = 3.0,
        sample_rows: int = 200_000,
        max_cols_plot: int = 30,
        bins: int = 50,
        kde: bool = True,
        figsize: tuple[int, int] = (5, 4),
        palette: str = "viridis",
        save_dir: str | None = None,
        heatmap: bool = True,
        show: bool = True,
        dpi: int = 120,
        progress: bool = True,
        return_stats: bool = True,
        seed: int = 42,
    ):
        """
        Visual anomaly (outlier) inspection for numeric columns.

        Works with parquet (stream-sampled) or in-memory dataframe.

        Args:
            column_name / columns : Target column(s). If both None -> all numeric columns.
            method : 'iqr' or 'zscore'.
            iqr_factor : IQR multiplier for fences.
            zscore_threshold : Absolute z-score threshold.
            sample_rows : Max rows sampled per column (parquet) / from in-memory df.
            max_cols_plot : Plot at most this many columns (extra skipped with notice).
            bins : Histogram bins.
            kde : Overlay KDE on histogram (if seaborn available).
            figsize : Base (w,h) per single-column figure (h doubled internally).
            palette : Matplotlib / seaborn palette for consistency.
            save_dir : If set, saves per-column PNGs + optional summary heatmap.
            heatmap : Generate outlier% heatmap (if <= max_cols_plot).
            show : Display figures (disable for batch).
            dpi : Figure DPI for saving.
            progress : Show tqdm bars.
            return_stats : Return pandas DataFrame with outlier stats.
            seed : Sampling seed.

        Returns:
            pandas.DataFrame with: column, rows, sampled, outliers, outlier_pct,
            lower_bound, upper_bound, method_meta (mean/std or q1/q3).
        """
        import numpy as np
        import pandas as pd
        import math
        import os
        from tqdm import tqdm
        import matplotlib.pyplot as plt

        try:
            import seaborn as sns
            sns.set_theme(style="whitegrid")
            have_sns = True
        except Exception:
            have_sns = False

        rng = np.random.default_rng(seed)

        # ---------- Resolve target columns ----------
        if  column_name and columns:
            target_cols = list(dict.fromkeys([column_name] + columns))
        elif column_name:
            target_cols = [column_name]
        elif columns:
            target_cols = columns
        else:
            if self.data_type == 'parquet' and hasattr(self, 'dataset'):
                import pyarrow as pa
                numeric_types = (
                    pa.int8(), pa.int16(), pa.int32(), pa.int64(),
                    pa.uint8(), pa.uint16(), pa.uint32(), pa.uint64(),
                    pa.float16(), pa.float32(), pa.float64()
                )
                target_cols = [
                    f.name for f in self.dataset.schema
                    if any(f.type == t for t in numeric_types)
                ]
            else:
                target_cols = self.get_dataframe().select_dtypes(include=np.number).columns.tolist()

        if not target_cols:
            print("No numeric columns found.")
            return pd.DataFrame() if return_stats else None

        # Drop non-existent (safety)
        if self.data_type != 'parquet' or not hasattr(self, 'dataset'):
            target_cols = [c for c in target_cols if c in self.dataframe.columns]

        # Limit plotted columns
        plot_cols = target_cols[:max_cols_plot]
        skipped = len(target_cols) - len(plot_cols)
        if skipped > 0:
            print(f"Skipping plotting for {skipped} extra columns (showing first {max_cols_plot}).")

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        stats_rows = []

        # ---------- Helper: sample a column ----------
        def sample_column_parquet(col: str) -> np.ndarray:
            """Stream-sample up to sample_rows values from a parquet dataset column."""
            import pyarrow as pa
            import pyarrow.dataset as ds
            values = []
            taken = 0
            # Use a scanner for efficiency
            scanner = self.dataset.scanner(columns=[col], batch_size=64_000)
            for rec_batch in scanner.to_batches():
                arr = rec_batch.column(0).to_numpy(zero_copy_only=False)
                if arr.size == 0:
                    continue
                # Drop NaNs
                arr = arr[~np.isnan(arr)]
                if arr.size == 0:
                    continue
                if taken + arr.size <= sample_rows:
                    values.append(arr)
                    taken += arr.size
                else:
                    need = sample_rows - taken
                    if need > 0:
                        idx = rng.choice(arr.size, size=need, replace=False)
                        values.append(arr[idx])
                        taken += need
                    break
                if taken >= sample_rows:
                    break
            if not values:
                return np.array([], dtype=float)
            return np.concatenate(values)

        def sample_column_inmemory(col: str) -> np.ndarray:
            ser = pd.to_numeric(self.dataframe[col], errors="coerce").dropna()
            if ser.empty:
                return np.array([], dtype=float)
            if len(ser) > sample_rows:
                return ser.sample(sample_rows, random_state=seed).to_numpy()
            return ser.to_numpy()

        # ---------- Progress ----------
        iterator = tqdm(plot_cols, desc="Analyzing columns", disable=not progress)

        for col in iterator:
            # Sample
            if self.data_type == 'parquet' and hasattr(self, 'dataset'):
                data = sample_column_parquet(col)
            else:
                data = sample_column_inmemory(col)

            total_valid = int(data.size)
            if total_valid == 0:
                stats_rows.append({
                    "column": col,
                    "rows_sampled": 0,
                    "outliers": 0,
                    "outlier_pct": 0.0,
                    "lower_bound": np.nan,
                    "upper_bound": np.nan,
                    "method": method,
                    "method_meta": None
                })
                continue

            # Detect bounds + mask
            if method.lower() == "iqr":
                q1, q3 = np.percentile(data, [25, 75])
                iqr = q3 - q1
                lower = q1 - iqr_factor * iqr
                upper = q3 + iqr_factor * iqr
                mask = (data < lower) | (data > upper)
                meta = {"q1": q1, "q3": q3, "iqr": iqr}
            elif method.lower() == "zscore":
                mean = data.mean()
                std = data.std(ddof=0)
                if std == 0 or np.isnan(std):
                    mask = np.zeros_like(data, dtype=bool)
                else:
                    z = (data - mean) / std
                    mask = np.abs(z) > zscore_threshold
                lower = mean - zscore_threshold * std if std > 0 else np.nan
                upper = mean + zscore_threshold * std if std > 0 else np.nan
                meta = {"mean": mean, "std": std}
            else:
                raise ValueError("method must be 'iqr' or 'zscore'.")

            outliers = int(mask.sum())
            out_pct = (outliers / total_valid * 100) if total_valid else 0.0

            stats_rows.append({
                "column": col,
                "rows_sampled": total_valid,
                "outliers": outliers,
                "outlier_pct": round(out_pct, 3),
                "lower_bound": lower,
                "upper_bound": upper,
                "method": method.lower(),
                "method_meta": meta
            })

            # ---------- Plot ----------
            try:
                if show or save_dir:
                    fig, axes = plt.subplots(
                        2, 1,
                        figsize=(figsize[0], figsize[1] * 2),
                        gridspec_kw={"height_ratios": [1, 3]}
                    )

                    # Boxplot (horizontal)
                    axes[0].boxplot(data, vert=False, patch_artist=True,
                                    boxprops=dict(facecolor="#4C72B0", alpha=0.4))
                    axes[0].axvline(lower, color="red", ls="--", lw=1)
                    axes[0].axvline(upper, color="red", ls="--", lw=1)
                    axes[0].set_yticks([])
                    axes[0].set_title(f"{col} | outliers={outliers} ({out_pct:.2f}%)", fontsize=10)

                    # Histogram + KDE
                    if have_sns:
                        sns.histplot(data, bins=bins, kde=kde, ax=axes[1],
                                     color="#4C72B0", alpha=0.55, edgecolor="white", linewidth=0.5)
                    else:
                        axes[1].hist(data, bins=bins, color="#4C72B0", alpha=0.55)

                    # Shade outlier regions
                    x_min, x_max = data.min(), data.max()
                    axes[1].axvspan(x_min, lower, color="red", alpha=0.08)
                    axes[1].axvspan(upper, x_max, color="red", alpha=0.08)
                    axes[1].axvline(lower, color="red", ls="--", lw=1)
                    axes[1].axvline(upper, color="red", ls="--", lw=1)

                    # Overlay outlier points as a rug near baseline
                    if outliers:
                        out_data = data[mask]
                        y0 = axes[1].get_ylim()[0]
                        axes[1].scatter(out_data, np.full_like(out_data, y0 + (axes[1].get_ylim()[1]-y0)*0.01),
                                        s=8, c="red", alpha=0.6, marker='|', linewidths=1)

                    axes[1].set_xlabel(col)
                    axes[1].set_ylabel("Frequency")

                    plt.tight_layout()
                    if save_dir:
                        fig.savefig(os.path.join(save_dir, f"{col}_anomalies.png"), dpi=dpi)
                    if not show:
                        plt.close(fig)
            except Exception as e:
                print(f"Plot failed for {col}: {e}")

        stats_df = pd.DataFrame(stats_rows).sort_values("outlier_pct", ascending=False)

        # ---------- Heatmap ----------
        if heatmap and (show or save_dir) and not stats_df.empty:
            try:
                subset = stats_df.head(max_cols_plot).set_index("column")["outlier_pct"]
                fig, ax = plt.subplots(figsize=(min(1 + 0.25 * len(subset), 18), 4))
                if have_sns:
                    sns.heatmap(subset.to_frame().T, cmap=palette, annot=True, fmt=".2f",
                                cbar=True, ax=ax)
                else:
                    ax.imshow(subset.to_frame().T.values, aspect="auto")
                    ax.set_xticks(range(len(subset)))
                    ax.set_xticklabels(subset.index, rotation=45, ha="right", fontsize=8)
                    ax.set_yticks([0])
                    ax.set_yticklabels(["outlier_pct"])
                    for i, v in enumerate(subset.values):
                        ax.text(i, 0, f"{v:.2f}", ha="center", va="center", color="w", fontsize=7)
                ax.set_title("Outlier % (top columns)", fontsize=11)
                plt.tight_layout()
                if save_dir:
                    fig.savefig(os.path.join(save_dir, "outlier_summary_heatmap.png"), dpi=dpi)
                if not show:
                    plt.close(fig)
            except Exception as e:
                print(f"Heatmap failed: {e}")

        if return_stats:
            return stats_df

        return None
    
    
    def compare_two_time_series(self, time_series1_column_name, time_series2_column_name):
        from sklearn.metrics import r2_score, mean_squared_error 
        import math
        comparaison_dict = {}
        comparaison_dict['RMSE'] = math.sqrt(mean_squared_error(self.get_column(time_series1_column_name), self.get_column(time_series2_column_name)))
        comparaison_dict['R²'] = r2_score(self.get_column(time_series1_column_name), self.get_column(time_series2_column_name))
        comparaison_dict['R'] = self.get_column(time_series1_column_name).corr(self.get_column(time_series2_column_name))
        return comparaison_dict

    def eliminate_outliers_quantile(self, column, min_quantile, max_quantile):
        min_q, max_q = self.get_column(column).quantile(min_quantile), self.get_column(column).quantile(max_quantile)
        self.filter_dataframe(column, self.outliers_decision_function, min_q, max_q)

    def scale_column(self, column):
        max_column = self.get_column(column).describe()['max']
        self.transform_column(column, self.scale_trasform_fun, max_column)

    def drop_duplicated_rows(self, **kwargs):
        self.set_dataframe(self.dataframe.drop_duplicates(keep='first', **kwargs))
        
    def drop_duplicated_indexes(self, level=None,**kwargs):
        if level == 'd':
            self.dataframe = self.dataframe.groupby(self.dataframe.index.date).first()
        else:
            self.dataframe = self.dataframe[~self.dataframe.index.duplicated(keep='first', **kwargs)]
        
    def plot_dataframe(self, **kwarks):
        self.get_dataframe().plot(**kwarks)
        plt.show()
        
    def to_numpy(self):
        return self.get_dataframe().values
    
    @staticmethod
    def generate_datetime_range_dataframe(start_datetime='2013-01-01 00:00:00', 
                                          end_datetime='2013-12-31 00:00:00', 
                                          periods=None, 
                                          freq='1H'):
        dataframe = DataFrame()
        dataframe.set_dataframe_index(DataFrame.generate_datetime_range(
            starting_datetime=start_datetime, 
            end_datetime=end_datetime, 
            periods=periods,
            freq=freq))
        dataframe.rename_index('datetime')
        
        return dataframe.get_dataframe()
    
    def info(self):
        return self.get_dataframe().info()
    
    def drop_rows_by_year(self, year=2020, in_place=True):
        year = int(year)
        if in_place is True:
            self.set_dataframe(self.get_dataframe()[self.get_index(as_list=False).year != year]) 
        else:
            return self.get_dataframe()[self.get_index(as_list=False).year != year]
            
    def keep_rows_by_year(self, year=2020, in_place=True):
        year = int(year)
        if in_place is True:
            self.set_dataframe(self.get_dataframe()[self.get_index(as_list=False).year == year]) 
        else:
            return self.get_dataframe()[self.get_index(as_list=False).year == year]
        
    def train_test_split(self, train_percent=0.8):
        seuil = ceil(self.get_shape()[0]*train_percent)
        train = self.get_dataframe().iloc[:seuil]
        test = self.get_dataframe().iloc[seuil:]
        return train, test
    
    def train_test_split_column(self, column, train_percent=0.8):
        seuil = ceil(self.get_shape()[0]*train_percent)
        train = self.get_column(column).iloc[:seuil]
        test = self.get_column(column).iloc[seuil:]
        return train, test
    
    @staticmethod
    def get_elevation_and_latitude(lat, lon):
        """
        Returns the elevation (in meters) and latitude (in degrees) for a given set of coordinates.
        Uses the Open Elevation API (https://open-elevation.com/) to obtain the elevation information.
        """
        # 'https://api.open-elevation.com/api/v1/lookup?locations=10,10|20,20|41.161758,-8.583933'
        url = f'https://api.opentopodata.org/v1/aster30m?locations={lat},{lon}'
        response = requests.get(url)
        print(response.json())
        data = response.json()
        elevation = data['results'][0]['elevation']
        #latitude = data['results'][0]['latitude']
        return elevation
    
    @staticmethod
    def et0_penman_monteith(row):
        # input variables
        # T = 25.0  # air temperature in degrees Celsius
        # RH = 60.0  # relative humidity in percent
        # u2 = 2.0  # wind speed at 2 m height in m/s
        # Rs = 15.0  # incoming solar radiation in MJ/m2/day
        # lat = 35.0  # latitude in degrees
        
        ta_max, ta_min, rh_max, rh_min, u2_mean, rs_mean, lat, elevation, doy =  row['ta_max'], row['ta_min'], row['rh_max'], row['rh_min'], row['u2_mean'], row['rg_mean'], row['lat'], row['elevation'], row['doy']
        
        # constants
        ALBEDO = 0.23  # Albedo coefficient for grass reference surface
        GSC = 0.082  # solar constant in MJ/m2/min
        SIGMA = 0.000000004903  # Stefan-Boltzmann constant in MJ/K4/m2/day
        G = 0  # Soil heat flux density (MJ/m2/day)
        z = 2 # Convert wind speed measured at different heights above the soil surface to wind speed at 2 m above the surface, assuming a short grass surface.

        # convert units
        rs_mean *= 0.0864  # convert watts per square meter to megajoules per square meter 0.0288 = 60x60x8hours or 0.0864 for 24 hours
        ta_mean = (ta_max + ta_min) / 2
        ta_max_kelvin = ta_max + 273.16  # air temperature in Kelvin
        ta_min_kelvin = ta_min + 273.16  # air temperature in Kelvin
        
        # saturation vapor pressure in kPa
        es_max = 0.6108 * math.exp((17.27 * ta_max) / (ta_max + 237.3))
        es_min = 0.6108 * math.exp((17.27 * ta_min) / (ta_min + 237.3))
        es = (es_max + es_min) / 2
        
        # actual vapor pressure in kPa
        ea_max_term = es_max * (rh_min / 100)
        ea_min_term = es_min * (rh_max / 100)
        ea = (ea_max_term + ea_min_term) / 2
        
        # in the absence of rh_max and rh_min
        #ea = (rh_mean / 100) * es
        
        # when using equipement where errors in estimation rh min can be large or when rh data integrity are in doubt use only rh_max term
        #ea = ea_min_term  
        
        delta = (4098 * (0.6108 * math.exp((17.27 * ta_mean) / (ta_mean + 237.3)))) / math.pow((ta_mean + 237.3), 2) # slope of the vapor pressure curve in kPa/K
        
        
        atm_pressure = math.pow(((293.0 - (0.0065 * elevation)) / 293.0), 5.26) * 101.3
        # psychrometric constant in kPa/K
        gamma = 0.000665 * atm_pressure
        
        # Calculate u2
        u2 = u2_mean * (4.87 / math.log((67.8 * z) - 5.42))
        
        # Calculate extraterrestrial radiation
        dr = 1 + 0.033 * math.cos(2 * math.pi / 365 * doy)
        d = 0.409*math.sin( (((2*math.pi) / 365) * doy) - 1.39)
        
        
        # sunset hour angle
        phi = math.radians(lat)
        omega = math.acos(-math.tan(phi)*math.tan(d))
        ra = ((24 * 60)/math.pi) * GSC * dr * ((omega * math.sin(phi) * math.sin(d)) + (math.cos(phi) * math.cos(d) * math.sin(omega)))
        
        # Calculate clear sky solar radiation
        rso = (0.75 + (2e-5 * elevation)) * ra
        
        # Calculate net solar shortwave radiation 
        rns = (1 - ALBEDO) * rs_mean
        
        # Calculate net longwave radiation
        rnl = SIGMA * (((math.pow(ta_max_kelvin, 4) + math.pow(ta_min_kelvin, 4)) / 2) * (0.34 - (0.14 * math.sqrt(ea))) * ((1.35 * (rs_mean / rso)) - 0.35))
        
        # Calculate net radiation
        rn = rns - rnl
        
        
        # decompose et0 to two terms to facilitate the calculation
        """rng = 0.408 * rn
        radiation_term = ((delta) / (delta + (gamma * (1 + 0.34 * u2)))) * rng
        pt = (gamma) / (delta + gamma * (1 + (0.34 * u2)))
        tt = ((900) / (ta_mean + 273) ) * u2
        wind_term = pt * tt * (es - ea)
        et0 = radiation_term + wind_term """
        
        et0 = ((0.408 * delta * (rn - G)) + gamma * ((900 / (ta_mean + 273)) * u2 * (es - ea))) / (delta + (gamma * (1 + 0.34 * u2)))

        # output result
        return et0

    @staticmethod
    def et0_hargreaves(row):
        ta_mean, ta_max, ta_min, lat, doy =  row['ta_mean'], row['ta_max'], row['ta_min'], row['lat'], row['doy']
        
        # constants
        GSC = 0.082  # solar constant in MJ/m2/min

        # Calculate extraterrestrial radiation
        dr = 1 + 0.033 * math.cos(2 * math.pi / 365 * doy)
        d = 0.409*math.sin( (((2*math.pi) / 365) * doy) - 1.39)

        # sunset hour angle
        phi = math.radians(lat)
        omega = math.acos(-math.tan(phi)*math.tan(d))
        ra = ((24 * 60)/math.pi) * GSC * dr * ((omega * math.sin(phi) * math.sin(d)) + (math.cos(phi) * math.cos(d) * math.sin(omega)))
        
        et0 = 0.0023 * (ta_mean + 17.8) * (ta_max - ta_min) ** 0.5 * 0.408 * ra

        return et0
    
    def et0_estimation(self, 
                       air_temperture_column_name='ta',
                       global_solar_radiation_column_name='rs',
                       air_relative_humidity_column_name='rh',
                       wind_speed_column_name='ws',
                       date_time_column_name='date_time',
                       latitude=31.65410805,
                       longitude=-7.603140831,
                       method='pm',
                       in_place=True
                       ):
        
        et0_data = DataFrame()
        et0_data.add_column('ta_mean', self.resample_timeseries(in_place=False)[air_temperture_column_name])
        et0_data.add_column('ta_max', self.resample_timeseries(in_place=False, agg='max')[air_temperture_column_name])
        et0_data.add_column('ta_min', self.resample_timeseries(in_place=False, agg='min')[air_temperture_column_name], )
        et0_data.add_column('rh_max', self.resample_timeseries(in_place=False, agg='max')[air_relative_humidity_column_name])
        et0_data.add_column('rh_min', self.resample_timeseries(in_place=False, agg='min')[air_relative_humidity_column_name])
        et0_data.add_column('rh_mean', self.resample_timeseries(in_place=False)[air_relative_humidity_column_name])
        et0_data.add_column('u2_mean', self.resample_timeseries(in_place=False)[wind_speed_column_name])
        et0_data.add_column('rg_mean', self.resample_timeseries(in_place=False)[global_solar_radiation_column_name])
        et0_data.index_to_column()
        et0_data.add_doy_column('date_time')
        et0_data.add_one_value_column('elevation', DataFrame.get_elevation_and_latitude(latitude, longitude))
        et0_data.add_one_value_column('lat', latitude)
        
        if method == 'pm':
            et0_data.add_column_based_on_function('et0_pm', DataFrame.et0_penman_monteith)
        elif method == 'hargreaves':
            et0_data.add_column_based_on_function('et0_hargreaves', DataFrame.et0_hargreaves)
            
        if in_place == True:
            self.dataframe = et0_data.get_dataframe()
            
        return et0_data.get_dataframe()
        
        
    def column_to_date(self, column_name, format='%Y-%m-%d %H:%M:%S', extraction_func=None):
        if extraction_func is None:
            self.set_column(column_name, pd.to_datetime(self.get_column(column_name)))
            self.set_column(column_name, self.get_column(column_name).dt.strftime(format))
            self.set_column(column_name, pd.to_datetime(self.get_column(column_name)))
        else:
            self.transform_column(column_name, extraction_func)
        
    def datetime_reformate(self, date_time_column_name, new_format='%Y-%m-%d %H:%M:%S'):
        self.set_column(date_time_column_name, self.get_column(date_time_column_name).dt.strftime(new_format))
        return self.get_dataframe()
    
    def split_date_and_time_as_colmns(self, datetime_column_name):
        # Extract the date and time as new columns
        self.dataframe['time'] = self.dataframe[datetime_column_name].dt.time
        self.ignore_time_in_datetime(datetime_column_name)
        self.rename_columns({datetime_column_name: 'date'})
        #self.dataframe['date'] = self.dataframe[datetime_column_name].dt.date
        return self.get_dataframe()

    
    
    def resample_timeseries_parquet(
        self,
        frequency='D',
        agg='mean',
        between_time_tuple=None, # e.g. ('08:00','16:00')
        date_column_name=None,   # REQUIRED
        specific_hour_day=11,    # hour used for 'specific_hour_day' and 'keep_specific_hour'
        output_parquet_path=None,
        overwrite=False,
        compression='zstd',
        show_progress=True,
    ):
        """
        Resample a parquet dataset by a datetime column and export result to a single parquet file.

        Supported agg modes:
          mean | sum | max | min | median | std | var
          ffill | bfill
          specific_hour_day   -> pick value at specific_hour_day (if multiple within hour, aggregated later)
          keep_hourly         -> keep only rows at exact hour (minute=0 second=0) (no aggregation)
          keep_specific_hour  -> keep only rows whose hour == specific_hour_day (no aggregation)
        
        Notes:
          - keep_hourly / keep_specific_hour do NOT aggregate; they just filter and de‑duplicate timestamps.
          - specific_hour_day (legacy) still aggregates if multiple fragments provide same day/hour.

        Args:
            frequency: Pandas offset alias for resample (ignored for keep_* modes unless later aggregation).
            agg: aggregation mode (see above).
            between_time_tuple: (start_time_str, end_time_str) to filter by time-of-day before processing.
            date_column_name: name of datetime column (must exist).
            specific_hour_day: hour (0-23) used for specific hour modes.
            output_parquet_path: destination file or directory.
            overwrite: allow overwrite.
            compression: parquet codec.
            show_progress: show tqdm bars.

        Returns:
            dict summary
        """
        import os
        import pandas as pd
        import pyarrow as pa
        import pyarrow.parquet as pq
        from tqdm import tqdm

        if self.data_type != 'parquet' or not hasattr(self, 'dataset'):
            raise ValueError("resample_timeseries_parquet requires a parquet-backed DataFrame instance.")
        if date_column_name is None:
            raise ValueError("date_column_name is required.")
        if output_parquet_path is None:
            raise ValueError("output_parquet_path must be provided.")

        out_path = output_parquet_path
        if out_path.lower().endswith('.parquet'):
            out_dir = os.path.dirname(out_path) or "."
            os.makedirs(out_dir, exist_ok=True)
            if os.path.exists(out_path) and not overwrite:
                raise FileExistsError(f"{out_path} exists (overwrite=False).")
        else:
            os.makedirs(out_path, exist_ok=True)
            out_path = os.path.join(out_path, "resampled.parquet")
            if os.path.exists(out_path) and not overwrite:
                raise FileExistsError(f"{out_path} exists (overwrite=False).")

        if date_column_name not in self.dataset.schema.names:
            raise ValueError(f"Column '{date_column_name}' not found in parquet schema.")

        numeric_cols = []
        for f in self.dataset.schema:
            if f.name == date_column_name:
                continue
            if pa.types.is_integer(f.type) or pa.types.is_floating(f.type):
                numeric_cols.append(f.name)
        if not numeric_cols:
            raise ValueError("No numeric columns found to aggregate.")

        cols_to_read = [date_column_name] + numeric_cols
        partial_results = []
        total_in_rows = 0

        fragments = list(self.dataset.get_fragments())
        frag_iter = tqdm(fragments, desc="Fragments", disable=not show_progress)

        for fragment in frag_iter:
            try:
                table = fragment.to_table(columns=cols_to_read)
            except Exception as e:
                print(f"⚠️ Skipping fragment {getattr(fragment,'path', '')}: {e}")
                continue
            if table.num_rows == 0:
                continue

            df = table.to_pandas()
            df[date_column_name] = pd.to_datetime(df[date_column_name], errors='coerce')
            df = df.dropna(subset=[date_column_name])
            if df.empty:
                continue

            if between_time_tuple is not None:
                start_t, end_t = between_time_tuple
                mask = (df[date_column_name].dt.time >= pd.to_datetime(start_t).time()) & \
                       (df[date_column_name].dt.time <= pd.to_datetime(end_t).time())
                df = df.loc[mask]
                if df.empty:
                    continue

            if agg == 'specific_hour_day':
                df = df[df[date_column_name].dt.hour == specific_hour_day]
                if df.empty:
                    continue
                # normalize to date for later daily aggregation (one row per day)
                df['_resample_key'] = df[date_column_name].dt.normalize()

            elif agg == 'keep_hourly':
                mask = (
                    (df[date_column_name].dt.minute == 0) &
                    (df[date_column_name].dt.second == 0) &
                    (df[date_column_name].dt.microsecond == 0)
                )
                df = df.loc[mask]
                if df.empty:
                    continue
                df['_resample_key'] = df[date_column_name]  # keep original stamps (no aggregation)

            elif agg == 'keep_specific_hour':
                df = df[df[date_column_name].dt.hour == specific_hour_day]
                if df.empty:
                    continue
                df['_resample_key'] = df[date_column_name]  # keep only that hour rows directly

            else:
                # Standard resampling aggregation
                df = df.set_index(date_column_name)
                if agg in ['mean','sum','max','min','median','std','var']:
                    df = getattr(df.resample(frequency), agg)()
                elif agg in ['ffill','bfill']:
                    df = df.resample(frequency).asfreq()
                    df = getattr(df, agg)()
                else:
                    df = df.resample(frequency).mean()
                df['_resample_key'] = df.index

            agg_chunk = df[numeric_cols + ['_resample_key']].reset_index(drop=True)
            partial_results.append(agg_chunk)
            total_in_rows += table.num_rows

        if not partial_results:
            pq.write_table(
                pa.Table.from_arrays(
                    [pa.array([], type=pa.timestamp('ns'))] +
                    [pa.array([], type=pa.float64()) for _ in numeric_cols],
                    names=[date_column_name] + numeric_cols
                ),
                out_path,
                compression=compression
            )
            return {
                "output_file": out_path,
                "rows_in": 0,
                "rows_out": 0,
                "agg": agg,
                "frequency": frequency
            }

        import pandas as pd
        combined = pd.concat(partial_results, ignore_index=True)

        if agg in ['keep_hourly', 'keep_specific_hour']:
            # Pure filtering; de-duplicate timestamps
            combined.drop_duplicates(subset='_resample_key', inplace=True)
        elif agg == 'specific_hour_day':
            # Aggregate per day if multiple fragments supply same day
            combined = combined.groupby('_resample_key', as_index=False).agg(
                {c: 'mean' for c in numeric_cols}
            )
        else:
            combined = combined.groupby('_resample_key', as_index=False).agg(
                {c: (agg if agg in ['mean','sum','max','min','median','std','var'] else 'mean')
                 for c in numeric_cols}
            )

        combined.rename(columns={'_resample_key': date_column_name}, inplace=True)
        combined.sort_values(date_column_name, inplace=True)

        table_out = pa.Table.from_pandas(combined, preserve_index=False)
        pq.write_table(table_out, out_path, compression=compression)

        return {
            "output_file": out_path,
            "rows_in": total_in_rows,
            "rows_out": int(combined.shape[0]),
            "agg": agg,
            "frequency": frequency,
            "numeric_columns": numeric_cols,
            "specific_hour_day": specific_hour_day if 'hour' in agg else None
        }
    
    
    
    def resample_timeseries(self, 
                            frequency='d', 
                            agg='mean', 
                            skip_rows=None, 
                            intitial_index=0, 
                            between_time_tuple=None,
                            date_column_name=None,
                            specific_hour_day=11,
                            in_place=True):
        if in_place is True:
            if skip_rows is not None:
                index_name = self.dataframe.index.name
                self.index_to_column()
                self.set_dataframe(self.get_dataframe().loc[intitial_index:self.get_shape()[0]:skip_rows])
                self.reindex_dataframe(index_name)
            else:
                if date_column_name is not None:
                    self.column_to_date(date_column_name)
                    self.reindex_dataframe(date_column_name)
                    
                if between_time_tuple is not None:
                    temp_time_series = self.get_dataframe()
                    temp_time_series = temp_time_series.between_time(between_time_tuple[0], between_time_tuple[1])
                    self.dataframe = temp_time_series
                    
                if agg == 'sum':
                    self.set_dataframe(self.dataframe.resample(frequency).sum())
                if agg == 'mean':
                    self.set_dataframe(self.dataframe.resample(frequency).mean())
                if agg == 'max':
                    self.set_dataframe(self.dataframe.resample(frequency).max())
                if agg == 'min':
                    self.set_dataframe(self.dataframe.resample(frequency).min())
                if agg == 'median':
                    self.set_dataframe(self.dataframe.resample(frequency).median())
                if agg == 'std':
                    self.set_dataframe(self.dataframe.resample(frequency).std())
                if agg == 'var':
                    self.set_dataframe(self.dataframe.resample(frequency).var())
                if agg == 'ffill':
                    self.set_dataframe(self.dataframe.resample(frequency).ffill())
                if agg == 'bfill':
                    self.set_dataframe(self.dataframe.resample(frequency).bfill())
                if agg == 'specific_hour_day':
                    # Filtering the DataFrame to keep only rows at 'hour_of_day'
                    self.dataframe = self.dataframe[self.dataframe.index.hour == specific_hour_day]
                    # Resampling the DataFrame to daily frequency
                    # 'first' is used here to take the first available value for each day, which should be unique in this scenario
                    self.set_dataframe(self.dataframe.resample('D').first())
                else:
                    self.set_dataframe(self.dataframe.resample(frequency).mean())
            return self.get_dataframe()
        else:
            if skip_rows is not None:
                self.set_dataframe(self.get_dataframe().loc[intitial_index:self.get_shape()[0]:skip_rows])
            else:
                if date_column_name is not None:
                    self.reindex_dataframe(date_column_name)
                    
                temp_time_series = self.dataframe
                
                if between_time_tuple is not None:
                    temp_time_series = temp_time_series.between_time(between_time_tuple[0], between_time_tuple[1])

                if agg == 'sum':
                    resampled_dataframe = temp_time_series.resample(frequency).sum()
                if agg == 'mean':
                    resampled_dataframe = temp_time_series.resample(frequency).mean()
                if agg == 'max':
                    resampled_dataframe = temp_time_series.resample(frequency).max()
                if agg == 'min':
                    resampled_dataframe = temp_time_series.resample(frequency).min()
                if agg == 'median':
                    resampled_dataframe = temp_time_series.resample(frequency).median()
                if agg == 'std':
                    resampled_dataframe = temp_time_series.resample(frequency).std()
                if agg == 'var':
                    resampled_dataframe = temp_time_series.resample(frequency).var()
                if agg == 'ffill':
                    resampled_dataframe = temp_time_series.resample(frequency).ffill()
                if agg == 'bfill':
                    resampled_dataframe = temp_time_series.resample(frequency).bfill()
            
            return resampled_dataframe
        
    def to_time_series(self, date_column, value_column, date_format='%Y-%m-%d', window_size=2, one_row=False):
        from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator as SG
        # when working with train test generators
        """def to_time_series_generators(self, date_column, time_series_column, date_format='%Y-%m-%d', window_size=2, train_percent=0.8):
        self.column_to_date(date_column, format=date_format)
        self.reindex_dataframe(self.get_column(date_column))
        self.drop_column(date_column)
        #dataframa.asfreq('d') # h hourly w weekly d normal daily b business day  m monthly a annualy
        train, test = self.train_test_split_column(time_series_column, 0.8)
        self.set_train_generator(TimeseriesGenerator(np.reshape(train.values,
                                                          (len(train),1)),
                                               np.reshape(train.values,
                                                          (len(train),1)),
                                               length=window_size,
                                               batch_size=1,
                                               ))
        self.set_test_generator(TimeseriesGenerator(np.reshape(test.values,
                                                          (len(test),1)),
                                               np.reshape(test.values,
                                                          (len(test),1)),
                                               length=window_size,
                                               batch_size=1,
                                               ))
        return self.get_train_generator(), self.get_test_generator()"""
        self.column_to_date(date_column, format=date_format)
        self.reindex_dataframe(self.get_column(date_column))
        self.keep_columns(value_column)
        if one_row is False:
            #dataframa.asfreq('d') # h hourly w weekly d normal daily b business day  m monthly a annualy
            self.set_generator(SG(self.get_min_max_scaled_dataframe(),
                                                self.get_min_max_scaled_dataframe(),
                                                length=window_size,
                                                batch_size=1,))
            """self.set_generator(
                TimeseriesGenerator(self.get_min_max_scaled_dataframe(), 
                                    self.get_min_max_scaled_dataframe(), 
                                    length=window_size, 
                                    length_output=7, 
                                    batch_size=1)"""
            return self.get_generator()
    
    def drop_rows(self, nbr_rows=1):
        """Drop the first nbr_rows of rows from the dataframe

        Args:
            nbr_rows (int, optional): if negative value is given then thelen the last nbr_rows. Defaults to 1.

        Returns:
            None
        """
        
        if nbr_rows < 0:
            self.set_dataframe(self.get_dataframe().iloc[:self.get_shape()[0]+nbr_rows])
        else:
            self.set_dataframe(self.get_dataframe().iloc[nbr_rows:])
            
    def drop_rows_by_indices(self, indexes_as_list=[0]):
        """Drop rows given their indexes

        Args:
            indexes_as_list (list, optional): [description]. Defaults to [0].
        """
        self.set_dataframe(self.get_dataframe().drop(indexes_as_list))
        
    def dataframe_skip_columns(self, intitial_index, final_index, step=2):
        self.set_dataframe(self.get_dataframe().loc[intitial_index:final_index:step])
        
    def class_distribution(self, 
                           class_column_name,
                           show_x_labels=True,
                           show_y_labels=True,
                           show_title=True,
                           title='Class Distribution',
                           xlabel='Classes',
                           ylabel='Percentage'
                           ):
        """
        Show the distribution of classes in a pie chart with both percentages and counts.
        
        Parameters:
        -----------
        class_column_name : str
            Name of the column containing class labels
            
        Returns:
        --------
        pandas.Series
            Series with value counts for each class
        """
        class_distribution = self.get_column(class_column_name).value_counts()
        
        # Create a function to format labels with both percentage and count
        def autopct_format(values):
            def my_format(pct):
                total = sum(values)
                count = int(round(pct*total/100.0))
                return '{:.1f}% ({:d})'.format(pct, count)
            return my_format
        
        # Plot pie chart with both percentage and count and smaller text
        class_distribution.plot(
            kind='pie', 
            autopct=autopct_format(class_distribution.values),
            textprops={'fontsize': 8},  # Smaller font size for the labels
            ylabel='',  # Remove default y-label
        )
        if show_title:
            plt.title(title)
        if show_x_labels:
            plt.xlabel(xlabel)
        if show_y_labels:
            plt.ylabel(ylabel)
        plt.show()
        
        return class_distribution
        
    def shuffle_dataframe(self):
        self.set_dataframe(self.get_dataframe().sample(frac=1).reset_index(drop=True))
        
    def add_doy_column(self, new_column_name='doy', datetime_column_name='datetime'):
        try:
            if not self.get_column(datetime_column_name).dtype == 'datetime64[ns]':
                self.column_to_date(datetime_column_name)
            self.add_column(new_column_name, self.get_column(datetime_column_name).dt.day_of_year)
        except AttributeError as e:
            print(f'Try to convert {datetime_column_name} to datetime first.')
            
    def add_hod_column(self, new_column_name='hod', datetime_column_name='datetime'):
        try:
            self.add_column(new_column_name, self.get_column(datetime_column_name).dt.hour)
        except AttributeError as e:
            print(f'Try to convert {datetime_column_name} to datetime first.')
        
            
    def add_month_column(self, new_column_name='month', datetime_column_name='datetime'):
        try:
            self.add_column(new_column_name, self.get_column(datetime_column_name).dt.month)
        except AttributeError as e:
            print(f'Try to convert {datetime_column_name} to datetime first.')
            
    def add_year_column(self, new_column_name='year', datetime_column_name='datetime'):
        try:
            self.add_column(new_column_name, self.get_column(datetime_column_name).dt.year)
        except AttributeError as e:
            print(f'Try to convert {datetime_column_name} to datetime first.')
        
    def datetime_to_doy(self, new_column_name='doy', date_time_column_name='datetime'):
        try:
            self.add_column(new_column_name, self.get_column(date_time_column_name).dt.day_of_year)
            self.drop_column(date_time_column_name)
            self.rename_columns({new_column_name: date_time_column_name})
        except AttributeError as e:
            print(f'Try to convert {date_time_column_name} to datetime first.')
        
    def add_month_day_column(self, date_time_column_name='date_time'):
        self.add_column('month-day', self.get_column(date_time_column_name).dt.month.astype(str) + '-' + self.get_column(date_time_column_name).dt.day.astype(str))
    
    def scale_columns(self, columns_names_as_list, scaler_type='min_max', in_place=True):
        """A method  to standardize the independent features present in the concerned columns in a fixed range.

        Args:
            column_name ([type]): 
            scaler_type (str, optional): ['min_max', 'standard', 'adjusted_log']. Defaults to 'min_max'.
            in_place (bool, optional): if False the modification do not  affects the original columns. Defaults to True.
        """
        if scaler_type == 'min_max':
            self.vectorizer = MinMaxScaler() 
            dest_columns = self.get_columns(columns_names_as_list)
            dest_dataframe = DataFrame(self.vectorizer.fit_transform(X=dest_columns), 
                                       line_index=self.get_index(),
                                       columns_names_as_list=columns_names_as_list, 
                                       data_type='matrix')
            self.drop_columns(columns_names_as_list)
            self.concatinate(dest_dataframe.get_dataframe())
            return dest_dataframe.get_dataframe()
        elif scaler_type == 'standard':
            self.vectorizer = StandardScaler()
            dest_columns = self.get_columns(columns_names_as_list)
            dest_dataframe = DataFrame(self.vectorizer.fit_transform(X=dest_columns), 
                                       line_index=self.get_index(),
                                       columns_names_as_list=columns_names_as_list, 
                                       data_type='matrix')
            self.drop_columns(columns_names_as_list)
            self.concatinate(dest_dataframe.get_dataframe())
            return dest_dataframe.get_dataframe()
        elif scaler_type == 'adjusted_log':
            def log_function(o, min_column):
                return np.log(1 + o - min_column)
            for name in columns_names_as_list:
                min_column = self.get_column(name).min()
                self.transform_column(name, log_function, min_column)
            return self.get_columns(columns_names_as_list)
                        
    def scale_dataframe(self, scaler_type='min_max', sense='direct', in_place=True):
        """A method  to standardize the independent features present in the dataframe in a fixed range.

        Args:
            column_name ([type]): 
            scaler_type (str, optional): ['min_max', 'standard', 'adjusted_log']. Defaults to 'min_max'.
            in_place (bool, optional): if False the modification do not  affects the dataframe. Defaults to True.
        """
        if sense == 'inverse':
            original_data = self.vectorizer.inverse_transform(self.dataframe)
            original_data = self.vectorizer.inverse_transform(self.dataframe)
            self.dataframe = pd.DataFrame(original_data, columns=self.dataframe.columns, index=self.dataframe.index)
        else:
            if scaler_type == 'min_max':
                self.vectorizer = MinMaxScaler() 
                column_names = self.get_columns_names()
                self.set_dataframe(DataFrame(self.vectorizer.fit_transform(X=self.get_dataframe()), 
                                        line_index=self.get_index(),
                                        columns_names_as_list=column_names, 
                                        data_type='matrix').get_dataframe())
            elif scaler_type == 'standard':
                self.vectorizer = StandardScaler() 
                self.set_dataframe(self.vectorizer.fit_transform(X=self.get_dataframe()))
            elif scaler_type == 'adjusted_log':
                def log_function(o, min_column):
                    return np.log(1 + o - min_column)
                for name in self.get_columns_names():
                    min_column = self.get_column(name).min()
                    self.transform_column(name, log_function, min_column)
            self.convert_dataframe_type()
            return self.get_dataframe()
    
    def load_dataset(self, dataset='iris'):
        """
        boston: Load and return the boston house-prices dataset (regression)
        iris: Load and return the iris dataset (classification).
        """
        if dataset == 'iris':
            data = load_iris(as_frame=True)
            x = data.data
            y = data.target
            self.set_dataframe(x)
            self.add_column('target', y)  
            
    def similarity_measure(self, column_name1, column_name2, similarity_method='cosine', weighting_method='tfidf'):
        if similarity_method == 'cosine':
            corpus = self.get_column_as_list(column_name1) + self.get_column_as_list(column_name2)
            vectorizer = Vectorizer(corpus, weighting_method)
            print(len(self.get_column_as_list(column_name1)))
            print(len(self.get_column_as_list(column_name2)))
            new_column = []
            for p in zip(self.get_column_as_list(column_name1), self.get_column_as_list(column_name2)):
                new_column.append(vectorizer.cosine_similarity(p[0], p[1])) 
            
            self.add_column('Similarity score', new_column)
        
        elif similarity_method == 'ts':
            from sklearn.metrics import r2_score, mean_squared_error
            import math
            comparaison_dict = {}
            comparaison_dict['RMSE'] = math.sqrt(mean_squared_error(self.get_column(column_name1), self.get_column(column_name2)))
            comparaison_dict['R²'] = r2_score(self.get_column(column_name1), self.get_column(column_name2))
            comparaison_dict['R'] = self.get_column(column_name1).corr(self.get_column(column_name2))
            
            return comparaison_dict
        
        return self.get_dataframe()
        
    @staticmethod
    def outliers_decision_function(o, min_quantile, max_quantile):
        if min_quantile < o < max_quantile:
            return True
        return False
    
    @staticmethod
    def generate_datetime_range(starting_datetime='2013-01-01 00:00:00', end_datetime='2013-12-31 00:00:00', freq='1H', periods=None):
        if periods is not None:
            return pd.date_range(start=starting_datetime, periods=periods, freq=freq)
        return pd.date_range(start=starting_datetime, end=end_datetime, freq=freq)

    @staticmethod
    def scale_trasform_fun(o, max_column):
        return o / max_column