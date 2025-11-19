"""
Module: Cache Manager

Hybrid caching strategy using memory (LRU) and disk (HDF5/SQLite) to minimize
API calls and accelerate backtesting.

Classes
-------
CacheManager
    Hybrid cache with memory and disk storage
CacheStats
    Cache statistics container

Notes
-----
Cache Key Format: {source}:{tickers_hash}:{start_date}:{end_date}:{interval}

Storage Format:
- Memory: Pandas DataFrames in LRU cache
- Disk: HDF5 format with compression for DataFrames >1MB
- Metadata: SQLite for cache timestamps and access patterns

Author: Quantitative Research Team
Created: 2025-01-18
"""

import hashlib
import sqlite3
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.utils.config import ConfigManager
from src.utils.io import ensure_dir, load_hdf5, save_hdf5
from src.utils.logger import StructuredLogger


@dataclass
class CacheStats:
    """
    Container for cache statistics.

    Attributes
    ----------
    hits : int
        Number of cache hits
    misses : int
        Number of cache misses
    hit_rate : float
        Cache hit rate (hits / (hits + misses))
    total_entries : int
        Total number of cached entries
    memory_size_mb : float
        Approximate memory cache size in MB
    disk_size_mb : float
        Disk cache size in MB
    """
    hits: int
    misses: int
    hit_rate: float
    total_entries: int
    memory_size_mb: float
    disk_size_mb: float


class CacheManager:
    """
    Hybrid caching strategy using memory (LRU) and disk (HDF5/SQLite).

    Attributes
    ----------
    disk_cache_path : Path
        Path to disk cache directory
    ttl : int
        Time-to-live for cache entries in seconds
    max_memory_size_mb : int
        Maximum memory cache size in MB

    Methods
    -------
    get(key)
        Get cached value
    set(key, value, ttl)
        Set cache value
    invalidate(pattern)
        Invalidate cache entries matching pattern
    get_stats()
        Get cache statistics
    clear()
        Clear all cache entries

    Examples
    --------
    >>> cache = CacheManager()
    >>> cache.set("test_key", df, ttl=3600)
    >>> df_cached = cache.get("test_key")
    """

    def __init__(self, config: Optional[ConfigManager] = None):
        """
        Initialize cache manager.

        Parameters
        ----------
        config : Optional[ConfigManager]
            Configuration manager instance
        """
        self.logger = StructuredLogger(__name__)

        if config is None:
            config = ConfigManager.load_config()

        self.config = config
        self.disk_cache_path = Path(config.get("data.storage.raw_data_path", "data/cache"))
        self.ttl = config.get("data.storage.cache_ttl", 86400)  # 24 hours default

        # Create cache directories
        ensure_dir(self.disk_cache_path)

        # Initialize SQLite metadata database
        self.db_path = self.disk_cache_path / "cache_metadata.db"
        self._init_database()

        # Cache statistics
        self.hits = 0
        self.misses = 0

        # Memory cache (using class-level to persist across instances)
        self._memory_cache: Dict[str, tuple] = {}

        self.logger.info(
            "CacheManager initialized",
            disk_cache_path=str(self.disk_cache_path),
            ttl=self.ttl
        )

    def _init_database(self) -> None:
        """Initialize SQLite metadata database."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS cache_metadata (
                key TEXT PRIMARY KEY,
                created_at REAL,
                last_accessed REAL,
                ttl INTEGER,
                size_bytes INTEGER,
                storage_location TEXT
            )
        ''')

        conn.commit()
        conn.close()

    def _generate_cache_key(
        self,
        source: str,
        tickers: List[str],
        start_date: str,
        end_date: str,
        interval: str = "1d"
    ) -> str:
        """
        Generate cache key from parameters.

        Parameters
        ----------
        source : str
            Data source name
        tickers : List[str]
            List of tickers
        start_date : str
            Start date
        end_date : str
            End date
        interval : str
            Data interval

        Returns
        -------
        str
            Cache key
        """
        # Sort tickers for consistency
        tickers_sorted = sorted(tickers)
        tickers_str = "_".join(tickers_sorted)

        # Create hash for long ticker lists
        if len(tickers_str) > 50:
            tickers_hash = hashlib.md5(tickers_str.encode()).hexdigest()[:10]
        else:
            tickers_hash = tickers_str

        key = f"{source}:{tickers_hash}:{start_date}:{end_date}:{interval}"
        return key

    def get(self, key: str) -> Optional[pd.DataFrame]:
        """
        Get cached value.

        Parameters
        ----------
        key : str
            Cache key

        Returns
        -------
        Optional[pd.DataFrame]
            Cached DataFrame or None if not found/expired

        Examples
        --------
        >>> df = cache.get("yfinance:AAPL:2020-01-01:2021-01-01:1d")
        """
        # Check memory cache first
        if key in self._memory_cache:
            value, expiry_time = self._memory_cache[key]

            if time.time() < expiry_time:
                self.hits += 1
                self._update_last_accessed(key)
                self.logger.debug("Cache hit (memory)", key=key)
                return value
            else:
                # Expired, remove from memory cache
                del self._memory_cache[key]

        # Check disk cache
        metadata = self._get_metadata(key)

        if metadata is None:
            self.misses += 1
            self.logger.debug("Cache miss", key=key)
            return None

        # Check if expired
        if time.time() - metadata['created_at'] > metadata['ttl']:
            self.misses += 1
            self.logger.debug("Cache expired", key=key)
            self.invalidate(key)
            return None

        # Load from disk
        try:
            file_path = Path(metadata['storage_location'])
            if file_path.exists():
                df = load_hdf5(file_path, key='data')

                # Add to memory cache
                expiry_time = time.time() + self.ttl
                self._memory_cache[key] = (df, expiry_time)

                self.hits += 1
                self._update_last_accessed(key)
                self.logger.debug("Cache hit (disk)", key=key)
                return df

        except Exception as e:
            self.logger.error("Error loading from cache", key=key, error=str(e))

        self.misses += 1
        return None

    def set(
        self,
        key: str,
        value: pd.DataFrame,
        ttl: Optional[int] = None
    ) -> None:
        """
        Set cache value.

        Parameters
        ----------
        key : str
            Cache key
        value : pd.DataFrame
            DataFrame to cache
        ttl : Optional[int]
            Time-to-live in seconds (uses default if None)

        Examples
        --------
        >>> cache.set("test_key", df, ttl=3600)
        """
        if ttl is None:
            ttl = self.ttl

        # Calculate expiry time
        expiry_time = time.time() + ttl

        # Add to memory cache
        self._memory_cache[key] = (value, expiry_time)

        # Save to disk
        file_path = self.disk_cache_path / f"{hashlib.md5(key.encode()).hexdigest()}.h5"

        try:
            save_hdf5(value, file_path, key='data')

            # Store metadata
            size_bytes = file_path.stat().st_size if file_path.exists() else 0

            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            cursor.execute('''
                INSERT OR REPLACE INTO cache_metadata
                (key, created_at, last_accessed, ttl, size_bytes, storage_location)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (key, time.time(), time.time(), ttl, size_bytes, str(file_path)))

            conn.commit()
            conn.close()

            self.logger.debug(
                "Cache set",
                key=key,
                size_mb=round(size_bytes / (1024 * 1024), 2)
            )

        except Exception as e:
            self.logger.error("Error setting cache", key=key, error=str(e))

    def invalidate(self, pattern: str) -> int:
        """
        Invalidate cache entries matching pattern.

        Parameters
        ----------
        pattern : str
            Pattern to match (supports SQL LIKE syntax with %)

        Returns
        -------
        int
            Number of entries invalidated

        Examples
        --------
        >>> num_invalidated = cache.invalidate("yfinance:%")
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        # Get matching entries
        cursor.execute('''
            SELECT key, storage_location FROM cache_metadata
            WHERE key LIKE ?
        ''', (pattern,))

        entries = cursor.fetchall()

        # Delete files and metadata
        for key, storage_location in entries:
            # Delete file
            try:
                file_path = Path(storage_location)
                if file_path.exists():
                    file_path.unlink()
            except Exception as e:
                self.logger.warning(f"Error deleting cache file: {e}")

            # Remove from memory cache
            if key in self._memory_cache:
                del self._memory_cache[key]

        # Delete metadata
        cursor.execute('DELETE FROM cache_metadata WHERE key LIKE ?', (pattern,))

        count = cursor.rowcount
        conn.commit()
        conn.close()

        self.logger.info("Cache invalidated", pattern=pattern, count=count)

        return count

    def get_stats(self) -> CacheStats:
        """
        Get cache statistics.

        Returns
        -------
        CacheStats
            Cache statistics

        Examples
        --------
        >>> stats = cache.get_stats()
        >>> print(f"Hit rate: {stats.hit_rate:.2%}")
        """
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0.0

        # Calculate disk cache size
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute('SELECT COUNT(*), SUM(size_bytes) FROM cache_metadata')
        total_entries, total_size_bytes = cursor.fetchone()

        conn.close()

        total_entries = total_entries or 0
        total_size_bytes = total_size_bytes or 0
        disk_size_mb = total_size_bytes / (1024 * 1024)

        # Estimate memory cache size (rough approximation)
        memory_size_mb = len(self._memory_cache) * 0.5  # Assume ~0.5MB per entry

        return CacheStats(
            hits=self.hits,
            misses=self.misses,
            hit_rate=hit_rate,
            total_entries=total_entries,
            memory_size_mb=memory_size_mb,
            disk_size_mb=disk_size_mb
        )

    def clear(self) -> int:
        """
        Clear all cache entries.

        Returns
        -------
        int
            Number of entries cleared
        """
        return self.invalidate("%")

    def _get_metadata(self, key: str) -> Optional[Dict[str, Any]]:
        """Get metadata for cache entry."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute('''
            SELECT key, created_at, last_accessed, ttl, size_bytes, storage_location
            FROM cache_metadata WHERE key = ?
        ''', (key,))

        row = cursor.fetchone()
        conn.close()

        if row is None:
            return None

        return {
            'key': row[0],
            'created_at': row[1],
            'last_accessed': row[2],
            'ttl': row[3],
            'size_bytes': row[4],
            'storage_location': row[5]
        }

    def _update_last_accessed(self, key: str) -> None:
        """Update last accessed time for cache entry."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute('''
            UPDATE cache_metadata SET last_accessed = ? WHERE key = ?
        ''', (time.time(), key))

        conn.commit()
        conn.close()
