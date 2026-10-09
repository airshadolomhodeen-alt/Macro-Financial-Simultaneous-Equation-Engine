"""
Parquet Local Cache Manager
"""
import os
import pandas as pd
from pathlib import Path

class CacheManager:
    def __init__(self, cache_dir: str = "data/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def save_to_cache(self, df: pd.DataFrame, filename: str) -> None:
        file_path = self.cache_dir / filename
        df.to_parquet(file_path, compression="snappy")

    def load_from_cache(self, filename: str) -> pd.DataFrame | None:
        file_path = self.cache_dir / filename
        if file_path.exists():
            return pd.read_parquet(file_path)
        return None

    def clear_cache(self) -> None:
        for file in self.cache_dir.glob("*.parquet"):
            file.unlink()
