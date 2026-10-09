"""
Macroeconomic Data Provider Abstraction Layer
"""
import pandas as pd
import numpy as np
from datetime import datetime

class MacroDataProvider:
    def __init__(self, provider_name: str = "synthetic_fred_fallback"):
        self.provider_name = provider_name

    def fetch_macro_series(self, start_date: str = "2010-01-01", end_date: str = "2026-01-01") -> pd.DataFrame:
        """
        Retrieves historical macroeconomic indicators.
        In production, integrate FRED API (pandas_datareader / requests) or BIS data portals.
        """
        date_range = pd.date_range(start=start_date, end=end_date, freq="ME")
        np.random.seed(42)
        n = len(date_range)
        
        # Robust synthetic generation for structural testing when external API limits are hit
        macro_data = pd.DataFrame({
            "USM2": np.linspace(10000, 21000, n) + np.cumsum(np.random.normal(50, 15, n)),
            "FEDFUNDS": np.maximum(0.1, 2.0 + np.sin(np.linspace(0, 10, n)) * 2.5 + np.random.normal(0, 0.2, n)),
            "CPIAUCSL": np.linspace(220, 320, n) + np.cumsum(np.random.normal(0.5, 0.1, n)),
            "GDPC1": np.linspace(18000, 24000, n) + np.cumsum(np.random.normal(40, 10, n)),
            "UNRATE": np.maximum(3.0, 5.5 + np.cos(np.linspace(0, 8, n)) * 1.5 + np.random.normal(0, 0.2, n)),
            "PCEC96": np.linspace(13000, 18000, n) + np.cumsum(np.random.normal(30, 8, n)),
            "GCEC1": np.linspace(3000, 4000, n) + np.cumsum(np.random.normal(5, 2, n)),
            "NETEXC": np.random.normal(-800, 100, n),
            "RBUSBIS": np.linspace(95, 105, n) + np.random.normal(0, 1, n),
            "USINTR": np.maximum(0.2, 2.5 + np.sin(np.linspace(0, 10, n)) * 2.0 + np.random.normal(0, 0.1, n))
        }, index=date_range)
        
        macro_data.index.name = "datetime"
        return macro_data
