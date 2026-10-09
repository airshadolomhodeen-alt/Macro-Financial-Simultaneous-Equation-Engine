"""
Global Configuration and Environment Settings
"""
import os
import streamlit as st

class Settings:
    PROJECT_NAME: str = "Macro-Financial Simultaneous Equation Engine"
    VERSION: str = "1.0.0"
    DEFAULT_FREQUENCY: str = "monthly"
    TWELVE_DATA_BASE_URL: str = "https://api.twelvedata.com"
    
    @property
    def TWELVE_DATA_API_KEY(self) -> str:
        if "api" in st.secrets and "twelve_data_key" in st.secrets["api"]:
            return st.secrets["api"]["twelve_data_key"]
        return os.getenv("TWELVE_DATA_API_KEY", "32b6a749e8c14835b95b8a9c271eec95")

settings = Settings()
