# File: forecast_app/config.py
import os
from pathlib import Path

OBSERVATION_HTTP_URL = os.getenv("OBSERVATION_HTTP_URL", "http://127.0.0.1:8000")
OBSERVATION_WS_URL = os.getenv("OBSERVATION_WS_URL", "ws://127.0.0.1:8000")
MODEL_PATH = Path(os.getenv("MODEL_PATH", "artifacts/model.joblib"))
FEATURE_COLUMNS_PATH = Path(os.getenv("FEATURE_COLUMNS_PATH", "artifacts/feature_columns.json"))
METRICS_PATH = Path(os.getenv("METRICS_PATH", "artifacts/metrics.json"))

BUFFER_HOURS = int(os.getenv("BUFFER_HOURS", "48"))
FORECAST_HORIZON = int(os.getenv("FORECAST_HORIZON", "1"))
MODEL_VERSION = os.getenv("MODEL_VERSION", "rf-ensemble-v1")