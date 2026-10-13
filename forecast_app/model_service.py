# File: forecast_app/model_service.py
import json
from pathlib import Path
import joblib
import pandas as pd

class ModelService:
    def __init__(self, model_path: Path, feature_path: Path):
        self.model = joblib.load(model_path)
        self.feature_columns = json.loads(feature_path.read_text(encoding="utf-8"))

    def predict(self, feature_rows: list[dict]) -> list[float]:
        frame = pd.DataFrame(feature_rows)
        frame = frame[self.feature_columns]
        values = self.model.predict(frame)
        # Nồng độ bụi không nhận giá trị âm
        return values.clip(min=0.0).tolist()