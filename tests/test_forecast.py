# File: tests/test_forecast.py
from pathlib import Path
import joblib
import pandas as pd
from forecast_app.feature_builder import build_latest_feature

def test_model_artifact_load():
    """Kiểm tra file model.joblib tồn tại và nạp được vào bộ nhớ."""
    model_path = Path("artifacts/model.joblib")
    assert model_path.exists(), "Không tìm thấy file model.joblib trong artifacts/"
    model = joblib.load(model_path)
    assert model is not None

def test_feature_builder_minimum_rows():
    """Kiểm tra không đủ 25 dòng thì tuyệt đối không được sinh đặc trưng."""
    df_short = pd.DataFrame({
        "source_datetime": pd.date_range("2026-01-01", periods=10, freq="h"),
        "pm25": [15.0] * 10
    })
    feat = build_latest_feature(df_short)
    assert feat is None, "Khi thiếu dữ liệu lookback, builder phải trả về None"