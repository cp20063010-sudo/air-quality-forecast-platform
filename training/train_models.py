# File: training/train_models.py
import json
from pathlib import Path
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestRegressor,
    VotingRegressor,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

INPUT_DATA = Path("data/training_features.parquet")
ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(exist_ok=True)

print("--> 1. Đang nạp dữ liệu đặc trưng...")
data = pd.read_parquet(INPUT_DATA)

categorical = ["station"]
numeric = [
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
    "pm25", "pm10", "so2", "no2", "co", "o3",
    "temperature", "pressure", "dew_point", "rain", "wind_speed",
    "pm25_lag_1", "pm25_lag_2", "pm25_lag_3", "pm25_lag_6", "pm25_lag_12", "pm25_lag_24",
    "pm25_roll_mean_3", "pm25_roll_mean_6", "pm25_roll_mean_12", "pm25_roll_mean_24",
    "pm25_roll_std_3", "pm25_roll_std_6", "pm25_roll_std_12", "pm25_roll_std_24",
]
feature_columns = categorical + numeric

preprocessor = ColumnTransformer(
    transformers=[
        ("station", OneHotEncoder(handle_unknown="ignore"), categorical),
        ("numeric", "passthrough", numeric),
    ]
)

# Cấu hình siêu nhẹ tối ưu cho RAM 512MB Render
print("--> 2. Cấu hình mô hình tối ưu bộ nhớ...")
rf = RandomForestRegressor(
    n_estimators=15,
    max_depth=8,
    max_samples=0.1,
    n_jobs=-1,
    random_state=42,
)

hgb = HistGradientBoostingRegressor(
    max_iter=50,
    learning_rate=0.08,
    max_leaf_nodes=31,
    random_state=42,
)

voting = VotingRegressor(
    estimators=[("rf", rf), ("hgb", hgb)],
    weights=[0.5, 0.5],
    n_jobs=None,
)

voting_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", voting),
])

TRAIN_END = pd.Timestamp("2015-12-31 23:00:00")
VALID_END = pd.Timestamp("2016-08-31 23:00:00")

dt = data["source_datetime"]
train = data[dt <= TRAIN_END]
valid = data[(dt > TRAIN_END) & (dt <= VALID_END)]
test = data[dt > VALID_END]

X_train, y_train = train[feature_columns], train["pm25_target"]
X_valid, y_valid = valid[feature_columns], valid["pm25_target"]
X_test, y_test = test[feature_columns], test["pm25_target"]

print("--> 3. Đang huấn luyện...")
voting_pipeline.fit(X_train, y_train)

test_preds = voting_pipeline.predict(X_test)
test_metrics = {
    "mae": round(float(mean_absolute_error(y_test, test_preds)), 3),
    "rmse": round(float(mean_squared_error(y_test, test_preds) ** 0.5), 3),
    "r2": round(float(r2_score(y_test, test_preds)), 4),
}
print("    Test Metrics:", test_metrics)

print("--> 4. Lưu Artifacts...")
joblib.dump(voting_pipeline, ARTIFACTS_DIR / "model.joblib", compress=3)

(ARTIFACTS_DIR / "feature_columns.json").write_text(
    json.dumps(feature_columns, indent=2), encoding="utf-8"
)

(ARTIFACTS_DIR / "metrics.json").write_text(
    json.dumps(test_metrics, indent=2), encoding="utf-8"
)

print(f"--> [HOÀN TẤT] File model.joblib mới đã được tạo với dung lượng siêu nhẹ!")