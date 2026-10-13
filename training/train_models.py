# File: training/train_models.py
import json
from pathlib import Path
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesRegressor,
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

# 1. Chỉ định chính xác 32 đặc trưng chuẩn theo đề bài
categorical = ["station"]
numeric = [
    # Đặc trưng chu kỳ thời gian
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
    # Chất ô nhiễm tại thời điểm t
    "pm25", "pm10", "so2", "no2", "co", "o3",
    # Khí tượng
    "temperature", "pressure", "dew_point", "rain", "wind_speed",
    # Giá trị PM2.5 quá khứ (Lag)
    "pm25_lag_1", "pm25_lag_2", "pm25_lag_3", "pm25_lag_6", "pm25_lag_12", "pm25_lag_24",
    # Thống kê trượt (Rolling mean & std)
    "pm25_roll_mean_3", "pm25_roll_mean_6", "pm25_roll_mean_12", "pm25_roll_mean_24",
    "pm25_roll_std_3", "pm25_roll_std_6", "pm25_roll_std_12", "pm25_roll_std_24",
]
feature_columns = categorical + numeric

print(f"    Tổng số đặc trưng chuẩn: {len(feature_columns)} (1 categorical, {len(numeric)} numeric)")

# 2. Pipeline tiền xử lý: OneHot cho trạm quan trắc, giữ nguyên biến số
preprocessor = ColumnTransformer(
    transformers=[
        ("station", OneHotEncoder(handle_unknown="ignore"), categorical),
        ("numeric", "passthrough", numeric),
    ]
)

# 3. Cấu hình các Base Models với n_estimators = 50 cây
print("--> 2. Cấu hình các mô hình Ensemble (n_estimators=50)...")
rf = RandomForestRegressor(
    n_estimators=50,
    max_depth=18,
    min_samples_leaf=2,
    max_features=0.7,
    n_jobs=-1,
    random_state=42,
)

extra = ExtraTreesRegressor(
    n_estimators=50,
    max_depth=18,
    min_samples_leaf=2,
    n_jobs=-1,
    random_state=42,
)

hgb = HistGradientBoostingRegressor(
    max_iter=100,
    learning_rate=0.05,
    max_leaf_nodes=31,
    random_state=42,
)

# 4. Voting Regressor (n_jobs=None để tránh nghẽn luồng CPU trên Windows)
voting = VotingRegressor(
    estimators=[("rf", rf), ("extra", extra), ("hgb", hgb)],
    weights=[0.4, 0.35, 0.25],
    n_jobs=None,
)

voting_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", voting),
])

# 5. Phân chia Train, Validation, Test theo mốc thời gian
print("--> 3. Phân chia Train, Validation, Test theo mốc thời gian...")
TRAIN_END = pd.Timestamp("2015-12-31 23:00:00")
VALID_END = pd.Timestamp("2016-08-31 23:00:00")

dt = data["source_datetime"]
train = data[dt <= TRAIN_END]
valid = data[(dt > TRAIN_END) & (dt <= VALID_END)]
test = data[dt > VALID_END]

X_train, y_train = train[feature_columns], train["pm25_target"]
X_valid, y_valid = valid[feature_columns], valid["pm25_target"]
X_test, y_test = test[feature_columns], test["pm25_target"]

print(f"    Train: {len(train):,} dòng | Valid: {len(valid):,} dòng | Test: {len(test):,} dòng")

# 6. Huấn luyện trên tập Train và đánh giá trên Validation
print("--> 4. Đang huấn luyện Ensemble Voting Model trên tập Train...")
voting_pipeline.fit(X_train, y_train)

val_preds = voting_pipeline.predict(X_valid)
val_mae = mean_absolute_error(y_valid, val_preds)
val_rmse = mean_squared_error(y_valid, val_preds) ** 0.5
print(f"    Validation -> MAE: {val_mae:.3f} µg/m³ | RMSE: {val_rmse:.3f} µg/m³")

# 7. Tái huấn luyện trên cả Train + Validation trước khi xuất model
print("--> 5. Đang tái huấn luyện mô hình trên tập Train + Validation...")
train_valid = pd.concat([train, valid], ignore_index=True)
voting_pipeline.fit(train_valid[feature_columns], train_valid["pm25_target"])

# Đánh giá độc lập trên tập Test
test_preds = voting_pipeline.predict(X_test)
test_metrics = {
    "mae": round(float(mean_absolute_error(y_test, test_preds)), 3),
    "rmse": round(float(mean_squared_error(y_test, test_preds) ** 0.5), 3),
    "r2": round(float(r2_score(y_test, test_preds)), 4),
}
print(f"    Test -> MAE: {test_metrics['mae']} | RMSE: {test_metrics['rmse']} | R2: {test_metrics['r2']}")

# 8. Lưu Artifacts vào thư mục artifacts/
print("--> 6. Đang lưu Artifacts...")
joblib.dump(voting_pipeline, ARTIFACTS_DIR / "model.joblib", compress=3)

(ARTIFACTS_DIR / "feature_columns.json").write_text(
    json.dumps(feature_columns, indent=2), encoding="utf-8"
)

(ARTIFACTS_DIR / "metrics.json").write_text(
    json.dumps(test_metrics, indent=2), encoding="utf-8"
)

print(f"--> Hoàn tất! Toàn bộ file đã được lưu trong: {ARTIFACTS_DIR.resolve()}")