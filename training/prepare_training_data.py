# File: training/prepare_training_data.py
from pathlib import Path
import numpy as np
import pandas as pd

INPUT = Path("data/observations.parquet")
OUTPUT = Path("data/training_features.parquet")

print("1. Đang đọc dữ liệu từ observations.parquet...")
df = pd.read_parquet(INPUT)
df["source_datetime"] = pd.to_datetime(df["source_datetime"])
df = df.sort_values(["station", "source_datetime"]).reset_index(drop=True)

# Điền khuyết dữ liệu quá khứ cho từng trạm bằng ffill()
measurements = [
    "pm25", "pm10", "so2", "no2", "co", "o3",
    "temperature", "pressure", "dew_point", "rain",
    "wind_speed",
]
print("2. Đang điền khuyết dữ liệu (ffill)...")
df[measurements] = df.groupby("station")[measurements].ffill()

# Mã hóa chu kỳ thời gian bằng sin / cos
print("3. Đang tạo đặc trưng thời gian tuần hoàn (sin/cos)...")
dt = df["source_datetime"]
df["hour"] = dt.dt.hour
df["day_of_week"] = dt.dt.dayofweek
df["month"] = dt.dt.month

df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
df["month_sin"] = np.sin(2 * np.pi * (df["month"] - 1) / 12)
df["month_cos"] = np.cos(2 * np.pi * (df["month"] - 1) / 12)

# Tạo Lag và Rolling (Không gây rò rỉ dữ liệu)
print("4. Đang tạo Lag và Rolling statistics...")
grouped = df.groupby("station", group_keys=False)

for lag in [1, 2, 3, 6, 12, 24]:
    df[f"pm25_lag_{lag}"] = grouped["pm25"].shift(lag)

# Dịch chuyển lùi 1 bước trước khi tính rolling để không nhìn thấy PM2.5 hiện tại
past_pm25 = grouped["pm25"].shift(1)
for window in [3, 6, 12, 24]:
    df[f"pm25_roll_mean_{window}"] = (
        past_pm25.groupby(df["station"])
        .rolling(window).mean()
        .reset_index(level=0, drop=True)
    )
    df[f"pm25_roll_std_{window}"] = (
        past_pm25.groupby(df["station"])
        .rolling(window).std()
        .reset_index(level=0, drop=True)
    )

# Tạo target: PM2.5 ở thời điểm t + 1 giờ
HORIZON = 1
df["pm25_target"] = grouped["pm25"].shift(-HORIZON)

# Bỏ các dòng trống phát sinh do shift và lưu kết quả
df = df.dropna().reset_index(drop=True)
df.to_parquet(OUTPUT, index=False)
print(f"--> Thành công! Đã tạo file đặc trưng: {OUTPUT} ({len(df)} dòng)")