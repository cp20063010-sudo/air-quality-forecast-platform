# File: forecast_app/feature_builder.py
import numpy as np
import pandas as pd

LAGS = [1, 2, 3, 6, 12, 24]
WINDOWS = [3, 6, 12, 24]

def build_latest_feature(station_frame: pd.DataFrame):
    frame = station_frame.sort_values("source_datetime")
    # Cần tối thiểu 24 giờ quá khứ và 1 giờ hiện tại (25 dòng)
    if len(frame) < 25:
        return None

    latest = frame.iloc[-1].copy()
    timestamp = pd.Timestamp(latest["source_datetime"])

    # 1. Tính toán sin/cos cho chu kỳ thời gian
    latest["hour_sin"] = np.sin(2 * np.pi * timestamp.hour / 24)
    latest["hour_cos"] = np.cos(2 * np.pi * timestamp.hour / 24)
    latest["dow_sin"] = np.sin(2 * np.pi * timestamp.dayofweek / 7)
    latest["dow_cos"] = np.cos(2 * np.pi * timestamp.dayofweek / 7)
    latest["month_sin"] = np.sin(2 * np.pi * (timestamp.month - 1) / 12)
    latest["month_cos"] = np.cos(2 * np.pi * (timestamp.month - 1) / 12)

    # 2. Lấy các giá trị trễ (Lag)
    for lag in LAGS:
        latest[f"pm25_lag_{lag}"] = frame["pm25"].iloc[-1 - lag]

    # 3. Tính rolling statistics từ quá khứ (không tính giờ hiện tại)
    for window in WINDOWS:
        past = frame["pm25"].iloc[-window - 1 : -1]
        latest[f"pm25_roll_mean_{window}"] = past.mean()
        latest[f"pm25_roll_std_{window}"] = past.std()

    return latest