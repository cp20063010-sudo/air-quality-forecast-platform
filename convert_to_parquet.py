# File: convert_to_parquet.py
from pathlib import Path
import pandas as pd

raw_path = Path("data/raw")
output_path = Path("data/observations.parquet")

csv_files = sorted(list(raw_path.glob("*.csv")))
print(f"--> Tìm thấy {len(csv_files)} file CSV. Đang đọc và gộp dữ liệu...")

dfs = []
for file in csv_files:
    print(f"    Đang đọc: {file.name}")
    dfs.append(pd.read_csv(file))

df = pd.concat(dfs, ignore_index=True)

# Ghép year, month, day, hour thành source_datetime
df["source_datetime"] = pd.to_datetime(df[["year", "month", "day", "hour"]])

# Đổi tên các cột cho khớp với đặc tả của Dự án 3
column_mapping = {
    "PM2.5": "pm25",
    "PM10": "pm10",
    "SO2": "so2",
    "NO2": "no2",
    "CO": "co",
    "O3": "o3",
    "TEMP": "temperature",
    "PRES": "pressure",
    "DEWP": "dew_point",
    "RAIN": "rain",
    "WSPM": "wind_speed",
}
df = df.rename(columns=column_mapping)

# Chuẩn hóa tên trạm về chữ thường
df["station"] = df["station"].astype(str).str.strip().str.lower()

# Sắp xếp theo trạm và thời gian
df = df.sort_values(["station", "source_datetime"]).reset_index(drop=True)

# Lưu thành file observations.parquet
df.to_parquet(output_path, index=False)
print(f"--> Thành công! Đã tạo file: {output_path} ({len(df)} dòng dữ liệu)")