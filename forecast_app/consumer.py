# File: forecast_app/consumer.py
import asyncio
from collections import deque
from datetime import datetime, timedelta
import json
import pandas as pd
import websockets
from forecast_app import config
from forecast_app.feature_builder import build_latest_feature
from forecast_app.schemas import ForecastRecord

class RealtimeState:
    def __init__(self, model_service):
        self.model_service = model_service
        self.buffers: dict[str, deque] = {}
        self.latest_forecasts_map: dict[str, ForecastRecord] = {}
        self.latest_observations_map: dict[str, dict] = {}
        self.current_session_id: str | None = None
        self.last_sequence: int = -1
        self.connected: bool = False

    def append_observation(self, station: str, row: dict):
        if station not in self.buffers:
            self.buffers[station] = deque(maxlen=config.BUFFER_HOURS)
        self.buffers[station].append(row)
        self.latest_observations_map[station] = row

    async def update_forecasts(self, session_id: str, sequence: int, source_datetime: str):
        if session_id == self.current_session_id and sequence <= self.last_sequence:
            return

        if self.current_session_id != session_id:
            print(f"--> [Dự án 1] Nhận session mới: {session_id}. Reset buffer...")
            self.current_session_id = session_id
            self.buffers.clear()

        self.last_sequence = sequence
        obs_dt = datetime.fromisoformat(source_datetime)
        forecast_dt = obs_dt + timedelta(hours=config.FORECAST_HORIZON)

        valid_stations = []
        feature_rows = []

        for station, records in self.buffers.items():
            if len(records) >= 25:
                frame = pd.DataFrame(list(records))
                feat = build_latest_feature(frame)
                if feat is not None:
                    valid_stations.append(station)
                    feature_rows.append(feat.to_dict())

        if feature_rows:
            predictions = self.model_service.predict(feature_rows)
            for st, pred in zip(valid_stations, predictions):
                latest_obs = self.latest_observations_map.get(st, {})
                record = ForecastRecord(
                    station=st,
                    observation_time=obs_dt,
                    forecast_time=forecast_dt,
                    observed_pm25=latest_obs.get("pm25"),
                    predicted_pm25=round(pred, 2),
                    horizon_hours=config.FORECAST_HORIZON,
                    model_version=config.MODEL_VERSION,
                    source_session_id=session_id,
                    source_sequence=sequence,
                )
                self.latest_forecasts_map[st] = record
            print(f"--> [Dự báo] Sequence {sequence}: Đã tính dự báo thành công cho {len(valid_stations)} trạm!")
        else:
            first_len = len(next(iter(self.buffers.values()))) if self.buffers else 0
            print(f"--> [Buffer] Đang tích lũy: {first_len}/25 mốc giờ (Sequence {sequence})...")

async def consume_forever(state: RealtimeState):
    ws_url = f"{config.OBSERVATION_WS_URL}/ws/v1/live"
    print(f"--> Đang kết nối tới WebSocket Dự án 1: {ws_url}")
    while True:
        try:
            state.connected = False
            async with websockets.connect(ws_url, ping_interval=20, ping_timeout=20) as ws:
                state.connected = True
                print("--> [WebSocket] KẾT NỐI THÀNH CÔNG! Đang đón nhận luồng quan trắc...")
                async for message in ws:
                    event = json.loads(message)

                    # Bỏ qua các gói tin bắt tay/chào mừng hoặc gói tin không chứa dữ liệu quan trắc
                    if not isinstance(event, dict) or "source_datetime" not in event or "observations" not in event:
                        event_type = event.get("event_type", "handshake/ping")
                        print(f"--> [WebSocket] Bỏ qua gói tin không phải quan trắc: {event_type}")
                        continue

                    source_dt = event["source_datetime"]
                    session_id = event.get("session_id", "default_session")
                    sequence = event.get("sequence", 0)

                    for observation in event.get("observations", []):
                        station = observation["station"].strip().lower()
                        obs_clean = {**observation, "station": station}
                        row = {**obs_clean, "source_datetime": source_dt}
                        state.append_observation(station, row)

                    await state.update_forecasts(
                        session_id=session_id,
                        sequence=sequence,
                        source_datetime=source_dt,
                    )
        except Exception as e:
            state.connected = False
            print(f"--> [Cảnh báo WebSocket] Mất kết nối ({e}). Thử lại sau 3 giây...")
            await asyncio.sleep(3)