# File: forecast_app/schemas.py
from datetime import datetime
from pydantic import BaseModel

class ForecastRecord(BaseModel):
    station: str
    observation_time: datetime
    forecast_time: datetime
    observed_pm25: float | None
    predicted_pm25: float
    horizon_hours: int
    model_version: str
    source_session_id: str
    source_sequence: int

class ForecastResponse(BaseModel):
    generated_at: datetime
    count: int
    forecasts: list[ForecastRecord]