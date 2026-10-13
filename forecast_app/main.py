# File: forecast_app/main.py
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime
import json
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from forecast_app import config
from forecast_app.consumer import RealtimeState, consume_forever
from forecast_app.model_service import ModelService
from forecast_app.schemas import ForecastResponse

@asynccontextmanager
async def lifespan(app: FastAPI):
    model_service = ModelService(config.MODEL_PATH, config.FEATURE_COLUMNS_PATH)
    state = RealtimeState(model_service)
    app.state.state = state
    app.state.model_service = model_service
    consumer_task = asyncio.create_task(consume_forever(state))
    yield
    consumer_task.cancel()

app = FastAPI(title="Air Quality Forecast Service", version="1.0.0", lifespan=lifespan)

# Gắn thư mục giao diện tĩnh (HTML/CSS/JS)
app.mount("/static", StaticFiles(directory="forecast_app/static"), name="static")
templates = Jinja2Templates(directory="forecast_app/templates")

@app.get("/", include_in_schema=False)
def dashboard_view(request: Request):
    return templates.TemplateResponse(request, name="dashboard.html")

@app.get("/health")
def health_check():
    state: RealtimeState = app.state.state
    ready_stations = sum(1 for b in state.buffers.values() if len(b) >= 25)
    return {
        "status": "healthy",
        "model_loaded": True,
        "consumer_connected": state.connected,
        "source_session_id": state.current_session_id,
        "last_sequence": state.last_sequence,
        "buffer_ready_stations": ready_stations,
        "is_warming_up": ready_stations < 12,
    }

@app.get("/api/v1/model")
def get_model_metadata():
    metrics = json.loads(config.METRICS_PATH.read_text(encoding="utf-8"))
    return {
        "version": config.MODEL_VERSION,
        "horizon_hours": config.FORECAST_HORIZON,
        "metrics": metrics,
    }

@app.get("/api/v1/forecasts/latest", response_model=ForecastResponse)
def get_latest_forecasts():
    state: RealtimeState = app.state.state
    forecasts = list(state.latest_forecasts_map.values())
    return ForecastResponse(
        generated_at=datetime.utcnow(),
        count=len(forecasts),
        forecasts=forecasts,
    )