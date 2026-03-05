from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from starlette.requests import Request

from .curves import DEFAULT_DASHBOARD_MATURITIES
from .dashboard_service import DashboardService

LOGGER = logging.getLogger(__name__)


class RunRequest(BaseModel):
    data_source: Literal["auto", "treasury", "sofr", "effr"] = "auto"
    as_of_date: str | None = None
    horizon: Literal["1d", "1w", "1m", "3m", "1y"] = "1m"
    target: Literal["short_rate", "2y", "10y", "30y", "curve"] = "10y"
    history_years: float = 10.0
    short_rate_tenor: str = "3M"
    allow_fred_fallback: bool = True
    cache_enabled: bool = True
    force_refresh: bool = False
    weighting_method: Literal["performance", "curve_fit", "forecast"] = "performance"
    optimization_mode: Literal["fast", "accurate"] = "fast"
    mc_paths: int | None = None
    random_seed: int | None = None
    selected_models: list[str] | None = None
    maturities: list[float] = Field(default_factory=lambda: [float(x) for x in DEFAULT_DASHBOARD_MATURITIES.tolist()])


class ScenarioRequest(BaseModel):
    maturities: list[float]
    zero_curve: list[float]
    scenario_type: Literal["parallel_25bp", "steepen", "flatten", "custom"] = "parallel_25bp"
    short_end_bps: float | None = None
    long_end_bps: float | None = None


class CacheClearRequest(BaseModel):
    namespace: str | None = None


class RunOneRequest(RunRequest):
    model_config = {"protected_namespaces": ()}
    model_name: str


class SourceStatusRequest(BaseModel):
    allow_fred_fallback: bool = True
    cache_enabled: bool = True


TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"
STATIC_DIR = Path(__file__).resolve().parent / "static"


def _raise_api_error(exc: Exception, *, public_message: str) -> None:
    if isinstance(exc, HTTPException):
        raise exc
    if isinstance(exc, (ValueError, KeyError)):
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    LOGGER.exception("Unhandled dashboard API error")
    raise HTTPException(status_code=500, detail=public_message) from exc


def create_app(service: DashboardService | None = None) -> FastAPI:
    svc = service or DashboardService()
    app = FastAPI(title="Interest Rate Meta Model Dashboard API")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
        allow_credentials=True,
    )
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    templates = Jinja2Templates(directory=str(TEMPLATE_DIR))

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(request, "index.html", {})

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {"ok": True}

    @app.post("/run")
    async def run_dashboard(request: RunRequest) -> dict[str, Any]:
        try:
            return svc.run_dashboard(request.model_dump())
        except Exception as exc:
            _raise_api_error(exc, public_message="Run failed due to an internal error.")

    @app.post("/scenario")
    async def scenario(request: ScenarioRequest) -> dict[str, Any]:
        try:
            return svc.scenario(request.model_dump())
        except Exception as exc:
            _raise_api_error(exc, public_message="Scenario request failed due to an internal error.")

    @app.get("/sources/status")
    async def sources_status(allow_fred_fallback: bool = True, cache_enabled: bool = True) -> dict[str, Any]:
        try:
            return svc.source_status(allow_fred_fallback=allow_fred_fallback, cache_enabled=cache_enabled)
        except Exception as exc:
            _raise_api_error(exc, public_message="Unable to fetch source status due to an internal error.")

    @app.post("/cache/clear")
    async def cache_clear(request: CacheClearRequest) -> dict[str, Any]:
        try:
            return svc.clear_cache(request.namespace)
        except Exception as exc:
            _raise_api_error(exc, public_message="Unable to clear cache due to an internal error.")

    @app.get("/models")
    async def models() -> dict[str, Any]:
        return svc.available_models()

    @app.post("/models/run_one")
    async def run_one(request: RunOneRequest) -> dict[str, Any]:
        try:
            payload = request.model_dump()
            model_name = payload.pop("model_name")
            return svc.run_single_model(payload, model_name)
        except Exception as exc:
            _raise_api_error(exc, public_message="Single-model run failed due to an internal error.")

    return app


app = create_app()


def main() -> None:
    import uvicorn

    uvicorn.run("interest_rate_meta_model.api:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()
