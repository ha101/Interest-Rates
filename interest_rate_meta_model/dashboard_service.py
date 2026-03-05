from __future__ import annotations

import json
import math
import uuid
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from .cache import HTTPCache, HTTPCacheConfig
from .curves import DEFAULT_DASHBOARD_MATURITIES, ZeroCurve, zero_curve_from_points
from .data_sources import (
    FREDConfig,
    FREDClient,
    NewYorkFedReferenceRateClient,
    NewYorkFedReferenceRateConfig,
    TreasuryGovClient,
    TreasuryGovConfig,
    build_market_data_from_gov_sources,
    normalize_short_rate_source,
    normalize_treasury_tenor,
)
from .meta import (
    MODEL_DESCRIPTIONS,
    WEIGHTING_LABELS,
    InterestRateMetaModel,
    MarketData,
    RuntimeModelConfig,
    build_model,
    model_catalog,
    recommended_model_names,
)

HORIZON_TO_YEARS: dict[str, float] = {
    "1d": 1.0 / 252.0,
    "1w": 1.0 / 52.0,
    "1m": 1.0 / 12.0,
    "3m": 0.25,
    "1y": 1.0,
}

HORIZON_TO_DAYS: dict[str, int] = {"1d": 1, "1w": 7, "1m": 30, "3m": 91, "1y": 365}

TARGET_LABELS = {
    "short_rate": "Short rate",
    "2y": "2Y yield",
    "10y": "10Y yield",
    "30y": "30Y yield",
    "curve": "Curve",
}


@dataclass
class DashboardLogStore:
    base_dir: Path

    @property
    def log_path(self) -> Path:
        return self.base_dir / "dashboard_fetch_log.jsonl"

    def append(self, records: list[dict[str, Any]]) -> None:
        if not records:
            return
        self.base_dir.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as fh:
            for record in records:
                fh.write(json.dumps(record, default=str) + "\n")

    def tail(self, n: int = 10) -> list[dict[str, Any]]:
        if not self.log_path.exists():
            return []
        lines = self.log_path.read_text(encoding="utf-8").splitlines()[-n:]
        out = []
        for line in lines:
            try:
                out.append(json.loads(line))
            except Exception:
                continue
        return out


class DashboardService:
    def __init__(self, cache_dir: str | Path = "~/.cache/interest-rate-meta-model") -> None:
        self.cache_dir = Path(cache_dir).expanduser().resolve()
        self.log_store = DashboardLogStore(self.cache_dir)

    def _cache_config(self, enabled: bool = True) -> HTTPCacheConfig:
        return HTTPCacheConfig(base_dir=self.cache_dir, enabled=enabled)

    def _build_clients(
        self,
        *,
        cache_enabled: bool = True,
        allow_fred_fallback: bool = True,
    ) -> tuple[TreasuryGovClient, NewYorkFedReferenceRateClient, FREDClient]:
        cache_config = self._cache_config(enabled=cache_enabled)
        shared_cache = HTTPCache(cache_config)
        treasury = TreasuryGovClient(TreasuryGovConfig(cache_config=cache_config), cache=shared_cache)
        fred = FREDClient(FREDConfig(cache_config=cache_config), cache=shared_cache)
        nyfed = NewYorkFedReferenceRateClient(
            NewYorkFedReferenceRateConfig(allow_fred_fallback=allow_fred_fallback, cache_config=cache_config),
            cache=shared_cache,
            fred_client=fred,
        )
        return treasury, nyfed, fred

    def clear_cache(self, namespace: str | None = None) -> dict[str, Any]:
        cache = HTTPCache(self._cache_config(enabled=True))
        cache.clear(namespace)
        record = {
            "timestamp": pd.Timestamp.utcnow().isoformat(),
            "action": "cache.clear",
            "namespace": namespace,
            "message": f"Cleared cache{f' for {namespace}' if namespace else ''}.",
        }
        self.log_store.append([record])
        return {"ok": True, "namespace": namespace, "message": record["message"]}

    def source_status(self, *, allow_fred_fallback: bool = True, cache_enabled: bool = True) -> dict[str, Any]:
        cache = HTTPCache(self._cache_config(enabled=cache_enabled))
        summary = {}
        for namespace in ("treasury", "newyorkfed", "fred"):
            ns_dir = cache.base_dir / namespace
            files = list(ns_dir.glob("*.json")) if ns_dir.exists() else []
            latest = max((f.stat().st_mtime for f in files), default=None)
            summary[namespace] = {
                "entries": len(files),
                "last_updated": None if latest is None else pd.Timestamp(latest, unit="s", tz="UTC").isoformat(),
            }
        return {
            "supported_sources": [
                {"key": "treasury", "label": "Treasury curve (official)"},
                {"key": "sofr", "label": "SOFR (official)"},
                {"key": "effr", "label": "Fed Funds (official)"},
                {"key": "auto", "label": "Auto (official then fallback)"},
            ],
            "cache_dir": str(cache.base_dir),
            "cache_enabled": bool(cache_enabled),
            "allow_fred_fallback": bool(allow_fred_fallback),
            "cache_summary": summary,
            "recent_logs": self._humanize_logs(self.log_store.tail(10)),
        }

    def available_models(self) -> dict[str, Any]:
        return {
            "models": model_catalog(),
            "recommended": recommended_model_names(),
            "weighting_methods": [{"key": k, "label": v} for k, v in WEIGHTING_LABELS.items()],
        }

    def run_dashboard(self, params: dict[str, Any]) -> dict[str, Any]:
        data_source = normalize_short_rate_source(params.get("data_source", "auto"))
        as_of_date = params.get("as_of_date") or date.today().isoformat()
        target = params.get("target", "10y")
        if target not in TARGET_LABELS:
            raise ValueError("target must be one of: short_rate, 2y, 10y, 30y, curve")
        horizon_key = params.get("horizon", "1m")
        if horizon_key not in HORIZON_TO_YEARS:
            raise ValueError("Unsupported horizon")
        horizon_years = HORIZON_TO_YEARS[horizon_key]
        history_years = float(params.get("history_years", 10.0))
        short_rate_tenor = normalize_treasury_tenor(params.get("short_rate_tenor", "3M"))
        allow_fred_fallback = bool(params.get("allow_fred_fallback", True))
        cache_enabled = bool(params.get("cache_enabled", True))
        force_refresh = bool(params.get("force_refresh", False))
        weighting_method = params.get("weighting_method", "performance")
        optimization_mode = params.get("optimization_mode", "fast")
        maturities = np.asarray(params.get("maturities") or DEFAULT_DASHBOARD_MATURITIES, dtype=float)
        selected_models = params.get("selected_models") or recommended_model_names()
        selected_models = [m for m in selected_models if m in MODEL_DESCRIPTIONS]
        if not selected_models:
            selected_models = recommended_model_names()

        if force_refresh and cache_enabled:
            cache = HTTPCache(self._cache_config(enabled=True))
            for namespace in ("treasury", "newyorkfed", "fred"):
                cache.clear(namespace)

        treasury_client, nyfed_client, fred_client = self._build_clients(cache_enabled=cache_enabled, allow_fred_fallback=allow_fred_fallback)
        market_data = build_market_data_from_gov_sources(
            history_years=history_years,
            short_rate_tenor=short_rate_tenor,
            short_rate_source=data_source,
            as_of=as_of_date,
            treasury_client=treasury_client,
            reference_rate_client=nyfed_client,
        )
        market_data = self._fill_missing_short_rate_days(market_data)

        runtime_cfg = RuntimeModelConfig(
            optimization_mode=optimization_mode,
            mc_paths=params.get("mc_paths"),
            random_seed=params.get("random_seed"),
        )
        ensemble = InterestRateMetaModel(
            model_names=selected_models,
            runtime_config=runtime_cfg,
            weighting_method=weighting_method,
        ).fit(market_data)
        prediction = ensemble.predict(horizon_years, maturities)
        dashboard_curve_maturities = DEFAULT_DASHBOARD_MATURITIES
        aggregate_curve = zero_curve_from_points(dashboard_curve_maturities, ensemble.predict(horizon_years, dashboard_curve_maturities).aggregate_curve, name="aggregate_predicted_curve")
        current_curve = market_data.current_curve

        current_bundle = current_curve.curve_bundle(dashboard_curve_maturities)
        predicted_bundle = aggregate_curve.curve_bundle(dashboard_curve_maturities)
        model_curve_bundles = self._model_curve_bundles(ensemble, horizon_years, dashboard_curve_maturities)

        point_history = self._build_target_history(target, market_data)
        current_value = self._current_target_value(target, market_data, current_bundle)
        predicted_value = self._predicted_target_value(target, prediction, predicted_bundle)
        confidence = self._confidence_summary(target, market_data, prediction, predicted_bundle, current_value)
        top_drivers = self._top_drivers(prediction.weights)
        diagnostics = self._diagnostics_payload(ensemble, prediction)

        events = self._collect_events(treasury_client, nyfed_client, fred_client)
        self.log_store.append(events)
        status = self._status_from_events(events, market_data)

        overview = {
            "target": target,
            "target_label": TARGET_LABELS[target],
            "predicted_level": float(predicted_value) if predicted_value is not None else None,
            "current_level": float(current_value) if current_value is not None else None,
            "change_bps": None if current_value is None or predicted_value is None else float((predicted_value - current_value) * 10000.0),
            "confidence": confidence,
            "top_drivers": top_drivers,
            "data_health": self._data_health(market_data, events),
        }

        future_date = (pd.Timestamp(as_of_date) + pd.Timedelta(days=HORIZON_TO_DAYS[horizon_key])).date().isoformat()
        response = {
            "run_id": uuid.uuid4().hex,
            "request": {
                "data_source": data_source,
                "as_of_date": str(pd.Timestamp(as_of_date).date()),
                "horizon": horizon_key,
                "target": target,
                "history_years": history_years,
                "short_rate_tenor": short_rate_tenor,
                "allow_fred_fallback": allow_fred_fallback,
                "cache_enabled": cache_enabled,
                "selected_models": selected_models,
                "weighting_method": weighting_method,
                "optimization_mode": optimization_mode,
                "maturities": [float(m) for m in maturities],
            },
            "status": status,
            "overview": overview,
            "charts": {
                "target_history": None if point_history is None else self._series_points(point_history),
                "target_forecast": {
                    "date": future_date,
                    "value": None if predicted_value is None else float(predicted_value),
                    "lower": confidence["lower"],
                    "upper": confidence["upper"],
                },
                "current_curve": current_bundle,
                "predicted_curve": predicted_bundle,
                "model_curves": model_curve_bundles,
            },
            "ensemble": {
                "aggregate_short_rate": float(prediction.aggregate_short_rate),
                "aggregate_curve_zero": {str(float(m)): float(y) for m, y in zip(prediction.maturities, prediction.aggregate_curve)},
                "model_short_rates": {k: float(v) for k, v in prediction.model_short_rates.to_dict().items()},
                "weights": prediction.weights.to_dict(orient="records"),
            },
            "curve_explorer": {
                "curve_types": ["par", "zero", "forward"],
                "tenor_labels": [self._format_tenor(m) for m in dashboard_curve_maturities],
                "market": current_bundle,
                "ensemble": predicted_bundle,
                "models": model_curve_bundles,
            },
            "diagnostics": diagnostics,
            "data_cache": {
                "cache_dir": str(self.cache_dir),
                "cache_enabled": cache_enabled,
                "allow_fred_fallback": allow_fred_fallback,
                "logs": self._humanize_logs(self.log_store.tail(10)),
            },
        }
        return response

    def run_single_model(self, params: dict[str, Any], model_name: str) -> dict[str, Any]:
        params = dict(params)
        params["selected_models"] = [model_name]
        response = self.run_dashboard(params)
        response["single_model"] = model_name
        return response

    def scenario(self, payload: dict[str, Any]) -> dict[str, Any]:
        maturities = np.asarray(payload["maturities"], dtype=float)
        zero_curve = np.asarray(payload["zero_curve"], dtype=float)
        scenario_type = payload.get("scenario_type", "parallel_25bp")
        if scenario_type == "parallel_25bp":
            short_bps, long_bps = 25.0, 25.0
        elif scenario_type == "steepen":
            short_bps, long_bps = 25.0, 0.0
        elif scenario_type == "flatten":
            short_bps, long_bps = 0.0, -25.0
        else:
            short_bps = float(payload.get("short_end_bps", 0.0))
            long_bps = float(payload.get("long_end_bps", 0.0))
        shocks = np.interp(maturities, [maturities.min(), maturities.max()], [short_bps / 10000.0, long_bps / 10000.0])
        shocked_zero = zero_curve + shocks
        base_curve = zero_curve_from_points(maturities, zero_curve, name="base")
        shocked_curve = zero_curve_from_points(maturities, shocked_zero, name="scenario")
        return {
            "scenario_type": scenario_type,
            "applied_shock": {"short_end_bps": short_bps, "long_end_bps": long_bps},
            "base": base_curve.curve_bundle(maturities),
            "shocked": shocked_curve.curve_bundle(maturities),
            "deltas_bps": [
                {"maturity": float(m), "delta_bps": float(s * 10000.0)} for m, s in zip(maturities, shocks)
            ],
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _fill_missing_short_rate_days(self, market_data: MarketData) -> MarketData:
        series = market_data.short_rates.sort_index()
        if not isinstance(series.index, pd.DatetimeIndex) or len(series) < 2:
            market_data.metadata["missing_days_filled"] = False
            return market_data
        bdays = pd.bdate_range(series.index.min(), series.index.max())
        filled = series.reindex(bdays).ffill()
        filled_flag = len(filled) > len(series)
        market_data.short_rates = filled.astype(float)
        market_data.metadata["missing_days_filled"] = filled_flag
        market_data.metadata["history_points_original"] = int(len(series))
        market_data.metadata["history_points_filled"] = int(len(filled))
        return market_data

    def _build_target_history(self, target: str, market_data: MarketData) -> pd.Series | None:
        if target == "curve":
            return None
        if target == "short_rate":
            return market_data.short_rates
        treasury_history = market_data.metadata.get("treasury_history_frame")
        if isinstance(treasury_history, pd.DataFrame):
            tag_map = {"2y": "BC_2YEAR", "10y": "BC_10YEAR", "30y": "BC_30YEAR"}
            tag = tag_map[target]
            series = (
                treasury_history[["date", tag]]
                .dropna()
                .rename(columns={tag: "rate"})
                .set_index("date")["rate"]
                .astype(float)
                / 100.0
            )
            return series.sort_index()
        return None

    def _current_target_value(self, target: str, market_data: MarketData, current_bundle: dict[str, list[float]]) -> float | None:
        if target == "curve":
            return None
        if target == "short_rate":
            return float(market_data.short_rates.iloc[-1])
        latest_row = market_data.metadata.get("treasury_latest_row")
        tag_map = {"2y": "BC_2YEAR", "10y": "BC_10YEAR", "30y": "BC_30YEAR"}
        if latest_row is not None and tag_map[target] in latest_row:
            return float(latest_row[tag_map[target]]) / 100.0
        mats = np.asarray(current_bundle["maturities"])
        vals = np.asarray(current_bundle["par"])
        target_mat = {"2y": 2.0, "10y": 10.0, "30y": 30.0}[target]
        return float(np.interp(target_mat, mats, vals))

    def _predicted_target_value(self, target: str, prediction: Any, predicted_bundle: dict[str, list[float]]) -> float | None:
        if target == "curve":
            return None
        if target == "short_rate":
            return float(prediction.aggregate_short_rate)
        mats = np.asarray(predicted_bundle["maturities"])
        vals = np.asarray(predicted_bundle["par"])
        target_mat = {"2y": 2.0, "10y": 10.0, "30y": 30.0}[target]
        return float(np.interp(target_mat, mats, vals))

    def _confidence_summary(self, target: str, market_data: MarketData, prediction: Any, predicted_bundle: dict[str, list[float]], current_value: float | None) -> dict[str, Any]:
        weights = prediction.weights.set_index("model")["weight"]
        if target == "short_rate":
            model_values = prediction.model_short_rates.astype(float)
        else:
            target_mat = {"2y": 2.0, "10y": 10.0, "30y": 30.0, "curve": 10.0}.get(target, 10.0)
            model_values = pd.Series({
                model: float(np.interp(target_mat, DEFAULT_DASHBOARD_MATURITIES, bundle["par"]))
                for model, bundle in self._model_curve_bundles_from_prediction(prediction).items()
            })
        common = model_values.index.intersection(weights.index)
        mean = float(np.sum(model_values[common] * weights[common])) if len(common) else float(model_values.mean())
        dispersion = float(np.sqrt(np.sum(weights[common] * (model_values[common] - mean) ** 2))) if len(common) else 0.0
        short_diff = market_data.short_rates.diff().dropna()
        step_vol = float(short_diff.std(ddof=0)) if len(short_diff) else 0.0
        target_scale = {"short_rate": 1.0, "2y": 0.75, "10y": 0.55, "30y": 0.45, "curve": 0.60}[target]
        interval_half = max(dispersion, 1.64 * step_vol * math.sqrt(max(1.0, 21.0 * target_scale)))
        if current_value is not None:
            interval_half = max(interval_half, abs(mean - current_value) * 0.35)
        label = "high" if interval_half <= 0.0010 else "medium" if interval_half <= 0.0025 else "low"
        return {
            "label": label,
            "lower": float(mean - interval_half),
            "upper": float(mean + interval_half),
            "half_width_bps": float(interval_half * 10000.0),
        }

    def _top_drivers(self, weights_df: pd.DataFrame) -> list[dict[str, Any]]:
        top = weights_df.sort_values("weight", ascending=False).head(3)
        out = []
        for _, row in top.iterrows():
            info = MODEL_DESCRIPTIONS.get(row["model"], {})
            out.append(
                {
                    "model": row["model"],
                    "label": info.get("label", row["model"]),
                    "weight": float(row["weight"]),
                    "description": info.get("short_description", ""),
                    "note": info.get("notes", ""),
                }
            )
        return out

    def _diagnostics_payload(self, ensemble: InterestRateMetaModel, prediction: Any) -> dict[str, Any]:
        rolling = ensemble.rolling_one_step_backtest(window=60)
        per_model = []
        weight_map = prediction.weights.set_index("model")["weight"].to_dict()
        for model in ensemble.models:
            summary = model.summary()
            params = self._model_parameters(model)
            info = MODEL_DESCRIPTIONS.get(model.name, {})
            per_model.append(
                {
                    "model": model.name,
                    "label": info.get("label", model.name),
                    "weight": float(weight_map.get(model.name, 0.0)),
                    "curve_fit_rmse_bps": None if not np.isfinite(model.diagnostics.current_curve_rmse) else float(model.diagnostics.current_curve_rmse * 10000.0),
                    "forecast_rmse_bps": float(model.diagnostics.historical_one_step_rmse * 10000.0),
                    "parameters": params,
                    "notes": info.get("notes", ""),
                    "traits": summary["traits"],
                }
            )
        return {
            "ensemble_health": {
                "weights": prediction.weights.to_dict(orient="records"),
                "rolling_backtest_error": [
                    {"date": str(pd.Timestamp(row["date"]).date()), "rolling_rmse": float(row["rolling_rmse"])}
                    for _, row in rolling.iterrows()
                ],
                "weighting_method": ensemble.weighting_method,
            },
            "per_model": per_model,
        }

    def _data_health(self, market_data: MarketData, events: list[dict[str, Any]]) -> dict[str, Any]:
        last_obs = pd.Timestamp(market_data.short_rates.index.max()).date().isoformat() if len(market_data.short_rates) else None
        source_used = self._source_label(market_data, events)
        return {
            "last_observation_date": last_obs,
            "missing_days_filled": bool(market_data.metadata.get("missing_days_filled", False)),
            "source_used": source_used,
            "history_points": int(len(market_data.short_rates)),
        }

    def _model_curve_bundles(self, ensemble: InterestRateMetaModel, horizon_years: float, maturities: np.ndarray) -> dict[str, dict[str, list[float]]]:
        bundles = {}
        for model in ensemble.models:
            zero = model.predict_yield_curve(horizon_years, maturities)
            curve = zero_curve_from_points(maturities, zero, name=model.name)
            bundles[model.name] = curve.curve_bundle(maturities)
        return bundles

    def _model_curve_bundles_from_prediction(self, prediction: Any) -> dict[str, dict[str, list[float]]]:
        bundles = {}
        for model, row in prediction.model_curves.iterrows():
            maturities = np.array([float(c.replace("y", "")) for c in prediction.model_curves.columns], dtype=float)
            curve = zero_curve_from_points(maturities, row.to_numpy(dtype=float), name=model)
            bundles[model] = curve.curve_bundle(DEFAULT_DASHBOARD_MATURITIES)
        return bundles

    def _model_parameters(self, model: Any) -> dict[str, Any]:
        skip = {"name", "traits", "current_curve", "diagnostics"}
        out = {}
        for key, value in model.__dict__.items():
            if key.startswith("_") or key in skip:
                continue
            if isinstance(value, (str, int, float, bool)) or value is None:
                out[key] = value
        return out

    def _collect_events(self, *clients: Any) -> list[dict[str, Any]]:
        events = []
        for client in clients:
            for event in getattr(client, "fetch_events", []):
                events.append(event)
            fred_client = getattr(client, "fred_client", None)
            if fred_client is not None:
                for event in getattr(fred_client, "fetch_events", []):
                    events.append(event)
        return events

    def _status_from_events(self, events: list[dict[str, Any]], market_data: MarketData) -> dict[str, Any]:
        if any(event.get("fallback_used") for event in events):
            pill = "Fallback used"
        elif events and all(event.get("from_cache") is True for event in events if "from_cache" in event):
            pill = "Using cache"
        else:
            pill = "Refreshed just now"
        return {
            "pill": pill,
            "using_cache": any(event.get("from_cache") for event in events),
            "fallback_used": any(event.get("fallback_used") for event in events),
            "source_used": self._source_label(market_data, events),
        }

    def _source_label(self, market_data: MarketData, events: list[dict[str, Any]]) -> str:
        if any(event.get("fallback_used") for event in events):
            rate_source = "FRED fallback"
        else:
            selected = market_data.metadata.get("selected_short_rate_source", "treasury")
            rate_source = {
                "treasury": "U.S. Treasury",
                "sofr": "New York Fed (SOFR)",
                "effr": "New York Fed (Fed Funds)",
            }.get(selected, str(selected))
        return f"Short rate: {rate_source}; Curve: U.S. Treasury"

    def _series_points(self, series: pd.Series, limit: int | None = 500) -> list[dict[str, Any]]:
        series = series.sort_index()
        if limit is not None and len(series) > limit:
            series = series.iloc[-limit:]
        return [{"date": str(pd.Timestamp(idx).date()), "value": float(val)} for idx, val in series.items()]

    def _humanize_logs(self, logs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out = []
        for record in logs:
            action = record.get("action") or record.get("namespace") or "fetch"
            msg = record.get("message")
            if msg is None:
                if record.get("fallback_used"):
                    msg = f"{action}: used fallback source."
                elif record.get("from_cache") is True:
                    msg = f"{action}: served from cache."
                else:
                    msg = f"{action}: refreshed from source."
            out.append({"timestamp": record.get("timestamp"), "message": msg, "details": record})
        return out

    def _format_tenor(self, maturity: float) -> str:
        if maturity < 1.0:
            months = maturity * 12.0
            if abs(months - round(months)) < 1e-6:
                return f"{int(round(months))}M"
            return f"{months:.1f}M"
        if abs(maturity - round(maturity)) < 1e-6:
            return f"{int(round(maturity))}Y"
        return f"{maturity:.2f}Y"
