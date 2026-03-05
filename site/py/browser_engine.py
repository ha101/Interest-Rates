from __future__ import annotations

import math
import re
import uuid
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from interest_rate_meta_model.curves import (
    DEFAULT_DASHBOARD_MATURITIES,
    TENOR_TO_TAG,
    bootstrap_zero_curve_from_treasury_par_row,
    zero_curve_from_points,
)
from interest_rate_meta_model.meta import (
    MODEL_DESCRIPTIONS,
    InterestRateMetaModel,
    MarketData,
    RuntimeModelConfig,
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


def _normalize_short_rate_source(source: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9]+", "", str(source)).lower()
    aliases = {
        "auto": "auto",
        "treasury": "treasury",
        "sofr": "sofr",
        "effr": "effr",
        "fedfunds": "effr",
        "fedfundseffective": "effr",
        "effectivefedfunds": "effr",
        "effectivefederalfundsrate": "effr",
    }
    if clean not in aliases:
        raise ValueError("data source must be one of: auto, treasury, sofr, effr (or fed-funds aliases)")
    return aliases[clean]


def _normalize_treasury_tenor(tenor: str) -> str:
    clean = re.sub(r"\s+", "", str(tenor).upper())
    clean = clean.replace("MO", "M").replace("MON", "M").replace("YR", "Y")
    if clean in TENOR_TO_TAG:
        return clean
    raise ValueError(f"Unsupported Treasury tenor {tenor!r}. Choose one of: {', '.join(TENOR_TO_TAG)}")


def _find_as_of_index(dates: list[str], as_of: str) -> int:
    if not dates:
        raise ValueError("No Treasury dates found in cached data")
    if as_of in dates:
        return dates.index(as_of)
    for i in range(len(dates) - 1, -1, -1):
        if dates[i] <= as_of:
            return i
    return 0


def _build_treasury_frame(treasury_payload: dict[str, Any]) -> pd.DataFrame:
    dates = list(treasury_payload.get("dates") or [])
    tenors = list(treasury_payload.get("tenors") or [])
    rows = list(treasury_payload.get("par_yields") or [])
    if not dates or not tenors or not rows:
        raise ValueError("Treasury payload is missing required dates/tenors/par_yields data")
    if len(dates) != len(rows):
        raise ValueError("Treasury payload is malformed: dates and par_yields length mismatch")

    normalized_rows: list[dict[str, Any]] = []
    for date_s, row in zip(dates, rows):
        if not isinstance(row, list):
            raise ValueError("Treasury payload is malformed: par_yields must be a list of rows")
        record: dict[str, Any] = {"date": pd.Timestamp(date_s)}
        for idx, tenor in enumerate(tenors):
            value = row[idx] if idx < len(row) else None
            record[tenor] = None if value is None else float(value)
        normalized_rows.append(record)

    frame = pd.DataFrame(normalized_rows).sort_values("date").reset_index(drop=True)
    if frame.empty:
        raise ValueError("Treasury payload contains no usable rows")
    return frame


def _reference_rate_series(
    ref_payload: dict[str, Any],
    key: str,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.Series:
    dates = list(ref_payload.get("dates") or [])
    values = list(ref_payload.get(key) or [])
    if len(dates) != len(values):
        raise ValueError(f"Reference payload for {key!r} is malformed")
    rows = []
    for d, v in zip(dates, values):
        if v is None:
            continue
        ts = pd.Timestamp(d)
        if ts < start or ts > end:
            continue
        rows.append((ts, float(v) / 100.0))
    if not rows:
        raise ValueError(f"No {key} observations were available for the selected window")
    out = pd.Series({ts: val for ts, val in rows}).sort_index()
    out.name = f"{key}_short_rate_proxy"
    return out


def _treasury_short_rate_series(
    treasury_frame: pd.DataFrame,
    *,
    tenor: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.Series:
    tenor_key = _normalize_treasury_tenor(tenor)
    tag = TENOR_TO_TAG[tenor_key]
    if tag not in treasury_frame.columns:
        raise ValueError(f"Treasury source did not return tenor column {tag}")
    mask = (treasury_frame["date"] >= start) & (treasury_frame["date"] <= end)
    frame = treasury_frame.loc[mask, ["date", tag]].dropna()
    if frame.empty:
        raise ValueError("No short-rate history was available from Treasury for the selected window")
    out = pd.Series(frame[tag].to_numpy(dtype=float) / 100.0, index=pd.to_datetime(frame["date"]))
    out = out.sort_index()
    out.name = f"treasury_{tenor_key.lower()}_short_rate_proxy"
    return out


def _build_market_data(
    *,
    payload: dict[str, Any],
    treasury_frame: pd.DataFrame,
    ref_payload: dict[str, Any],
) -> tuple[MarketData, str, pd.Timestamp]:
    source_key = _normalize_short_rate_source(payload.get("data_source", "auto"))
    as_of_input = payload.get("as_of_date") or date.today().isoformat()
    as_of_ts = pd.Timestamp(as_of_input).normalize()

    all_dates = [pd.Timestamp(d).date().isoformat() for d in treasury_frame["date"].tolist()]
    as_of_idx = _find_as_of_index(all_dates, as_of_ts.date().isoformat())
    as_of_ts = pd.Timestamp(all_dates[as_of_idx]).normalize()

    history_years = float(payload.get("history_years", 10.0))
    start_ts = as_of_ts - pd.Timedelta(days=int(round(history_years * 366)))
    short_rate_tenor = payload.get("short_rate_tenor", "3M")

    mask = treasury_frame["date"] <= as_of_ts
    treasury_full = treasury_frame.loc[mask].copy()
    if treasury_full.empty:
        raise ValueError("No Treasury history was available on or before the requested as-of date")
    latest_row = treasury_full.iloc[-1]
    treasury_hist = treasury_full[treasury_full["date"] >= start_ts].copy()
    if treasury_hist.empty:
        treasury_hist = treasury_full.tail(1).copy()
    current_curve = bootstrap_zero_curve_from_treasury_par_row(latest_row.to_dict(), as_of=latest_row["date"])

    selected_source = source_key
    if source_key == "auto":
        last_error: Exception | None = None
        short_rates: pd.Series | None = None
        for candidate in ("sofr", "effr", "treasury"):
            try:
                if candidate == "treasury":
                    short_rates = _treasury_short_rate_series(
                        treasury_full,
                        tenor=short_rate_tenor,
                        start=start_ts,
                        end=as_of_ts,
                    )
                else:
                    short_rates = _reference_rate_series(ref_payload, candidate, start=start_ts, end=as_of_ts)
                selected_source = candidate
                break
            except Exception as exc:
                last_error = exc
        if short_rates is None:
            raise ValueError(f"Auto data-source selection failed: {last_error}")
    elif source_key == "treasury":
        short_rates = _treasury_short_rate_series(
            treasury_full,
            tenor=short_rate_tenor,
            start=start_ts,
            end=as_of_ts,
        )
    else:
        short_rates = _reference_rate_series(ref_payload, source_key, start=start_ts, end=as_of_ts)

    short_rates = short_rates.sort_index()
    if len(short_rates) < 2:
        raise ValueError("At least two short-rate observations are required to fit the models")

    market_data = MarketData(short_rates=short_rates, current_curve=current_curve, regime_features=None)
    market_data.metadata = {
        "selected_short_rate_source": selected_source,
        "treasury_latest_row": latest_row.to_dict(),
        "treasury_history_frame": treasury_hist,
        "curve_source": "treasury",
    }
    return market_data, selected_source, as_of_ts


def _fill_missing_short_rate_days(market_data: MarketData) -> MarketData:
    series = market_data.short_rates.sort_index()
    if not isinstance(series.index, pd.DatetimeIndex) or len(series) < 2:
        market_data.metadata["missing_days_filled"] = False
        return market_data
    bdays = pd.bdate_range(series.index.min(), series.index.max())
    filled = series.reindex(bdays).ffill()
    market_data.short_rates = filled.astype(float)
    market_data.metadata["missing_days_filled"] = len(filled) > len(series)
    market_data.metadata["history_points_original"] = int(len(series))
    market_data.metadata["history_points_filled"] = int(len(filled))
    return market_data


def _build_target_history(target: str, market_data: MarketData) -> pd.Series | None:
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


def _current_target_value(target: str, market_data: MarketData, current_bundle: dict[str, list[float]]) -> float | None:
    if target == "curve":
        return None
    if target == "short_rate":
        return float(market_data.short_rates.iloc[-1])
    latest_row = market_data.metadata.get("treasury_latest_row")
    tag_map = {"2y": "BC_2YEAR", "10y": "BC_10YEAR", "30y": "BC_30YEAR"}
    if latest_row is not None and tag_map[target] in latest_row and latest_row[tag_map[target]] is not None:
        return float(latest_row[tag_map[target]]) / 100.0
    mats = np.asarray(current_bundle["maturities"], dtype=float)
    vals = np.asarray(current_bundle["par"], dtype=float)
    target_mat = {"2y": 2.0, "10y": 10.0, "30y": 30.0}[target]
    return float(np.interp(target_mat, mats, vals))


def _predicted_target_value(target: str, prediction: Any, predicted_bundle: dict[str, list[float]]) -> float | None:
    if target == "curve":
        return None
    if target == "short_rate":
        return float(prediction.aggregate_short_rate)
    mats = np.asarray(predicted_bundle["maturities"], dtype=float)
    vals = np.asarray(predicted_bundle["par"], dtype=float)
    target_mat = {"2y": 2.0, "10y": 10.0, "30y": 30.0}[target]
    return float(np.interp(target_mat, mats, vals))


def _model_curve_bundles(ensemble: InterestRateMetaModel, horizon_years: float, maturities: np.ndarray) -> dict[str, dict[str, list[float]]]:
    bundles: dict[str, dict[str, list[float]]] = {}
    for model in ensemble.models:
        zero = model.predict_yield_curve(horizon_years, maturities)
        curve = zero_curve_from_points(maturities, zero, name=model.name)
        bundles[model.name] = curve.curve_bundle(maturities)
    return bundles


def _model_curve_bundles_from_prediction(prediction: Any) -> dict[str, dict[str, list[float]]]:
    bundles: dict[str, dict[str, list[float]]] = {}
    maturities = np.array([float(c.replace("y", "")) for c in prediction.model_curves.columns], dtype=float)
    for model, row in prediction.model_curves.iterrows():
        curve = zero_curve_from_points(maturities, row.to_numpy(dtype=float), name=model)
        bundles[model] = curve.curve_bundle(DEFAULT_DASHBOARD_MATURITIES)
    return bundles


def _confidence_summary(target: str, market_data: MarketData, prediction: Any, current_value: float | None) -> dict[str, Any]:
    weights = prediction.weights.set_index("model")["weight"]
    if target == "short_rate":
        model_values = prediction.model_short_rates.astype(float)
    else:
        target_mat = {"2y": 2.0, "10y": 10.0, "30y": 30.0, "curve": 10.0}.get(target, 10.0)
        model_values = pd.Series(
            {
                model: float(np.interp(target_mat, DEFAULT_DASHBOARD_MATURITIES, bundle["par"]))
                for model, bundle in _model_curve_bundles_from_prediction(prediction).items()
            }
        )
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


def _top_drivers(weights_df: pd.DataFrame) -> list[dict[str, Any]]:
    top = weights_df.sort_values("weight", ascending=False).head(3)
    out: list[dict[str, Any]] = []
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


def _model_parameters(model: Any) -> dict[str, Any]:
    skip = {"name", "traits", "current_curve", "diagnostics"}
    out: dict[str, Any] = {}
    for key, value in model.__dict__.items():
        if key.startswith("_") or key in skip:
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            out[key] = value
    return out


def _diagnostics_payload(ensemble: InterestRateMetaModel, prediction: Any) -> dict[str, Any]:
    rolling = ensemble.rolling_one_step_backtest(window=60)
    per_model: list[dict[str, Any]] = []
    weight_map = prediction.weights.set_index("model")["weight"].to_dict()
    for model in ensemble.models:
        summary = model.summary()
        params = _model_parameters(model)
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


def _source_label(selected_source: str, fallback_used: bool) -> str:
    if fallback_used and selected_source in {"sofr", "effr"}:
        rate_source = "FRED fallback"
    else:
        rate_source = {
            "treasury": "U.S. Treasury",
            "sofr": "New York Fed (SOFR)",
            "effr": "New York Fed (Fed Funds)",
        }.get(selected_source, str(selected_source))
    return f"Short rate: {rate_source}; Curve: U.S. Treasury"


def _data_health(market_data: MarketData, selected_source: str, fallback_used: bool) -> dict[str, Any]:
    last_obs = pd.Timestamp(market_data.short_rates.index.max()).date().isoformat() if len(market_data.short_rates) else None
    return {
        "last_observation_date": last_obs,
        "missing_days_filled": bool(market_data.metadata.get("missing_days_filled", False)),
        "source_used": _source_label(selected_source, fallback_used),
        "history_points": int(len(market_data.short_rates)),
    }


def _series_points(series: pd.Series, limit: int | None = 500) -> list[dict[str, Any]]:
    series = series.sort_index()
    if limit is not None and len(series) > limit:
        series = series.iloc[-limit:]
    return [{"date": str(pd.Timestamp(idx).date()), "value": float(val)} for idx, val in series.items()]


def _format_tenor(maturity: float) -> str:
    if maturity < 1.0:
        months = maturity * 12.0
        if abs(months - round(months)) < 1e-6:
            return f"{int(round(months))}M"
        return f"{months:.1f}M"
    if abs(maturity - round(maturity)) < 1e-6:
        return f"{int(round(maturity))}Y"
    return f"{maturity:.2f}Y"


def run_local(payload: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
    target = payload.get("target", "10y")
    if target not in TARGET_LABELS:
        raise ValueError("target must be one of: short_rate, 2y, 10y, 30y, curve")
    horizon_key = payload.get("horizon", "1m")
    if horizon_key not in HORIZON_TO_YEARS:
        raise ValueError("Unsupported horizon")

    treasury_payload = data.get("treasury_par_history", {})
    ref_payload = data.get("ref_rates", {})
    status_meta = data.get("status", {})

    market_data, selected_source, as_of_ts = _build_market_data(
        payload=payload,
        treasury_frame=_build_treasury_frame(treasury_payload),
        ref_payload=ref_payload,
    )
    market_data = _fill_missing_short_rate_days(market_data)

    weighting_method = payload.get("weighting_method", "performance")
    if weighting_method not in {"performance", "curve_fit", "forecast"}:
        weighting_method = "performance"
    optimization_mode = payload.get("optimization_mode", "fast")
    if optimization_mode not in {"fast", "accurate"}:
        optimization_mode = "fast"

    runtime_cfg = RuntimeModelConfig(
        optimization_mode=optimization_mode,
        mc_paths=payload.get("mc_paths"),
        random_seed=payload.get("random_seed"),
    )
    selected_models = payload.get("selected_models") or recommended_model_names()
    selected_models = [m for m in selected_models if m in MODEL_DESCRIPTIONS]
    if not selected_models:
        selected_models = recommended_model_names()

    ensemble = InterestRateMetaModel(
        model_names=selected_models,
        runtime_config=runtime_cfg,
        weighting_method=weighting_method,
    ).fit(market_data)

    maturities = np.asarray(payload.get("maturities") or DEFAULT_DASHBOARD_MATURITIES, dtype=float)
    horizon_years = HORIZON_TO_YEARS[horizon_key]
    prediction = ensemble.predict(horizon_years, maturities)

    dashboard_curve_maturities = DEFAULT_DASHBOARD_MATURITIES
    aggregate_curve = zero_curve_from_points(
        dashboard_curve_maturities,
        ensemble.predict(horizon_years, dashboard_curve_maturities).aggregate_curve,
        name="aggregate_predicted_curve",
    )
    current_bundle = market_data.current_curve.curve_bundle(dashboard_curve_maturities)
    predicted_bundle = aggregate_curve.curve_bundle(dashboard_curve_maturities)
    model_curve_bundles = _model_curve_bundles(ensemble, horizon_years, dashboard_curve_maturities)

    point_history = _build_target_history(target, market_data)
    current_value = _current_target_value(target, market_data, current_bundle)
    predicted_value = _predicted_target_value(target, prediction, predicted_bundle)
    confidence = _confidence_summary(target, market_data, prediction, current_value)
    top_drivers = _top_drivers(prediction.weights)
    diagnostics = _diagnostics_payload(ensemble, prediction)

    fallback_used = bool(status_meta.get("fallback_used", False))
    status = {
        "pill": "Using cached Pages data",
        "using_cache": True,
        "fallback_used": fallback_used,
        "source_used": _source_label(selected_source, fallback_used),
    }

    overview = {
        "target": target,
        "target_label": TARGET_LABELS[target],
        "predicted_level": float(predicted_value) if predicted_value is not None else None,
        "current_level": float(current_value) if current_value is not None else None,
        "change_bps": None if current_value is None or predicted_value is None else float((predicted_value - current_value) * 10000.0),
        "confidence": confidence,
        "top_drivers": top_drivers,
        "data_health": _data_health(market_data, selected_source, fallback_used),
    }

    future_date = (as_of_ts + pd.Timedelta(days=HORIZON_TO_DAYS[horizon_key])).date().isoformat()
    return {
        "run_id": uuid.uuid4().hex,
        "request": {
            "data_source": selected_source,
            "as_of_date": str(as_of_ts.date()),
            "horizon": horizon_key,
            "target": target,
            "history_years": float(payload.get("history_years", 10.0)),
            "short_rate_tenor": _normalize_treasury_tenor(payload.get("short_rate_tenor", "3M")),
            "allow_fred_fallback": bool(payload.get("allow_fred_fallback", True)),
            "cache_enabled": bool(payload.get("cache_enabled", True)),
            "selected_models": selected_models,
            "weighting_method": weighting_method,
            "optimization_mode": optimization_mode,
            "maturities": [float(m) for m in maturities],
        },
        "status": status,
        "overview": overview,
        "charts": {
            "target_history": None if point_history is None else _series_points(point_history),
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
            "tenor_labels": [_format_tenor(m) for m in dashboard_curve_maturities],
            "market": current_bundle,
            "ensemble": predicted_bundle,
            "models": model_curve_bundles,
        },
        "diagnostics": diagnostics,
        "data_cache": {
            "cache_dir": str(status_meta.get("cache_dir", "site/data")),
            "cache_enabled": bool(payload.get("cache_enabled", True)),
            "allow_fred_fallback": bool(payload.get("allow_fred_fallback", True)),
            "logs": list(status_meta.get("recent_logs") or []),
        },
    }


def run_scenario(payload: dict[str, Any]) -> dict[str, Any]:
    maturities = np.asarray(payload["maturities"], dtype=float)
    zero_curve = np.asarray(payload["zero_curve"], dtype=float)
    if maturities.size == 0 or zero_curve.size == 0 or maturities.size != zero_curve.size:
        raise ValueError("maturities and zero_curve must be non-empty arrays of equal length")

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

    shocks = np.interp(
        maturities,
        [float(maturities.min()), float(maturities.max())],
        [short_bps / 10000.0, long_bps / 10000.0],
    )
    shocked_zero = zero_curve + shocks
    base_curve = zero_curve_from_points(maturities, zero_curve, name="base")
    shocked_curve = zero_curve_from_points(maturities, shocked_zero, name="scenario")
    return {
        "scenario_type": scenario_type,
        "applied_shock": {"short_end_bps": short_bps, "long_end_bps": long_bps},
        "base": base_curve.curve_bundle(maturities),
        "shocked": shocked_curve.curve_bundle(maturities),
        "deltas_bps": [{"maturity": float(m), "delta_bps": float(s * 10000.0)} for m, s in zip(maturities, shocks)],
    }
