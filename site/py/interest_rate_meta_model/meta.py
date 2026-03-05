from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Literal

import numpy as np
import pandas as pd

from .curves import ZeroCurve
from .models import (
    CIRModel,
    CIRPlusPlusModel,
    DothanModel,
    ExponentiatedVasicekModel,
    GaussianHJMModel,
    HoLeeModel,
    HullWhiteModel,
    InterestRateModel,
    VasicekModel,
)


MODEL_DESCRIPTIONS: dict[str, dict[str, Any]] = {
    "vasicek": {
        "label": "Vasicek",
        "short_description": "Gaussian mean reversion with closed-form bond pricing.",
        "notes": "Fast, interpretable, and analytically convenient, but it allows negative rates.",
        "recommended": True,
    },
    "dothan": {
        "label": "Dothan",
        "short_description": "Positive lognormal short-rate model.",
        "notes": "Preserves positivity, but is less tractable and tends to fit curves less cleanly.",
        "recommended": False,
    },
    "exp_vasicek": {
        "label": "Exponentiated Vasicek",
        "short_description": "Mean reversion in log-rates with positive levels.",
        "notes": "Keeps rates positive and mean reverting, but bond pricing is not affine and is handled numerically here.",
        "recommended": False,
    },
    "cir": {
        "label": "CIR",
        "short_description": "Positive affine mean-reverting short-rate model.",
        "notes": "Strong default for positive-rate regimes: mean reverting, analytic, and often stable.",
        "recommended": True,
    },
    "ho_lee": {
        "label": "Ho-Lee",
        "short_description": "Exact initial-curve fit with Gaussian dynamics.",
        "notes": "Fits the starting curve exactly, but does not mean revert.",
        "recommended": False,
    },
    "hull_white": {
        "label": "Hull-White",
        "short_description": "Mean-reverting Gaussian model with exact initial-curve fit.",
        "notes": "A very practical blend of curve-fit flexibility and analytic tractability.",
        "recommended": True,
    },
    "cir_pp": {
        "label": "CIR++",
        "short_description": "Deterministic-shift extension of CIR for exact curve fitting.",
        "notes": "Keeps much of CIR's structure while matching the initial term structure more closely.",
        "recommended": True,
    },
    "hjm": {
        "label": "Gaussian HJM",
        "short_description": "Forward-curve model with flexible curve dynamics.",
        "notes": "Flexible and curve-aware, but less parsimonious than classic short-rate models.",
        "recommended": True,
    },
}

WEIGHTING_LABELS = {
    "performance": "Performance-based",
    "curve_fit": "More curve-fit emphasis",
    "forecast": "More forecast emphasis",
}


@dataclass
class RuntimeModelConfig:
    optimization_mode: Literal["fast", "accurate"] = "fast"
    mc_paths: int | None = None
    random_seed: int | None = None

    def resolved_mc_paths(self) -> int:
        if self.mc_paths is not None:
            return int(self.mc_paths)
        return 1500 if self.optimization_mode == "fast" else 5000


@dataclass
class MarketData:
    short_rates: pd.Series
    current_curve: ZeroCurve
    regime_features: dict[str, float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EnsemblePrediction:
    horizon_years: float
    maturities: np.ndarray
    aggregate_short_rate: float
    aggregate_curve: np.ndarray
    weights: pd.DataFrame
    model_curves: pd.DataFrame
    model_short_rates: pd.Series
    diagnostics: dict[str, Any]


def build_model(model_name: str, runtime: RuntimeModelConfig | None = None) -> InterestRateModel:
    runtime = runtime or RuntimeModelConfig()
    mc_paths = runtime.resolved_mc_paths()
    seed = runtime.random_seed if runtime.random_seed is not None else 7
    builders: dict[str, Any] = {
        "vasicek": lambda: VasicekModel(),
        "dothan": lambda: DothanModel(mc_paths=mc_paths, seed=seed),
        "exp_vasicek": lambda: ExponentiatedVasicekModel(mc_paths=mc_paths, seed=seed + 17),
        "cir": lambda: CIRModel(),
        "ho_lee": lambda: HoLeeModel(),
        "hull_white": lambda: HullWhiteModel(),
        "cir_pp": lambda: CIRPlusPlusModel(),
        "hjm": lambda: GaussianHJMModel(),
    }
    if model_name not in builders:
        raise KeyError(f"Unknown model: {model_name}")
    return builders[model_name]()


def model_catalog() -> list[dict[str, Any]]:
    rows = []
    for name, info in MODEL_DESCRIPTIONS.items():
        rows.append({"name": name, **info})
    return rows


def recommended_model_names() -> list[str]:
    return [name for name, info in MODEL_DESCRIPTIONS.items() if info.get("recommended")]


class InterestRateMetaModel:
    """Dynamic ensemble over the Chapter 3 interest-rate models.

    The weighting rule combines historical fit, current-curve fit, structural traits,
    and optional regime features. Weighting profiles can tilt the ensemble toward
    curve-fit or forecast performance without changing the underlying models.
    """

    def __init__(
        self,
        model_names: Iterable[str] | None = None,
        *,
        runtime_config: RuntimeModelConfig | None = None,
        weighting_method: Literal["performance", "curve_fit", "forecast"] = "performance",
    ) -> None:
        self.model_names = list(model_names or recommended_model_names())
        self.runtime_config = runtime_config or RuntimeModelConfig()
        self.weighting_method = weighting_method
        self.models: list[InterestRateModel] = []
        self.market_data: MarketData | None = None
        self.weight_breakdown_: pd.DataFrame | None = None

    def fit(self, market_data: MarketData, dt: float | None = None) -> "InterestRateMetaModel":
        self.market_data = market_data
        self.models = []
        for model_name in self.model_names:
            model = build_model(model_name, self.runtime_config)
            model.fit(market_data.short_rates, current_curve=market_data.current_curve, dt=dt)
            self.models.append(model)
        self.weight_breakdown_ = self._compute_weights(horizon_years=1.0)
        return self

    def _profile(self) -> dict[str, float]:
        if self.weighting_method == "curve_fit":
            return {
                "hist_mult": 0.8,
                "curve_mult": 1.25,
                "curve_fit_bonus": 0.18,
                "forecast_bonus": 0.0,
                "flex_bonus": 0.08,
            }
        if self.weighting_method == "forecast":
            return {
                "hist_mult": 1.25,
                "curve_mult": 0.45,
                "curve_fit_bonus": 0.0,
                "forecast_bonus": 0.14,
                "flex_bonus": 0.04,
            }
        return {
            "hist_mult": 1.0,
            "curve_mult": 0.75,
            "curve_fit_bonus": 0.05,
            "forecast_bonus": 0.05,
            "flex_bonus": 0.05,
        }

    def _compute_weights(self, horizon_years: float) -> pd.DataFrame:
        if not self.models:
            raise ValueError("fit the meta-model before computing weights")
        if self.market_data is None:
            raise ValueError("market data unavailable")

        profile = self._profile()
        positive_regime = float(np.nanmin(self.market_data.current_curve.zero_yields) > -0.001)
        steepness = float(self.market_data.current_curve.zero_yields[-1] - self.market_data.current_curve.zero_yields[0])
        inverted = float(steepness < 0.0)
        risk_off = float((self.market_data.regime_features or {}).get("risk_off_score", 0.0))

        hist_errors = np.array([m.diagnostics.historical_one_step_rmse for m in self.models], dtype=float)
        curve_errors = np.array([
            0.0 if not np.isfinite(m.diagnostics.current_curve_rmse) else m.diagnostics.current_curve_rmse for m in self.models
        ], dtype=float)
        hist_scale = max(np.nanmedian(hist_errors), 1e-6)
        curve_scale = max(np.nanmedian(curve_errors + 1e-12), 1e-6)

        rows: list[dict[str, float]] = []
        scores = []
        for model in self.models:
            traits = model.traits
            pred_short = model.predict_short_rate(horizon_years)

            fit_component = -(model.diagnostics.historical_one_step_rmse / hist_scale) * profile["hist_mult"]
            curve_component = -(model.diagnostics.current_curve_rmse / curve_scale) * profile["curve_mult"]
            structural_component = (
                (0.30 + profile["curve_fit_bonus"]) * traits.curve_fit
                + 0.18 * traits.mean_reversion
                + (0.12 + profile["flex_bonus"]) * traits.flexibility
                + 0.14 * traits.analytic
                + 0.14 * traits.simplicity
                + 0.12 * traits.positivity
            )
            positivity_component = 0.20 * positive_regime * traits.positivity
            inversion_component = 0.10 * inverted * traits.flexibility
            risk_component = 0.08 * risk_off * (0.5 * traits.curve_fit + 0.5 * traits.flexibility)
            forecast_component = profile["forecast_bonus"] * (traits.mean_reversion + 0.5 * traits.simplicity)
            negative_forecast_penalty = -0.60 if positive_regime and pred_short < -0.001 else 0.0

            score = (
                fit_component
                + curve_component
                + structural_component
                + positivity_component
                + inversion_component
                + risk_component
                + forecast_component
                + negative_forecast_penalty
            )
            scores.append(score)
            rows.append(
                {
                    "model": model.name,
                    "historical_rmse": model.diagnostics.historical_one_step_rmse,
                    "curve_rmse": model.diagnostics.current_curve_rmse,
                    "fit_component": fit_component,
                    "curve_component": curve_component,
                    "structural_component": structural_component,
                    "positivity_component": positivity_component,
                    "inversion_component": inversion_component,
                    "risk_component": risk_component,
                    "forecast_component": forecast_component,
                    "negative_forecast_penalty": negative_forecast_penalty,
                    "score": score,
                }
            )

        scores_arr = np.array(scores, dtype=float)
        scores_arr -= np.max(scores_arr)
        weights = np.exp(scores_arr)
        weights /= np.sum(weights)
        for row, weight in zip(rows, weights):
            row["weight"] = float(weight)
        df = pd.DataFrame(rows).sort_values("weight", ascending=False).reset_index(drop=True)
        return df

    def predict(self, horizon_years: float, maturities: Iterable[float]) -> EnsemblePrediction:
        if self.market_data is None or not self.models:
            raise ValueError("fit the meta-model before calling predict")
        maturities_arr = np.asarray(list(maturities), dtype=float)
        weights_df = self._compute_weights(horizon_years=horizon_years)
        weight_map = dict(zip(weights_df["model"], weights_df["weight"]))

        short_predictions = {}
        curve_predictions = {}
        aggregate_short = 0.0
        aggregate_curve = np.zeros_like(maturities_arr, dtype=float)

        for model in self.models:
            w = weight_map[model.name]
            short_pred = model.predict_short_rate(horizon_years)
            curve_pred = model.predict_yield_curve(horizon_years, maturities_arr)
            short_predictions[model.name] = short_pred
            curve_predictions[model.name] = curve_pred
            aggregate_short += w * short_pred
            aggregate_curve += w * curve_pred

        curve_df = pd.DataFrame(curve_predictions, index=maturities_arr).T
        curve_df.columns = [f"{m:.2f}y" for m in maturities_arr]
        short_series = pd.Series(short_predictions, name="short_rate_forecast")

        diagnostics = {
            "positive_regime": float(np.nanmin(self.market_data.current_curve.zero_yields) > -0.001),
            "curve_steepness": float(self.market_data.current_curve.zero_yields[-1] - self.market_data.current_curve.zero_yields[0]),
            "regime_features": self.market_data.regime_features or {},
            "weighting_method": self.weighting_method,
        }
        self.weight_breakdown_ = weights_df
        return EnsemblePrediction(
            horizon_years=horizon_years,
            maturities=maturities_arr,
            aggregate_short_rate=float(aggregate_short),
            aggregate_curve=aggregate_curve,
            weights=weights_df,
            model_curves=curve_df,
            model_short_rates=short_series,
            diagnostics=diagnostics,
        )

    def rolling_one_step_backtest(self, window: int = 60) -> pd.DataFrame:
        if self.market_data is None or not self.models:
            raise ValueError("fit the meta-model before running diagnostics")
        weights_df = self.weight_breakdown_ if self.weight_breakdown_ is not None else self._compute_weights(horizon_years=1.0)
        weight_map = dict(zip(weights_df["model"], weights_df["weight"]))
        short_rates = self.market_data.short_rates.astype(float)
        values = short_rates.to_numpy()
        dates = short_rates.index[1:]
        ensemble_pred = np.zeros(len(values) - 1, dtype=float)
        for model in self.models:
            ensemble_pred += weight_map[model.name] * model.one_step_expected_rate(values[:-1], model.dt)
        errors = values[1:] - ensemble_pred
        rolling = pd.Series(errors, index=dates).rolling(window=max(5, window)).apply(lambda x: float(np.sqrt(np.mean(np.square(x)))), raw=False)
        return pd.DataFrame({"date": rolling.index, "rolling_rmse": rolling.to_numpy()}).dropna().reset_index(drop=True)


__all__ = [
    "MODEL_DESCRIPTIONS",
    "WEIGHTING_LABELS",
    "RuntimeModelConfig",
    "MarketData",
    "EnsemblePrediction",
    "build_model",
    "model_catalog",
    "recommended_model_names",
    "InterestRateMetaModel",
]
