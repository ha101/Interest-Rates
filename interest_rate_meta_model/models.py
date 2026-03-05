from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .curves import ZeroCurve

EPS = 1e-10


@dataclass
class FitDiagnostics:
    historical_one_step_rmse: float
    current_curve_rmse: float
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelTraits:
    curve_fit: float
    positivity: float
    mean_reversion: float
    analytic: float
    flexibility: float
    simplicity: float


@dataclass
class InterestRateModel(ABC):
    name: str
    traits: ModelTraits
    r0: float | None = None
    dt: float | None = None
    current_curve: ZeroCurve | None = None
    diagnostics: FitDiagnostics | None = None

    def fit(self, short_rates: pd.Series | np.ndarray, current_curve: ZeroCurve | None = None, dt: float | None = None) -> "InterestRateModel":
        r = _to_numpy(short_rates)
        if np.any(~np.isfinite(r)):
            raise ValueError("short_rates contain non-finite values")
        self.r0 = float(r[-1])
        self.dt = _infer_dt(short_rates, dt)
        self.current_curve = current_curve
        self._fit_impl(r, self.dt, current_curve)
        hist_rmse = self.historical_one_step_rmse(r, self.dt)
        curve_rmse = self.current_curve_fit_rmse(current_curve) if current_curve is not None else np.nan
        self.diagnostics = FitDiagnostics(hist_rmse, curve_rmse)
        return self

    @abstractmethod
    def _fit_impl(self, short_rates: np.ndarray, dt: float, current_curve: ZeroCurve | None) -> None:
        raise NotImplementedError

    @abstractmethod
    def one_step_expected_rate(self, r_t: np.ndarray, dt: float) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def short_rate_mean(self, horizon_years: float) -> float:
        raise NotImplementedError

    @abstractmethod
    def yield_curve_from_state(self, maturities: np.ndarray, r_state: float, t: float = 0.0) -> np.ndarray:
        raise NotImplementedError

    def historical_one_step_rmse(self, short_rates: np.ndarray, dt: float) -> float:
        expected = self.one_step_expected_rate(short_rates[:-1], dt)
        rmse = float(np.sqrt(np.mean((short_rates[1:] - expected) ** 2)))
        return rmse

    def current_curve_fit_rmse(self, current_curve: ZeroCurve | None) -> float:
        if current_curve is None:
            return float("nan")
        model_yields = self.yield_curve_from_state(current_curve.maturities, r_state=float(self.r0), t=0.0)
        return float(np.sqrt(np.mean((model_yields - current_curve.zero_yields) ** 2)))

    def predict_short_rate(self, horizon_years: float) -> float:
        return float(self.short_rate_mean(horizon_years))

    def predict_yield_curve(self, horizon_years: float, maturities: np.ndarray) -> np.ndarray:
        return self.yield_curve_from_state(np.asarray(maturities, dtype=float), r_state=self.short_rate_mean(horizon_years), t=horizon_years)

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "traits": self.traits.__dict__,
            "r0": self.r0,
            "dt": self.dt,
            "diagnostics": None if self.diagnostics is None else {
                "historical_one_step_rmse": self.diagnostics.historical_one_step_rmse,
                "current_curve_rmse": self.diagnostics.current_curve_rmse,
                **self.diagnostics.extra,
            },
        }


# ---------------------------------------------------------------------------
# Calibration helpers
# ---------------------------------------------------------------------------


def _to_numpy(series: pd.Series | np.ndarray) -> np.ndarray:
    if isinstance(series, pd.Series):
        return series.astype(float).to_numpy()
    return np.asarray(series, dtype=float)


def _infer_dt(series: pd.Series | np.ndarray, dt: float | None) -> float:
    if dt is not None:
        return float(dt)
    if isinstance(series, pd.Series) and isinstance(series.index, pd.DatetimeIndex) and len(series) > 1:
        deltas = np.diff(series.index.view("int64")) / (24 * 60 * 60 * 1e9)
        return float(np.median(deltas) / 365.25)
    # daily-business default
    return 1.0 / 252.0


def _fit_ar1(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    X = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    c = float(beta[0])
    phi = float(beta[1])
    resid = y - (c + phi * x)
    s2 = float(np.mean(resid**2))
    return c, phi, s2


# ---------------------------------------------------------------------------
# Vasicek
# ---------------------------------------------------------------------------


@dataclass
class VasicekModel(InterestRateModel):
    kappa: float | None = None
    theta: float | None = None
    sigma: float | None = None

    def __init__(self) -> None:
        super().__init__(
            name="vasicek",
            traits=ModelTraits(
                curve_fit=0.35,
                positivity=0.0,
                mean_reversion=1.0,
                analytic=1.0,
                flexibility=0.55,
                simplicity=1.0,
            ),
        )

    def _fit_impl(self, short_rates: np.ndarray, dt: float, current_curve: ZeroCurve | None) -> None:
        x = short_rates[:-1]
        y = short_rates[1:]
        c, phi, s2 = _fit_ar1(x, y)
        phi = np.clip(phi, 1e-6, 0.999999)
        self.kappa = float(-np.log(phi) / dt)
        self.theta = float(c / (1.0 - phi))
        self.sigma = float(np.sqrt(max(s2, EPS) * 2.0 * self.kappa / (1.0 - phi**2)))

    def one_step_expected_rate(self, r_t: np.ndarray, dt: float) -> np.ndarray:
        return r_t * np.exp(-self.kappa * dt) + self.theta * (1.0 - np.exp(-self.kappa * dt))

    def short_rate_mean(self, horizon_years: float) -> float:
        return float(self.one_step_expected_rate(np.array([self.r0]), horizon_years)[0])

    def _A_B(self, tau: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        tau = np.asarray(tau, dtype=float)
        B = (1.0 - np.exp(-self.kappa * tau)) / self.kappa
        logA = (self.theta - (self.sigma**2) / (2.0 * self.kappa**2)) * (B - tau) - (self.sigma**2) * B**2 / (4.0 * self.kappa)
        return logA, B

    def yield_curve_from_state(self, maturities: np.ndarray, r_state: float, t: float = 0.0) -> np.ndarray:
        tau = np.asarray(maturities, dtype=float)
        logA, B = self._A_B(tau)
        prices = np.exp(logA - B * r_state)
        return -np.log(prices) / np.maximum(tau, EPS)


# ---------------------------------------------------------------------------
# Dothan
# ---------------------------------------------------------------------------


@dataclass
class DothanModel(InterestRateModel):
    k: float | None = None
    sigma: float | None = None
    mc_paths: int = 3000
    mc_steps_per_year: int = 120
    seed: int = 7

    def __init__(self, mc_paths: int = 3000, mc_steps_per_year: int = 120, seed: int = 7) -> None:
        super().__init__(
            name="dothan",
            traits=ModelTraits(
                curve_fit=0.25,
                positivity=1.0,
                mean_reversion=0.2,
                analytic=0.2,
                flexibility=0.30,
                simplicity=0.65,
            ),
        )
        self.mc_paths = mc_paths
        self.mc_steps_per_year = mc_steps_per_year
        self.seed = seed

    def _fit_impl(self, short_rates: np.ndarray, dt: float, current_curve: ZeroCurve | None) -> None:
        r = np.maximum(short_rates, EPS)
        dx = np.diff(np.log(r))
        v = np.var(dx, ddof=0) / dt
        self.sigma = float(np.sqrt(max(v, EPS)))
        m = float(np.mean(dx) / dt)
        self.k = float(m + 0.5 * self.sigma**2)

    def one_step_expected_rate(self, r_t: np.ndarray, dt: float) -> np.ndarray:
        return np.asarray(r_t, dtype=float) * np.exp(self.k * dt)

    def short_rate_mean(self, horizon_years: float) -> float:
        return float(self.r0 * np.exp(self.k * horizon_years))

    def _discount_price_mc(self, maturity: float, r_state: float) -> float:
        if maturity <= 0:
            return 1.0
        rng = np.random.default_rng(self.seed + int(1000 * maturity))
        n_steps = max(8, int(np.ceil(self.mc_steps_per_year * maturity)))
        dt = maturity / n_steps
        rates = np.full(self.mc_paths, max(r_state, EPS), dtype=float)
        disc_log = np.zeros(self.mc_paths, dtype=float)
        for _ in range(n_steps):
            z = rng.normal(size=self.mc_paths)
            prev = rates.copy()
            rates = prev * np.exp((self.k - 0.5 * self.sigma**2) * dt + self.sigma * np.sqrt(dt) * z)
            disc_log -= 0.5 * (prev + rates) * dt
        return float(np.mean(np.exp(disc_log)))

    def yield_curve_from_state(self, maturities: np.ndarray, r_state: float, t: float = 0.0) -> np.ndarray:
        tau = np.asarray(maturities, dtype=float)
        prices = np.array([self._discount_price_mc(m, r_state) for m in tau])
        return -np.log(np.maximum(prices, EPS)) / np.maximum(tau, EPS)


# ---------------------------------------------------------------------------
# Exponentiated Vasicek
# ---------------------------------------------------------------------------


@dataclass
class ExponentiatedVasicekModel(InterestRateModel):
    kappa: float | None = None
    theta: float | None = None
    sigma: float | None = None
    mc_paths: int = 3000
    mc_steps_per_year: int = 120
    seed: int = 11

    def __init__(self, mc_paths: int = 3000, mc_steps_per_year: int = 120, seed: int = 11) -> None:
        super().__init__(
            name="exp_vasicek",
            traits=ModelTraits(
                curve_fit=0.30,
                positivity=1.0,
                mean_reversion=0.95,
                analytic=0.2,
                flexibility=0.45,
                simplicity=0.70,
            ),
        )
        self.mc_paths = mc_paths
        self.mc_steps_per_year = mc_steps_per_year
        self.seed = seed

    def _fit_impl(self, short_rates: np.ndarray, dt: float, current_curve: ZeroCurve | None) -> None:
        y = np.log(np.maximum(short_rates, EPS))
        c, phi, s2 = _fit_ar1(y[:-1], y[1:])
        phi = np.clip(phi, 1e-6, 0.999999)
        self.kappa = float(-np.log(phi) / dt)
        sigma_y = float(np.sqrt(max(s2, EPS) * 2.0 * self.kappa / (1.0 - phi**2)))
        self.sigma = sigma_y
        long_run_mean_y = c / (1.0 - phi)
        self.theta = float(self.kappa * long_run_mean_y + 0.5 * self.sigma**2)

    def one_step_expected_rate(self, r_t: np.ndarray, dt: float) -> np.ndarray:
        r_t = np.maximum(np.asarray(r_t, dtype=float), EPS)
        y0 = np.log(r_t)
        mu_inf = (self.theta - 0.5 * self.sigma**2) / self.kappa
        mean_y = y0 * np.exp(-self.kappa * dt) + mu_inf * (1.0 - np.exp(-self.kappa * dt))
        var_y = (self.sigma**2) / (2.0 * self.kappa) * (1.0 - np.exp(-2.0 * self.kappa * dt))
        return np.exp(mean_y + 0.5 * var_y)

    def short_rate_mean(self, horizon_years: float) -> float:
        return float(self.one_step_expected_rate(np.array([self.r0]), horizon_years)[0])

    def _discount_price_mc(self, maturity: float, r_state: float) -> float:
        if maturity <= 0:
            return 1.0
        rng = np.random.default_rng(self.seed + int(1000 * maturity))
        n_steps = max(8, int(np.ceil(self.mc_steps_per_year * maturity)))
        dt = maturity / n_steps
        y = np.full(self.mc_paths, np.log(max(r_state, EPS)), dtype=float)
        disc_log = np.zeros(self.mc_paths, dtype=float)
        mu_inf = (self.theta - 0.5 * self.sigma**2) / self.kappa
        for _ in range(n_steps):
            z = rng.normal(size=self.mc_paths)
            mean_y = y * np.exp(-self.kappa * dt) + mu_inf * (1.0 - np.exp(-self.kappa * dt))
            var_y = (self.sigma**2) / (2.0 * self.kappa) * (1.0 - np.exp(-2.0 * self.kappa * dt))
            prev_r = np.exp(y)
            y = mean_y + np.sqrt(var_y) * z
            rates = np.exp(y)
            disc_log -= 0.5 * (prev_r + rates) * dt
        return float(np.mean(np.exp(disc_log)))

    def yield_curve_from_state(self, maturities: np.ndarray, r_state: float, t: float = 0.0) -> np.ndarray:
        tau = np.asarray(maturities, dtype=float)
        prices = np.array([self._discount_price_mc(m, r_state) for m in tau])
        return -np.log(np.maximum(prices, EPS)) / np.maximum(tau, EPS)


# ---------------------------------------------------------------------------
# CIR
# ---------------------------------------------------------------------------


@dataclass
class CIRModel(InterestRateModel):
    kappa: float | None = None
    theta: float | None = None
    sigma: float | None = None

    def __init__(self) -> None:
        super().__init__(
            name="cir",
            traits=ModelTraits(
                curve_fit=0.50,
                positivity=1.0,
                mean_reversion=1.0,
                analytic=1.0,
                flexibility=0.70,
                simplicity=0.80,
            ),
        )

    def _fit_impl(self, short_rates: np.ndarray, dt: float, current_curve: ZeroCurve | None) -> None:
        r = np.maximum(short_rates, 1e-6)
        x1 = dt / np.sqrt(r[:-1])
        x2 = -dt * np.sqrt(r[:-1])
        y = (r[1:] - r[:-1]) / np.sqrt(r[:-1])
        X = np.column_stack([x1, x2])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        a = float(beta[0])
        kappa = max(float(beta[1]), 1e-6)
        theta = max(a / kappa, 1e-6)
        resid = y - X @ beta
        sigma = max(float(np.std(resid, ddof=0) / np.sqrt(dt)), 1e-6)
        if 2.0 * kappa * theta < sigma**2:
            theta = (sigma**2) / (2.0 * kappa) + 1e-6
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma

    def one_step_expected_rate(self, r_t: np.ndarray, dt: float) -> np.ndarray:
        return np.asarray(r_t, dtype=float) * np.exp(-self.kappa * dt) + self.theta * (1.0 - np.exp(-self.kappa * dt))

    def short_rate_mean(self, horizon_years: float) -> float:
        return float(self.one_step_expected_rate(np.array([self.r0]), horizon_years)[0])

    def _A_B(self, tau: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        tau = np.asarray(tau, dtype=float)
        gamma = np.sqrt(self.kappa**2 + 2.0 * self.sigma**2)
        exp_gt = np.exp(gamma * tau)
        B = 2.0 * (exp_gt - 1.0) / ((gamma + self.kappa) * (exp_gt - 1.0) + 2.0 * gamma)
        A = (
            2.0
            * self.kappa
            * self.theta
            / (self.sigma**2)
            * np.log(
                2.0 * gamma * np.exp((self.kappa + gamma) * tau / 2.0)
                / ((gamma + self.kappa) * (exp_gt - 1.0) + 2.0 * gamma)
            )
        )
        return A, B

    def yield_curve_from_state(self, maturities: np.ndarray, r_state: float, t: float = 0.0) -> np.ndarray:
        tau = np.asarray(maturities, dtype=float)
        logA, B = self._A_B(tau)
        prices = np.exp(logA - B * max(r_state, EPS))
        return -np.log(prices) / np.maximum(tau, EPS)


# ---------------------------------------------------------------------------
# Ho-Lee
# ---------------------------------------------------------------------------


@dataclass
class HoLeeModel(InterestRateModel):
    sigma: float | None = None

    def __init__(self) -> None:
        super().__init__(
            name="ho_lee",
            traits=ModelTraits(
                curve_fit=1.0,
                positivity=0.0,
                mean_reversion=0.0,
                analytic=1.0,
                flexibility=0.80,
                simplicity=0.95,
            ),
        )

    def _fit_impl(self, short_rates: np.ndarray, dt: float, current_curve: ZeroCurve | None) -> None:
        if current_curve is None:
            raise ValueError("Ho-Lee requires a current zero curve")
        dr = np.diff(short_rates)
        self.sigma = float(np.std(dr, ddof=0) / np.sqrt(dt))

    def theta_function(self, t: np.ndarray) -> np.ndarray:
        return self.current_curve.instantaneous_forward_derivative(t) + (self.sigma**2) * np.asarray(t, dtype=float)

    def one_step_expected_rate(self, r_t: np.ndarray, dt: float) -> np.ndarray:
        return np.asarray(r_t, dtype=float) + self.theta_function(np.array([0.0]))[0] * dt

    def short_rate_mean(self, horizon_years: float) -> float:
        if horizon_years <= 0:
            return float(self.r0)
        grid = np.linspace(0.0, horizon_years, 200)
        theta_vals = self.theta_function(grid)
        integral = np.trapz(theta_vals, grid)
        return float(self.r0 + integral)

    def yield_curve_from_state(self, maturities: np.ndarray, r_state: float, t: float = 0.0) -> np.ndarray:
        tau = np.asarray(maturities, dtype=float)
        T = t + tau
        P0T = self.current_curve.discount_factor(T)
        P0t = np.exp(-self.current_curve.zero_yield(np.array([max(t, EPS)]))[0] * max(t, EPS)) if t > 0 else 1.0
        f0t = self.current_curve.instantaneous_forward(max(t, EPS)) if t > 0 else self.current_curve.instantaneous_forward(self.current_curve.maturities[0] * 0.5)
        prices = P0T / P0t * np.exp(tau * f0t - 0.5 * (self.sigma**2) * t * tau**2 - tau * r_state)
        return -np.log(np.maximum(prices, EPS)) / np.maximum(tau, EPS)


# ---------------------------------------------------------------------------
# Hull-White extended Vasicek
# ---------------------------------------------------------------------------


@dataclass
class HullWhiteModel(InterestRateModel):
    kappa: float | None = None
    sigma: float | None = None

    def __init__(self) -> None:
        super().__init__(
            name="hull_white",
            traits=ModelTraits(
                curve_fit=1.0,
                positivity=0.0,
                mean_reversion=1.0,
                analytic=1.0,
                flexibility=0.95,
                simplicity=0.85,
            ),
        )

    def _fit_impl(self, short_rates: np.ndarray, dt: float, current_curve: ZeroCurve | None) -> None:
        if current_curve is None:
            raise ValueError("Hull-White requires a current zero curve")
        x = short_rates[:-1]
        y = short_rates[1:]
        c, phi, s2 = _fit_ar1(x, y)
        phi = np.clip(phi, 1e-6, 0.999999)
        self.kappa = float(-np.log(phi) / dt)
        self.sigma = float(np.sqrt(max(s2, EPS) * 2.0 * self.kappa / (1.0 - phi**2)))

    def B(self, tau: np.ndarray) -> np.ndarray:
        tau = np.asarray(tau, dtype=float)
        return (1.0 - np.exp(-self.kappa * tau)) / self.kappa

    def theta_function(self, t: np.ndarray) -> np.ndarray:
        t = np.asarray(t, dtype=float)
        f0 = self.current_curve.instantaneous_forward(np.maximum(t, self.current_curve.maturities[0]))
        df0 = self.current_curve.instantaneous_forward_derivative(np.maximum(t, self.current_curve.maturities[0]))
        return (df0 + self.kappa * f0 + (self.sigma**2) / (2.0 * self.kappa) * (1.0 - np.exp(-2.0 * self.kappa * t))) / self.kappa

    def one_step_expected_rate(self, r_t: np.ndarray, dt: float) -> np.ndarray:
        expk = np.exp(-self.kappa * dt)
        theta0 = self.theta_function(np.array([0.0]))[0]
        return np.asarray(r_t, dtype=float) * expk + theta0 * (1.0 - expk)

    def short_rate_mean(self, horizon_years: float) -> float:
        if horizon_years <= 0:
            return float(self.r0)
        grid = np.linspace(0.0, horizon_years, 300)
        theta_vals = self.theta_function(grid)
        kernel = np.exp(-self.kappa * (horizon_years - grid))
        integral = np.trapz(self.kappa * theta_vals * kernel, grid)
        return float(self.r0 * np.exp(-self.kappa * horizon_years) + integral)

    def yield_curve_from_state(self, maturities: np.ndarray, r_state: float, t: float = 0.0) -> np.ndarray:
        tau = np.asarray(maturities, dtype=float)
        T = t + tau
        P0T = self.current_curve.discount_factor(T)
        P0t = np.exp(-self.current_curve.zero_yield(np.array([max(t, EPS)]))[0] * max(t, EPS)) if t > 0 else 1.0
        f0t = self.current_curve.instantaneous_forward(np.maximum(t, self.current_curve.maturities[0])) if t > 0 else self.current_curve.instantaneous_forward(self.current_curve.maturities[0] * 0.5)
        B = self.B(tau)
        prices = P0T / P0t * np.exp(B * f0t - (self.sigma**2) / (4.0 * self.kappa) * B**2 * (1.0 - np.exp(-2.0 * self.kappa * t)) - B * r_state)
        return -np.log(np.maximum(prices, EPS)) / np.maximum(tau, EPS)


# ---------------------------------------------------------------------------
# Deterministic shift / CIR++
# ---------------------------------------------------------------------------


@dataclass
class CIRPlusPlusModel(InterestRateModel):
    base_model: CIRModel = field(default_factory=CIRModel)

    def __init__(self) -> None:
        super().__init__(
            name="cir_pp",
            traits=ModelTraits(
                curve_fit=1.0,
                positivity=0.8,
                mean_reversion=1.0,
                analytic=1.0,
                flexibility=1.0,
                simplicity=0.70,
            ),
        )
        self.base_model = CIRModel()

    def _fit_impl(self, short_rates: np.ndarray, dt: float, current_curve: ZeroCurve | None) -> None:
        if current_curve is None:
            raise ValueError("CIR++ requires a current zero curve")
        self.base_model.fit(short_rates, current_curve=None, dt=dt)
        self.r0 = self.base_model.r0
        self.dt = dt
        self.current_curve = current_curve

    @property
    def kappa(self) -> float:
        return self.base_model.kappa

    @property
    def theta(self) -> float:
        return self.base_model.theta

    @property
    def sigma(self) -> float:
        return self.base_model.sigma

    def phi(self, t: np.ndarray) -> np.ndarray:
        t = np.asarray(t, dtype=float)
        f0 = self.current_curve.instantaneous_forward(np.maximum(t, self.current_curve.maturities[0]))
        base_forward = self._base_forward(np.maximum(t, self.current_curve.maturities[0]))
        return f0 - base_forward

    def _base_discount(self, maturity: np.ndarray, x0: float) -> np.ndarray:
        y = self.base_model.yield_curve_from_state(np.asarray(maturity, dtype=float), x0)
        return np.exp(-np.asarray(maturity, dtype=float) * y)

    def _base_forward(self, maturity: np.ndarray) -> np.ndarray:
        m = np.asarray(maturity, dtype=float)
        eps = 1e-4
        upper = np.maximum(m + eps, eps)
        lower = np.maximum(m - eps, eps * 0.5)
        logP_u = np.log(self._base_discount(upper, max(self.r0, EPS)))
        logP_l = np.log(self._base_discount(lower, max(self.r0, EPS)))
        return -(logP_u - logP_l) / (upper - lower)

    def one_step_expected_rate(self, r_t: np.ndarray, dt: float) -> np.ndarray:
        base_r = np.maximum(np.asarray(r_t, dtype=float) - self.phi(np.array([0.0]))[0], EPS)
        return self.base_model.one_step_expected_rate(base_r, dt) + self.phi(np.array([dt]))[0]

    def short_rate_mean(self, horizon_years: float) -> float:
        base_r0 = max(self.r0 - self.phi(np.array([0.0]))[0], EPS)
        base_mean = self.base_model.one_step_expected_rate(np.array([base_r0]), horizon_years)[0]
        return float(base_mean + self.phi(np.array([horizon_years]))[0])

    def yield_curve_from_state(self, maturities: np.ndarray, r_state: float, t: float = 0.0) -> np.ndarray:
        tau = np.asarray(maturities, dtype=float)
        T = t + tau
        P0T = self.current_curve.discount_factor(T)
        P0t = np.exp(-self.current_curve.zero_yield(np.array([max(t, EPS)]))[0] * max(t, EPS)) if t > 0 else 1.0
        base_x0 = max(self.r0 - self.phi(np.array([0.0]))[0], EPS)
        F0T = self._base_discount(T, base_x0)
        F0t = self._base_discount(np.array([max(t, EPS)]), base_x0)[0] if t > 0 else 1.0
        x_t = max(r_state - self.phi(np.array([t]))[0], EPS)
        FtT = self._base_discount(tau, x_t)
        prices = P0T * F0t / (P0t * F0T) * FtT
        return -np.log(np.maximum(prices, EPS)) / np.maximum(tau, EPS)


# ---------------------------------------------------------------------------
# One-factor Gaussian HJM with exponentially decaying volatility
# ---------------------------------------------------------------------------


@dataclass
class GaussianHJMModel(InterestRateModel):
    sigma0: float | None = None
    decay: float = 0.15

    def __init__(self, decay: float = 0.15) -> None:
        super().__init__(
            name="hjm",
            traits=ModelTraits(
                curve_fit=1.0,
                positivity=0.0,
                mean_reversion=0.55,
                analytic=0.60,
                flexibility=1.0,
                simplicity=0.55,
            ),
        )
        self.decay = decay

    def _fit_impl(self, short_rates: np.ndarray, dt: float, current_curve: ZeroCurve | None) -> None:
        if current_curve is None:
            raise ValueError("HJM requires a current zero curve")
        dr = np.diff(short_rates)
        self.sigma0 = float(np.std(dr, ddof=0) / np.sqrt(dt))

    def sigma(self, tau: np.ndarray) -> np.ndarray:
        tau = np.asarray(tau, dtype=float)
        if abs(self.decay) < 1e-10:
            return np.full_like(tau, self.sigma0, dtype=float)
        return self.sigma0 * np.exp(-self.decay * tau)

    def alpha(self, tau: np.ndarray) -> np.ndarray:
        tau = np.asarray(tau, dtype=float)
        if abs(self.decay) < 1e-10:
            return (self.sigma0**2) * tau
        sig = self.sigma(tau)
        integral = self.sigma0 * (1.0 - np.exp(-self.decay * tau)) / self.decay
        return sig * integral

    def one_step_expected_rate(self, r_t: np.ndarray, dt: float) -> np.ndarray:
        return np.asarray(r_t, dtype=float)

    def _expected_forward_at_horizon(self, horizon_years: float, maturities: np.ndarray) -> np.ndarray:
        maturities = np.asarray(maturities, dtype=float)
        T = horizon_years + maturities
        base = self.current_curve.instantaneous_forward(np.maximum(T, self.current_curve.maturities[0]))
        if horizon_years <= 0:
            return base
        grid = np.linspace(0.0, horizon_years, 300)
        out = np.empty_like(maturities, dtype=float)
        for i, tau in enumerate(maturities):
            u = T[i] - grid
            drift = self.alpha(np.maximum(u, 0.0))
            out[i] = base[i] + np.trapz(drift, grid)
        return out

    def short_rate_mean(self, horizon_years: float) -> float:
        return float(self._expected_forward_at_horizon(horizon_years, np.array([0.0]))[0])

    def yield_curve_from_state(self, maturities: np.ndarray, r_state: float, t: float = 0.0) -> np.ndarray:
        maturities = np.asarray(maturities, dtype=float)
        fwd = self._expected_forward_at_horizon(t, maturities)
        # Approximate future zero yields by averaging the forward curve over [0, maturity].
        yields = np.empty_like(maturities)
        for i, m in enumerate(maturities):
            if m <= 0:
                yields[i] = r_state
                continue
            grid = np.linspace(0.0, m, 200)
            f_grid = self._expected_forward_at_horizon(t, grid)
            yields[i] = np.trapz(f_grid, grid) / m
        return yields


DEFAULT_MODEL_BUILDERS = [
    VasicekModel,
    DothanModel,
    ExponentiatedVasicekModel,
    CIRModel,
    HoLeeModel,
    HullWhiteModel,
    CIRPlusPlusModel,
    GaussianHJMModel,
]
