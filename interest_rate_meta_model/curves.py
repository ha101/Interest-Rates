from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq

EPS = 1e-12

TREASURY_TAG_TO_MATURITY_YEARS: dict[str, float] = {
    "BC_1MONTH": 1.0 / 12.0,
    "BC_1_5MONTH": 1.5 / 12.0,
    "BC_2MONTH": 2.0 / 12.0,
    "BC_3MONTH": 3.0 / 12.0,
    "BC_4MONTH": 4.0 / 12.0,
    "BC_6MONTH": 6.0 / 12.0,
    "BC_1YEAR": 1.0,
    "BC_2YEAR": 2.0,
    "BC_3YEAR": 3.0,
    "BC_5YEAR": 5.0,
    "BC_7YEAR": 7.0,
    "BC_10YEAR": 10.0,
    "BC_20YEAR": 20.0,
    "BC_30YEAR": 30.0,
    "BC_30YEARDISPLAY": 30.0,
}

TENOR_TO_TAG: dict[str, str] = {
    "1M": "BC_1MONTH",
    "1.5M": "BC_1_5MONTH",
    "2M": "BC_2MONTH",
    "3M": "BC_3MONTH",
    "4M": "BC_4MONTH",
    "6M": "BC_6MONTH",
    "1Y": "BC_1YEAR",
    "2Y": "BC_2YEAR",
    "3Y": "BC_3YEAR",
    "5Y": "BC_5YEAR",
    "7Y": "BC_7YEAR",
    "10Y": "BC_10YEAR",
    "20Y": "BC_20YEAR",
    "30Y": "BC_30YEAR",
}

DEFAULT_DASHBOARD_MATURITIES = np.array(
    [1.0 / 12.0, 1.5 / 12.0, 2.0 / 12.0, 3.0 / 12.0, 4.0 / 12.0, 6.0 / 12.0, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0],
    dtype=float,
)


def _as_1d_float_array(values: Iterable[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(list(values) if not isinstance(values, np.ndarray) else values, dtype=float).reshape(-1)
    if arr.size == 0:
        raise ValueError("expected at least one value")
    if np.any(~np.isfinite(arr)):
        raise ValueError("array contains non-finite values")
    return arr


def _sort_and_validate(maturities: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(maturities)
    maturities = maturities[order]
    values = values[order]
    mask = np.isfinite(maturities) & np.isfinite(values) & (maturities > 0.0)
    maturities = maturities[mask]
    values = values[mask]
    if maturities.size == 0:
        raise ValueError("no valid maturities/yields")
    unique_mats: list[float] = []
    unique_vals: list[float] = []
    for m, v in zip(maturities, values):
        if unique_mats and abs(m - unique_mats[-1]) < 1e-10:
            unique_vals[-1] = float(v)
        else:
            unique_mats.append(float(m))
            unique_vals.append(float(v))
    return np.asarray(unique_mats, dtype=float), np.asarray(unique_vals, dtype=float)


def _interp_zero_from_nodes(maturities: np.ndarray, zero_yields: np.ndarray, t: float | np.ndarray) -> np.ndarray:
    t_arr = np.asarray(t, dtype=float)
    if maturities.size == 1:
        return np.full_like(t_arr, zero_yields[0], dtype=float)
    return np.interp(t_arr, maturities, zero_yields, left=zero_yields[0], right=zero_yields[-1])


def bootstrap_zero_curve_from_par_yields(
    maturities: Iterable[float] | np.ndarray,
    par_yields: Iterable[float] | np.ndarray,
    coupon_frequency: int = 2,
) -> np.ndarray:
    mats = _as_1d_float_array(maturities)
    par = _as_1d_float_array(par_yields)
    mats, par = _sort_and_validate(mats, par)
    coupon_dt = 1.0 / float(coupon_frequency)

    solved_mats: list[float] = []
    solved_zeros: list[float] = []

    for T, c in zip(mats, par):
        if T < 1.0 - 1e-10:
            zT = float(c)
            solved_mats.append(float(T))
            solved_zeros.append(zT)
            continue

        coupon_times = np.arange(coupon_dt, T + 1e-9, coupon_dt)
        if coupon_times.size == 0:
            zT = float(c)
            solved_mats.append(float(T))
            solved_zeros.append(zT)
            continue

        prev_mats = np.asarray(solved_mats, dtype=float)
        prev_zeros = np.asarray(solved_zeros, dtype=float)

        def price_minus_par(z_guess: float) -> float:
            if prev_mats.size:
                tmp_mats = np.concatenate([prev_mats, np.array([T], dtype=float)])
                tmp_zeros = np.concatenate([prev_zeros, np.array([z_guess], dtype=float)])
            else:
                tmp_mats = np.array([T], dtype=float)
                tmp_zeros = np.array([z_guess], dtype=float)
            z_coupon = _interp_zero_from_nodes(tmp_mats, tmp_zeros, coupon_times)
            dfs = np.exp(-z_coupon * coupon_times)
            coupon = c / coupon_frequency
            price = coupon * np.sum(dfs[:-1]) + (1.0 + coupon) * dfs[-1]
            return float(price - 1.0)

        lower, upper = -0.05, 0.25
        f_low = price_minus_par(lower)
        f_high = price_minus_par(upper)
        expand_steps = 0
        while f_low * f_high > 0.0 and expand_steps < 10:
            lower -= 0.05
            upper += 0.10
            f_low = price_minus_par(lower)
            f_high = price_minus_par(upper)
            expand_steps += 1

        if f_low * f_high > 0.0:
            zT = float(c)
        else:
            zT = float(brentq(price_minus_par, lower, upper, maxiter=200))

        solved_mats.append(float(T))
        solved_zeros.append(zT)

    return np.asarray(solved_zeros, dtype=float)


@dataclass
class ZeroCurve:
    maturities: np.ndarray
    zero_yields: np.ndarray
    name: str = "zero_curve"
    as_of: pd.Timestamp | None = None
    source: str | None = None
    metadata: dict[str, object] | None = None

    def __post_init__(self) -> None:
        mats = _as_1d_float_array(self.maturities)
        yld = _as_1d_float_array(self.zero_yields)
        mats, yld = _sort_and_validate(mats, yld)
        self.maturities = mats
        self.zero_yields = yld
        self._log_discount_nodes = -self.maturities * self.zero_yields
        self.metadata = dict(self.metadata or {})
        self.as_of = pd.Timestamp(self.as_of) if self.as_of is not None else None

        if self.maturities.size == 1:
            self._spline = None
        else:
            self._spline = CubicSpline(self.maturities, self._log_discount_nodes, bc_type="natural", extrapolate=True)

    @classmethod
    def from_par_yields(
        cls,
        maturities: Iterable[float] | np.ndarray,
        par_yields: Iterable[float] | np.ndarray,
        name: str = "bootstrapped_zero_curve",
        coupon_frequency: int = 2,
        *,
        as_of: pd.Timestamp | None = None,
        source: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> "ZeroCurve":
        zero_yields = bootstrap_zero_curve_from_par_yields(maturities, par_yields, coupon_frequency=coupon_frequency)
        return cls(
            np.asarray(list(maturities) if not isinstance(maturities, np.ndarray) else maturities, dtype=float),
            zero_yields,
            name=name,
            as_of=as_of,
            source=source,
            metadata=metadata,
        )

    @classmethod
    def from_discount_factors(
        cls,
        maturities: Sequence[float],
        discount_factors: Sequence[float],
        *,
        name: str = "zero_curve",
        as_of: pd.Timestamp | None = None,
        source: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> "ZeroCurve":
        mats = np.asarray(maturities, dtype=float)
        dfs = np.clip(np.asarray(discount_factors, dtype=float), EPS, None)
        zero_yields = -np.log(dfs) / np.maximum(mats, EPS)
        return cls(mats, zero_yields, name=name, as_of=as_of, source=source, metadata=metadata)

    @classmethod
    def from_treasury_par_yields(
        cls,
        row: Mapping[str, object] | pd.Series,
        *,
        as_of: pd.Timestamp | None = None,
        name: str = "us_treasury_bootstrapped_zero_curve",
        source: str = "U.S. Treasury Daily Treasury Par Yield Curve Rates",
    ) -> "ZeroCurve":
        if isinstance(row, pd.Series):
            record = row.to_dict()
        else:
            record = dict(row)
        return bootstrap_zero_curve_from_treasury_par_row(record, as_of=as_of, name=name, source=source)

    def _log_discount(self, t: float | np.ndarray) -> np.ndarray:
        t_arr = np.asarray(t, dtype=float)
        out = np.empty_like(t_arr, dtype=float)
        if self._spline is None:
            out[:] = -self.zero_yields[0] * t_arr
            return out

        left_mask = t_arr <= self.maturities[0]
        out[left_mask] = -self.zero_yields[0] * t_arr[left_mask]
        if np.any(~left_mask):
            out[~left_mask] = self._spline(t_arr[~left_mask])
        return out

    def zero_yield(self, t: float | np.ndarray) -> np.ndarray:
        t_arr = np.asarray(t, dtype=float)
        out = np.empty_like(t_arr, dtype=float)
        small = np.abs(t_arr) < EPS
        out[small] = self.zero_yields[0]
        out[~small] = -self._log_discount(t_arr[~small]) / np.maximum(t_arr[~small], EPS)
        return out

    def discount_factor(self, t: float | np.ndarray) -> np.ndarray:
        return np.exp(self._log_discount(t))

    def instantaneous_forward(self, t: float | np.ndarray) -> np.ndarray:
        t_arr = np.asarray(t, dtype=float)
        if self._spline is None:
            return np.full_like(t_arr, self.zero_yields[0], dtype=float)

        out = np.empty_like(t_arr, dtype=float)
        left_mask = t_arr <= self.maturities[0]
        out[left_mask] = self.zero_yields[0]
        if np.any(~left_mask):
            out[~left_mask] = -self._spline.derivative(1)(t_arr[~left_mask])
        return out

    def instantaneous_forward_derivative(self, t: float | np.ndarray) -> np.ndarray:
        t_arr = np.asarray(t, dtype=float)
        if self._spline is None:
            return np.zeros_like(t_arr, dtype=float)
        out = np.zeros_like(t_arr, dtype=float)
        mask = t_arr > self.maturities[0]
        if np.any(mask):
            out[mask] = -self._spline.derivative(2)(t_arr[mask])
        return out

    def par_yield(self, maturities: float | np.ndarray, coupon_frequency: int = 2) -> np.ndarray:
        mats = np.asarray(maturities, dtype=float)
        out = np.empty_like(mats, dtype=float)
        coupon_dt = 1.0 / float(coupon_frequency)
        flat = mats.reshape(-1)
        for i, T in enumerate(flat):
            if T <= coupon_dt + 1e-12:
                out.reshape(-1)[i] = float(self.zero_yield(np.array([max(T, self.maturities[0])]))[0])
                continue
            coupon_times = np.arange(coupon_dt, T + 1e-10, coupon_dt)
            dfs = self.discount_factor(coupon_times)
            annuity = np.sum(dfs)
            out.reshape(-1)[i] = float(coupon_frequency * (1.0 - dfs[-1]) / max(annuity, EPS))
        return out

    def curve_bundle(self, maturities: Sequence[float] | np.ndarray) -> dict[str, list[float]]:
        mats = np.asarray(maturities, dtype=float)
        zero = self.zero_yield(mats)
        par = self.par_yield(mats)
        fwd = self.instantaneous_forward(mats)
        return {
            "maturities": [float(x) for x in mats],
            "zero": [float(x) for x in zero],
            "par": [float(x) for x in par],
            "forward": [float(x) for x in fwd],
        }

    def to_frame(self) -> pd.DataFrame:
        df = pd.DataFrame({"maturity": self.maturities, "yield": self.zero_yields})
        if self.as_of is not None:
            df["as_of"] = pd.Timestamp(self.as_of)
        if self.source is not None:
            df["source"] = self.source
        return df


def _parse_as_of(as_of: object | None, record: Mapping[str, object]) -> pd.Timestamp | None:
    if as_of is not None:
        return pd.Timestamp(as_of)
    for key in ("NEW_DATE", "Date", "date"):
        if key in record and record[key] is not None:
            return pd.Timestamp(record[key])
    return None


def bootstrap_zero_curve_from_treasury_par_row(
    record: Mapping[str, object],
    *,
    as_of: pd.Timestamp | None = None,
    name: str = "us_treasury_bootstrapped_zero_curve",
    source: str = "U.S. Treasury Daily Treasury Par Yield Curve Rates",
) -> ZeroCurve:
    par_points: list[tuple[float, float]] = []
    for tag, maturity in TREASURY_TAG_TO_MATURITY_YEARS.items():
        if tag == "BC_30YEARDISPLAY":
            continue
        value = record.get(tag)
        if value in (None, "", "N/A"):
            continue
        try:
            par = float(value) / 100.0
        except (TypeError, ValueError):
            continue
        if np.isfinite(par):
            par_points.append((maturity, par))

    if len(par_points) < 4:
        raise ValueError("Need at least four Treasury par-yield points to build a curve")

    par_points.sort(key=lambda x: x[0])
    mats = np.asarray([m for m, _ in par_points], dtype=float)
    par_yields = np.asarray([y for _, y in par_points], dtype=float)
    curve = ZeroCurve.from_par_yields(
        mats,
        par_yields,
        name=name,
        as_of=_parse_as_of(as_of, record),
        source=source,
        metadata={
            "input_curve_type": "treasury_par_curve",
            "bootstrap_method": "coupon_bond_bootstrap_with_linear_interpolation",
            "input_maturities": mats.tolist(),
            "input_par_yields": par_yields.tolist(),
        },
    )
    return curve


def zero_curve_from_points(
    maturities: Sequence[float] | np.ndarray,
    zero_yields: Sequence[float] | np.ndarray,
    *,
    name: str = "curve",
    as_of: pd.Timestamp | None = None,
    source: str | None = None,
    metadata: dict[str, object] | None = None,
) -> ZeroCurve:
    return ZeroCurve(np.asarray(maturities, dtype=float), np.asarray(zero_yields, dtype=float), name=name, as_of=as_of, source=source, metadata=metadata)
