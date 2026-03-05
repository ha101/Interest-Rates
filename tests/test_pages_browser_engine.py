from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_browser_engine():
    module_path = Path(__file__).resolve().parents[1] / "site" / "py" / "browser_engine.py"
    spec = importlib.util.spec_from_file_location("pages_browser_engine", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_run_scenario_returns_expected_shape():
    engine = _load_browser_engine()
    payload = {
        "maturities": [1.0, 2.0, 5.0, 10.0],
        "zero_curve": [0.03, 0.031, 0.032, 0.033],
        "scenario_type": "parallel_25bp",
    }
    out = engine.run_scenario(payload)
    assert out["scenario_type"] == "parallel_25bp"
    assert len(out["base"]["maturities"]) == 4
    assert len(out["shocked"]["maturities"]) == 4
    assert len(out["deltas_bps"]) == 4


def test_run_local_returns_dashboard_contract():
    engine = _load_browser_engine()
    data = {
        "treasury_par_history": {
            "dates": ["2026-01-02", "2026-01-05"],
            "tenors": ["BC_1MONTH", "BC_3MONTH", "BC_1YEAR", "BC_10YEAR"],
            "par_yields": [
                [4.50, 4.55, 4.40, 4.20],
                [4.45, 4.50, 4.35, 4.15],
            ],
        },
        "ref_rates": {
            "dates": ["2026-01-02", "2026-01-05"],
            "sofr": [4.60, 4.58],
            "effr": [4.62, 4.60],
        },
        "status": {
            "cache_dir": "site/data",
            "recent_logs": [],
            "fallback_used": False,
        },
    }
    payload = {
        "data_source": "treasury",
        "as_of_date": "2026-01-05",
        "horizon": "1m",
        "target": "10y",
        "history_years": 2.0,
        "short_rate_tenor": "3M",
        "weighting_method": "performance",
        "optimization_mode": "fast",
    }

    out = engine.run_local(payload, data)
    assert out["request"]["target"] == "10y"
    assert "overview" in out
    assert "charts" in out
    assert "curve_explorer" in out
    assert "diagnostics" in out
