import json

import pytest

from tools.validate_pages_data import validate_pages_data


def _write(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_validate_pages_data_accepts_valid_payloads(tmp_path):
    _write(
        tmp_path / "models.json",
        {
            "models": [{"name": "vasicek"}],
            "recommended": ["vasicek"],
            "weighting_methods": [{"key": "performance", "label": "Performance-based"}],
        },
    )
    _write(
        tmp_path / "status.json",
        {"supported_sources": [], "recent_logs": []},
    )
    _write(
        tmp_path / "treasury_par_history.json",
        {
            "dates": ["2026-01-05"],
            "tenors": ["BC_1MONTH"],
            "par_yields": [[4.5]],
        },
    )
    _write(
        tmp_path / "ref_rates.json",
        {
            "dates": ["2026-01-05"],
            "sofr": [4.6],
            "effr": [4.6],
        },
    )

    validate_pages_data(tmp_path, require_nonempty=True)


def test_validate_pages_data_rejects_empty_primary_data_when_required(tmp_path):
    _write(
        tmp_path / "models.json",
        {
            "models": [{"name": "vasicek"}],
            "recommended": ["vasicek"],
            "weighting_methods": [{"key": "performance", "label": "Performance-based"}],
        },
    )
    _write(
        tmp_path / "status.json",
        {"supported_sources": [], "recent_logs": []},
    )
    _write(
        tmp_path / "treasury_par_history.json",
        {"dates": [], "tenors": [], "par_yields": []},
    )
    _write(
        tmp_path / "ref_rates.json",
        {"dates": [], "sofr": [], "effr": []},
    )

    with pytest.raises(ValueError):
        validate_pages_data(tmp_path, require_nonempty=True)
