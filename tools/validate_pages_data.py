#!/usr/bin/env python3
"""Validate GitHub Pages JSON artifacts for dashboard local mode."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> Any:
    if not path.exists():
        raise ValueError(f"Missing required file: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc


def _expect_list(payload: dict[str, Any], key: str, *, where: str) -> list[Any]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise ValueError(f"{where}: '{key}' must be a list")
    return value


def validate_pages_data(data_dir: Path, *, require_nonempty: bool) -> None:
    models = _load_json(data_dir / "models.json")
    status = _load_json(data_dir / "status.json")
    treasury = _load_json(data_dir / "treasury_par_history.json")
    ref_rates = _load_json(data_dir / "ref_rates.json")

    if not isinstance(models, dict):
        raise ValueError("models.json must be an object")
    models_rows = _expect_list(models, "models", where="models.json")
    _expect_list(models, "recommended", where="models.json")
    _expect_list(models, "weighting_methods", where="models.json")
    if require_nonempty and not models_rows:
        raise ValueError("models.json must include at least one model")

    if not isinstance(status, dict):
        raise ValueError("status.json must be an object")
    _expect_list(status, "supported_sources", where="status.json")
    _expect_list(status, "recent_logs", where="status.json")

    if not isinstance(treasury, dict):
        raise ValueError("treasury_par_history.json must be an object")
    treasury_dates = _expect_list(treasury, "dates", where="treasury_par_history.json")
    treasury_tenors = _expect_list(treasury, "tenors", where="treasury_par_history.json")
    treasury_rows = _expect_list(treasury, "par_yields", where="treasury_par_history.json")
    if len(treasury_dates) != len(treasury_rows):
        raise ValueError("treasury_par_history.json: dates and par_yields length mismatch")
    for idx, row in enumerate(treasury_rows):
        if not isinstance(row, list):
            raise ValueError(f"treasury_par_history.json: par_yields[{idx}] must be a list")
        if len(row) != len(treasury_tenors):
            raise ValueError(f"treasury_par_history.json: par_yields[{idx}] length must match tenors length")
    if require_nonempty and not treasury_dates:
        raise ValueError("treasury_par_history.json must contain at least one date row")

    if not isinstance(ref_rates, dict):
        raise ValueError("ref_rates.json must be an object")
    ref_dates = _expect_list(ref_rates, "dates", where="ref_rates.json")
    ref_sofr = _expect_list(ref_rates, "sofr", where="ref_rates.json")
    ref_effr = _expect_list(ref_rates, "effr", where="ref_rates.json")
    if len(ref_dates) != len(ref_sofr) or len(ref_dates) != len(ref_effr):
        raise ValueError("ref_rates.json: dates, sofr, and effr arrays must have identical lengths")
    if require_nonempty and not ref_dates:
        raise ValueError("ref_rates.json must contain at least one date row")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="site/data", help="Directory containing Pages JSON artifacts.")
    parser.add_argument(
        "--require-nonempty",
        action="store_true",
        help="Fail if primary datasets are empty.",
    )
    args = parser.parse_args()
    validate_pages_data(Path(args.dir), require_nonempty=bool(args.require_nonempty))
    print(f"Validated Pages data in {args.dir}")


if __name__ == "__main__":
    main()
