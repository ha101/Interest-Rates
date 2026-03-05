#!/usr/bin/env python3
"""Prefetch rate data into static JSON files for GitHub Pages."""

from __future__ import annotations

import argparse
import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from interest_rate_meta_model.cache import HTTPCacheConfig
from interest_rate_meta_model.curves import TREASURY_TAG_TO_MATURITY_YEARS
from interest_rate_meta_model.data_sources import (
    FREDClient,
    FREDConfig,
    NewYorkFedReferenceRateClient,
    NewYorkFedReferenceRateConfig,
    TreasuryGovClient,
    TreasuryGovConfig,
)
from interest_rate_meta_model.meta import WEIGHTING_LABELS, model_catalog, recommended_model_names


def _parse_date(value: str | None, default: date) -> date:
    if not value:
        return default
    return datetime.strptime(value, "%Y-%m-%d").date()


def _iso_now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _json_default(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    raise TypeError(f"Object of type {type(value)!r} is not JSON serializable")


def _safe_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _normalize_treasury_payload(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        raise ValueError("Treasury source returned no rows for the requested date range")

    frame = frame.sort_values("date").reset_index(drop=True)
    tenors = [c for c in frame.columns if c in TREASURY_TAG_TO_MATURITY_YEARS and c != "BC_30YEARDISPLAY"]
    tenors = sorted(tenors, key=lambda c: TREASURY_TAG_TO_MATURITY_YEARS[c])
    if not tenors:
        raise ValueError("Treasury source did not return any recognized par-yield tenor columns")

    dates = [pd.Timestamp(d).date().isoformat() for d in frame["date"]]
    rows = [[_safe_float(v) for v in row] for row in frame[tenors].itertuples(index=False, name=None)]
    return {"dates": dates, "tenors": tenors, "par_yields": rows}


def _align_reference_payload(
    master_dates: list[str],
    ref_frame: pd.DataFrame,
) -> list[float | None]:
    if ref_frame.empty:
        return [None] * len(master_dates)
    ref_map = {
        pd.Timestamp(row.date).date().isoformat(): _safe_float(row.rate)
        for row in ref_frame.itertuples(index=False)
    }
    return [ref_map.get(d) for d in master_dates]


def _event_to_log(event: dict[str, Any]) -> dict[str, Any]:
    action = event.get("action") or event.get("namespace") or "fetch"
    if event.get("fallback_used"):
        message = f"{action}: used fallback source."
    elif event.get("from_cache") is True:
        message = f"{action}: served from cache."
    else:
        message = f"{action}: refreshed from source."
    return {"timestamp": event.get("timestamp"), "message": message, "details": event}


def _build_status_payload(
    *,
    cache_dir: str,
    allow_fred_fallback: bool,
    dates: list[str],
    sofr: list[float | None],
    effr: list[float | None],
    events: list[dict[str, Any]],
) -> dict[str, Any]:
    now = _iso_now_utc()
    fallback_used = any(bool(event.get("fallback_used")) for event in events)
    return {
        "supported_sources": [
            {"key": "treasury", "label": "Treasury curve (official)"},
            {"key": "sofr", "label": "SOFR (official)"},
            {"key": "effr", "label": "Fed Funds (official)"},
            {"key": "auto", "label": "Auto (official then fallback)"},
        ],
        "cache_dir": cache_dir,
        "cache_enabled": True,
        "allow_fred_fallback": bool(allow_fred_fallback),
        "cache_summary": {
            "treasury": {"entries": len(dates), "last_updated": now if dates else None},
            "newyorkfed": {"entries": sum(v is not None for v in sofr) + sum(v is not None for v in effr), "last_updated": now},
            "fred": {"entries": int(fallback_used), "last_updated": now if fallback_used else None},
        },
        "recent_logs": [_event_to_log(event) for event in events][-25:],
        "last_updated": now,
        "latest_date": dates[-1] if dates else None,
        "fallback_used": fallback_used,
        "mode": "pages_static_json",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="site/data", help="Output directory inside the Pages publish folder.")
    parser.add_argument("--start-date", default=None, help="YYYY-MM-DD (default: 1990-01-01)")
    parser.add_argument("--end-date", default=None, help="YYYY-MM-DD (default: today)")
    parser.add_argument("--cache-dir", default=".cache_prefetch", help="Local cache directory for the workflow run.")
    parser.add_argument("--allow-fred-fallback", action="store_true", help="Allow FRED fallback when NY Fed fetches fail.")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    start_date = _parse_date(args.start_date, default=date(1990, 1, 1))
    end_date = _parse_date(args.end_date, default=date.today())
    if end_date < start_date:
        raise ValueError("end-date must be on or after start-date")

    cache_cfg = HTTPCacheConfig(base_dir=args.cache_dir, enabled=True)
    treasury = TreasuryGovClient(TreasuryGovConfig(cache_config=cache_cfg))
    fred = FREDClient(FREDConfig(cache_config=cache_cfg))
    nyfed = NewYorkFedReferenceRateClient(
        NewYorkFedReferenceRateConfig(
            allow_fred_fallback=bool(args.allow_fred_fallback),
            cache_config=cache_cfg,
        ),
        fred_client=fred,
    )

    treasury_frame = treasury.fetch_range(start_date=start_date, end_date=end_date)
    treasury_payload = _normalize_treasury_payload(treasury_frame)
    master_dates = treasury_payload["dates"]

    sofr_frame = nyfed.fetch_range("sofr", start_date=start_date, end_date=end_date)
    effr_frame = nyfed.fetch_range("effr", start_date=start_date, end_date=end_date)
    ref_payload = {
        "dates": master_dates,
        "sofr": _align_reference_payload(master_dates, sofr_frame),
        "effr": _align_reference_payload(master_dates, effr_frame),
    }

    models_payload = {
        "models": model_catalog(),
        "recommended": recommended_model_names(),
        "weighting_methods": [{"key": key, "label": label} for key, label in WEIGHTING_LABELS.items()],
    }

    all_events: list[dict[str, Any]] = []
    all_events.extend(treasury.fetch_events)
    all_events.extend(nyfed.fetch_events)
    all_events.extend(fred.fetch_events)
    status_payload = _build_status_payload(
        cache_dir=args.cache_dir,
        allow_fred_fallback=bool(args.allow_fred_fallback),
        dates=master_dates,
        sofr=ref_payload["sofr"],
        effr=ref_payload["effr"],
        events=all_events,
    )

    (out_dir / "treasury_par_history.json").write_text(json.dumps(treasury_payload, indent=2, default=_json_default))
    (out_dir / "ref_rates.json").write_text(json.dumps(ref_payload, indent=2, default=_json_default))
    (out_dir / "models.json").write_text(json.dumps(models_payload, indent=2, default=_json_default))
    (out_dir / "status.json").write_text(json.dumps(status_payload, indent=2, default=_json_default))
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
