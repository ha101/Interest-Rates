from __future__ import annotations

import asyncio
import io
import json
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from .cache import HTTPCache, HTTPCacheConfig
from .curves import DEFAULT_DASHBOARD_MATURITIES, TENOR_TO_TAG, TREASURY_TAG_TO_MATURITY_YEARS, ZeroCurve


try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
except Exception:  # pragma: no cover - optional dependency
    ClientSession = None
    StdioServerParameters = None
    stdio_client = None


TREASURY_XML_BASE_URL = "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/pages/xml"
FRED_CSV_BASE_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
NYFED_MARKETS_API_BASE_URL = "https://markets.newyorkfed.org/api/rates"

_ATOM_NS = "http://www.w3.org/2005/Atom"
_M_NS = "http://schemas.microsoft.com/ado/2007/08/dataservices/metadata"
_D_NS = "http://schemas.microsoft.com/ado/2007/08/dataservices"
_NAMESPACES = {"atom": _ATOM_NS, "m": _M_NS, "d": _D_NS}

OFFICIAL_REFERENCE_RATE_ENDPOINTS: dict[str, tuple[str, str]] = {
    "sofr": ("secured", "sofr"),
    "effr": ("unsecured", "effr"),
}

FRED_REFERENCE_RATE_SERIES: dict[str, tuple[str, ...]] = {
    "sofr": ("SOFR",),
    "effr": ("EFFR", "DFF"),
}

TARGET_TO_TREASURY_TAG = {
    "2y": TENOR_TO_TAG["2Y"],
    "10y": TENOR_TO_TAG["10Y"],
    "30y": TENOR_TO_TAG["30Y"],
}


@dataclass(slots=True)
class TreasuryGovConfig:
    timeout_seconds: int = 30
    user_agent: str = "interest-rate-meta-model-dashboard/0.1"
    cache_config: HTTPCacheConfig | None = None


@dataclass(slots=True)
class FREDConfig:
    timeout_seconds: int = 30
    user_agent: str = "interest-rate-meta-model-dashboard/0.1"
    cache_config: HTTPCacheConfig | None = None


@dataclass(slots=True)
class NewYorkFedReferenceRateConfig:
    timeout_seconds: int = 30
    user_agent: str = "interest-rate-meta-model-dashboard/0.1"
    allow_fred_fallback: bool = True
    cache_config: HTTPCacheConfig | None = None


@dataclass(slots=True)
class TreasuryFetchSummary:
    zero_curve: ZeroCurve
    latest_row: pd.Series
    history_frame: pd.DataFrame


class _CachedHTTPMixin:
    fetch_events: list[dict[str, Any]]

    def _record_event(self, **kwargs: Any) -> None:
        if not hasattr(self, "fetch_events"):
            self.fetch_events = []
        event = {**kwargs}
        event.setdefault("timestamp", pd.Timestamp.utcnow().isoformat())
        self.fetch_events.append(event)

    def _cache_get(
        self,
        namespace: str,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        ttl_seconds: int | None = None,
    ) -> requests.Response:
        cached = self.cache.get(namespace, url, params=params, ttl_seconds=ttl_seconds)
        if cached is not None:
            content, meta = cached
            resp = requests.Response()
            resp.status_code = int(meta.get("status_code", 200))
            resp._content = content
            resp.headers = requests.structures.CaseInsensitiveDict(meta.get("headers", {}))
            resp.url = str(meta.get("url", url))
            resp.encoding = requests.utils.get_encoding_from_headers(resp.headers)
            resp._from_cache = True  # type: ignore[attr-defined]
            self._record_event(namespace=namespace, url=resp.url, params=params or {}, from_cache=True, status_code=resp.status_code)
            return resp

        resp = self.session.get(url, params=params, timeout=self.timeout_seconds)
        resp.raise_for_status()
        self.cache.set(
            namespace,
            url,
            content=resp.content,
            params=params,
            status_code=resp.status_code,
            headers=dict(resp.headers),
        )
        resp._from_cache = False  # type: ignore[attr-defined]
        self._record_event(namespace=namespace, url=resp.url, params=params or {}, from_cache=False, status_code=resp.status_code)
        return resp


def _normalize_date(value: str | date | pd.Timestamp | None) -> pd.Timestamp:
    if value is None:
        return pd.Timestamp(date.today()).normalize()
    return pd.Timestamp(value).normalize()


def normalize_treasury_tenor(tenor: str) -> str:
    clean = re.sub(r"\s+", "", tenor.upper())
    clean = clean.replace("MO", "M").replace("MON", "M").replace("YR", "Y")
    if clean in TENOR_TO_TAG:
        return clean
    raise ValueError(f"Unsupported Treasury tenor {tenor!r}. Choose one of: {', '.join(TENOR_TO_TAG)}")


def normalize_short_rate_source(source: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9]+", "", source).lower()
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


class TreasuryGovClient(_CachedHTTPMixin):
    """Pull U.S. Treasury par-yield data from the official XML feed."""

    def __init__(
        self,
        config: TreasuryGovConfig | None = None,
        session: requests.Session | None = None,
        cache: HTTPCache | None = None,
    ) -> None:
        self.config = config or TreasuryGovConfig()
        self.timeout_seconds = self.config.timeout_seconds
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": self.config.user_agent})
        self.cache = cache or HTTPCache(self.config.cache_config)
        self.fetch_events = []

    def _year_ttl_seconds(self, year: int) -> int | None:
        return None if year < date.today().year else self.cache.config.default_ttl_seconds

    def _get(self, params: dict[str, Any], ttl_seconds: int | None = None) -> requests.Response:
        return self._cache_get("treasury", TREASURY_XML_BASE_URL, params=params, ttl_seconds=ttl_seconds)

    def fetch_year(self, year: int) -> pd.DataFrame:
        if year < 1990:
            raise ValueError("Treasury daily par yield curve data is available from 1990 onward")
        resp = self._get(
            {"data": "daily_treasury_yield_curve", "field_tdr_date_value": int(year)},
            ttl_seconds=self._year_ttl_seconds(year),
        )
        df = self._parse_treasury_par_xml(resp.content)
        self._record_event(action="treasury.fetch_year", year=year, rows=int(len(df)))
        return df

    def fetch_range(self, start_date: str | date | pd.Timestamp, end_date: str | date | pd.Timestamp) -> pd.DataFrame:
        start_ts = _normalize_date(start_date)
        end_ts = _normalize_date(end_date)
        if end_ts < start_ts:
            raise ValueError("end_date must be on or after start_date")
        years = range(start_ts.year, end_ts.year + 1)
        frames = [self.fetch_year(y) for y in years]
        if not frames:
            return pd.DataFrame(columns=["date", *TENOR_TO_TAG.values()])
        df = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["date"]).sort_values("date")
        out = df[(df["date"] >= start_ts) & (df["date"] <= end_ts)].reset_index(drop=True)
        self._record_event(action="treasury.fetch_range", start=str(start_ts.date()), end=str(end_ts.date()), rows=int(len(out)))
        return out

    def fetch_latest_available(self, as_of: str | date | pd.Timestamp | None = None, lookback_days: int = 14) -> pd.Series:
        as_of_ts = _normalize_date(as_of)
        start = as_of_ts - pd.Timedelta(days=max(int(lookback_days), 3))
        df = self.fetch_range(start, as_of_ts)
        if df.empty:
            raise ValueError(f"No Treasury yield-curve observations found on or before {as_of_ts.date()}")
        row = df.sort_values("date").iloc[-1]
        self._record_event(action="treasury.fetch_latest_available", as_of=str(as_of_ts.date()), selected_date=str(pd.Timestamp(row['date']).date()))
        return row

    def fetch_target_history(
        self,
        target: str,
        start_date: str | date | pd.Timestamp,
        end_date: str | date | pd.Timestamp,
    ) -> pd.Series:
        target_key = target.lower()
        if target_key not in TARGET_TO_TREASURY_TAG:
            raise ValueError(f"Unsupported Treasury target history {target!r}")
        tag = TARGET_TO_TREASURY_TAG[target_key]
        df = self.fetch_range(start_date, end_date)
        if tag not in df.columns:
            raise ValueError(f"Treasury source did not return target column {tag}")
        series = (
            df[["date", tag]]
            .dropna()
            .rename(columns={tag: "rate"})
            .set_index("date")["rate"]
            .astype(float)
            / 100.0
        )
        series.name = target_key
        return series.sort_index()

    @staticmethod
    def _parse_treasury_par_xml(xml_bytes: bytes) -> pd.DataFrame:
        root = ET.fromstring(xml_bytes)
        rows: list[dict[str, Any]] = []
        for entry in root.findall(".//atom:entry", _NAMESPACES):
            props = entry.find(".//m:properties", _NAMESPACES)
            if props is None:
                continue
            row: dict[str, Any] = {}
            for child in list(props):
                tag = child.tag.split("}", 1)[-1]
                text = (child.text or "").strip()
                if not text:
                    continue
                if tag == "NEW_DATE":
                    row["date"] = pd.Timestamp(text[:10])
                else:
                    try:
                        row[tag] = float(text)
                    except ValueError:
                        row[tag] = text
            if "date" in row:
                rows.append(row)

        if not rows:
            return pd.DataFrame(columns=["date", *sorted(TREASURY_TAG_TO_MATURITY_YEARS)])

        df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
        if "BC_30YEARDISPLAY" in df.columns and "BC_30YEAR" in df.columns:
            df = df.drop(columns=["BC_30YEARDISPLAY"])
        return df


class FREDClient(_CachedHTTPMixin):
    """Simple no-key FRED CSV downloader used as a fallback layer."""

    def __init__(
        self,
        config: FREDConfig | None = None,
        session: requests.Session | None = None,
        cache: HTTPCache | None = None,
    ) -> None:
        self.config = config or FREDConfig()
        self.timeout_seconds = self.config.timeout_seconds
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": self.config.user_agent})
        self.cache = cache or HTTPCache(self.config.cache_config)
        self.fetch_events = []

    def _range_ttl_seconds(self, end_date: str | date | pd.Timestamp) -> int | None:
        end_ts = _normalize_date(end_date)
        return None if end_ts < _normalize_date(None) else self.cache.config.default_ttl_seconds

    def fetch_series(
        self,
        series_id: str,
        start_date: str | date | pd.Timestamp,
        end_date: str | date | pd.Timestamp,
    ) -> pd.DataFrame:
        start_ts = _normalize_date(start_date)
        end_ts = _normalize_date(end_date)
        if end_ts < start_ts:
            raise ValueError("end_date must be on or after start_date")
        params = {"id": series_id.upper(), "cosd": start_ts.date().isoformat(), "coed": end_ts.date().isoformat()}
        resp = self._cache_get("fred", FRED_CSV_BASE_URL, params=params, ttl_seconds=self._range_ttl_seconds(end_ts))
        text = resp.content.decode("utf-8", errors="replace")
        df = pd.read_csv(io.StringIO(text))
        if df.empty or len(df.columns) < 2:
            raise ValueError(f"Unexpected FRED CSV payload for series {series_id!r}")
        date_col, value_col = df.columns[:2]
        out = pd.DataFrame({
            "date": pd.to_datetime(df[date_col], errors="coerce"),
            "rate": pd.to_numeric(df[value_col].replace(".", np.nan), errors="coerce"),
        }).dropna(subset=["date", "rate"])
        if out.empty:
            raise ValueError(f"FRED series {series_id!r} returned no usable observations")
        out = out.sort_values("date").reset_index(drop=True)
        out["series_id"] = series_id.upper()
        out["source"] = "FRED"
        self._record_event(action="fred.fetch_series", series_id=series_id.upper(), rows=int(len(out)))
        return out

    def fetch_reference_rate(
        self,
        rate_name: str,
        start_date: str | date | pd.Timestamp,
        end_date: str | date | pd.Timestamp,
    ) -> pd.DataFrame:
        rate_key = normalize_short_rate_source(rate_name)
        if rate_key == "treasury":
            raise ValueError("FRED reference-rate fallback only applies to SOFR or EFFR")
        series_ids = FRED_REFERENCE_RATE_SERIES[rate_key]
        last_error: Exception | None = None
        for series_id in series_ids:
            try:
                df = self.fetch_series(series_id, start_date, end_date)
                df["rate_name"] = rate_key
                return df
            except Exception as exc:  # pragma: no cover
                last_error = exc
        raise ValueError(f"FRED fallback failed for {rate_key!r}: {last_error}")


class NewYorkFedReferenceRateClient(_CachedHTTPMixin):
    """Load SOFR and EFFR from official New York Fed endpoints with FRED fallback."""

    def __init__(
        self,
        config: NewYorkFedReferenceRateConfig | None = None,
        session: requests.Session | None = None,
        cache: HTTPCache | None = None,
        fred_client: FREDClient | None = None,
    ) -> None:
        self.config = config or NewYorkFedReferenceRateConfig()
        self.timeout_seconds = self.config.timeout_seconds
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": self.config.user_agent})
        self.cache = cache or HTTPCache(self.config.cache_config)
        self.fred_client = fred_client or FREDClient(
            FREDConfig(
                timeout_seconds=self.config.timeout_seconds,
                user_agent=self.config.user_agent,
                cache_config=self.config.cache_config,
            ),
            session=self.session,
            cache=self.cache,
        )
        self.fetch_events = []

    def _range_ttl_seconds(self, end_date: str | date | pd.Timestamp) -> int | None:
        end_ts = _normalize_date(end_date)
        return None if end_ts < _normalize_date(None) else self.cache.config.default_ttl_seconds

    def _official_search_json(
        self,
        rate_name: str,
        start_date: str | date | pd.Timestamp,
        end_date: str | date | pd.Timestamp,
    ) -> dict[str, Any]:
        rate_key = normalize_short_rate_source(rate_name)
        if rate_key not in OFFICIAL_REFERENCE_RATE_ENDPOINTS:
            raise ValueError(f"No official New York Fed loader is available for {rate_name!r}")
        bucket, slug = OFFICIAL_REFERENCE_RATE_ENDPOINTS[rate_key]
        url = f"{NYFED_MARKETS_API_BASE_URL}/{bucket}/{slug}/search.json"
        params = {"startDate": _normalize_date(start_date).date().isoformat(), "endDate": _normalize_date(end_date).date().isoformat()}
        resp = self._cache_get("newyorkfed", url, params=params, ttl_seconds=self._range_ttl_seconds(end_date))
        return json.loads(resp.content.decode("utf-8", errors="replace"))

    @staticmethod
    def _parse_newyorkfed_reference_json(payload: dict[str, Any], rate_name: str) -> pd.DataFrame:
        rows: Any = None
        for key in ("refRates", "ref_rates", "data", "results"):
            candidate = payload.get(key)
            if isinstance(candidate, list):
                rows = candidate
                break
        if rows is None and isinstance(payload, list):
            rows = payload
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"Unexpected New York Fed JSON payload for {rate_name!r}")

        cleaned_rows: list[dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            normalized = {re.sub(r"[^a-z0-9]", "", str(k).lower()): v for k, v in row.items()}
            rate_type = str(normalized.get("type", rate_name)).strip().lower()
            if rate_type and rate_name.lower() not in rate_type and rate_type not in {"", rate_name.lower()}:
                continue
            date_text = normalized.get("effectivedate") or normalized.get("date") or normalized.get("asofdate")
            rate_value = normalized.get("percentrate") or normalized.get("rate") or normalized.get("value")
            if date_text in (None, "") or rate_value in (None, ""):
                continue
            try:
                cleaned_rows.append({"date": pd.Timestamp(str(date_text)[:10]), "rate": float(rate_value)})
            except Exception:
                continue

        if not cleaned_rows:
            raise ValueError(f"New York Fed payload for {rate_name!r} contained no usable rows")

        df = pd.DataFrame(cleaned_rows).drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
        df["rate_name"] = rate_name.lower()
        df["source"] = "Federal Reserve Bank of New York"
        return df

    def fetch_range(
        self,
        rate_name: str,
        start_date: str | date | pd.Timestamp,
        end_date: str | date | pd.Timestamp,
    ) -> pd.DataFrame:
        rate_key = normalize_short_rate_source(rate_name)
        if rate_key == "treasury":
            raise ValueError("Use TreasuryGovClient for Treasury-tenor history")

        last_error: Exception | None = None
        try:
            payload = self._official_search_json(rate_key, start_date, end_date)
            df = self._parse_newyorkfed_reference_json(payload, rate_key)
            mask = (df["date"] >= _normalize_date(start_date)) & (df["date"] <= _normalize_date(end_date))
            df = df.loc[mask].reset_index(drop=True)
            if df.empty:
                raise ValueError(f"Official New York Fed source returned no rows for {rate_key!r}")
            self._record_event(action="nyfed.fetch_range", rate_name=rate_key, rows=int(len(df)), fallback_used=False)
            return df
        except Exception as exc:
            last_error = exc

        if not self.config.allow_fred_fallback:
            raise ValueError(f"Official New York Fed request failed for {rate_key!r}: {last_error}")

        df = self.fred_client.fetch_reference_rate(rate_key, start_date, end_date)
        df["source"] = "FRED"
        self._record_event(action="nyfed.fetch_range", rate_name=rate_key, rows=int(len(df)), fallback_used=True, fallback_reason=str(last_error))
        return df

    def fetch_latest_available(
        self,
        rate_name: str,
        as_of: str | date | pd.Timestamp | None = None,
        lookback_days: int = 14,
    ) -> pd.Series:
        as_of_ts = _normalize_date(as_of)
        start_ts = as_of_ts - pd.Timedelta(days=max(int(lookback_days), 3))
        df = self.fetch_range(rate_name, start_ts, as_of_ts)
        if df.empty:
            raise ValueError(f"No {rate_name} observations found on or before {as_of_ts.date()}")
        row = df.sort_values("date").iloc[-1]
        self._record_event(action="nyfed.fetch_latest_available", rate_name=rate_name, selected_date=str(pd.Timestamp(row['date']).date()))
        return row


# ---------------------------------------------------------------------------
# Higher-level helpers
# ---------------------------------------------------------------------------


def build_short_rate_history_from_treasury(
    client: TreasuryGovClient | None = None,
    *,
    start_date: str | date | pd.Timestamp,
    end_date: str | date | pd.Timestamp,
    tenor: str = "3M",
) -> pd.Series:
    client = client or TreasuryGovClient()
    tenor_key = normalize_treasury_tenor(tenor)
    tag = TENOR_TO_TAG[tenor_key]
    df = client.fetch_range(start_date=start_date, end_date=end_date)
    if tag not in df.columns:
        raise ValueError(f"Treasury source did not return tenor column {tag}")
    series = (
        df[["date", tag]]
        .dropna()
        .rename(columns={tag: "rate"})
        .set_index("date")["rate"]
        .astype(float)
        / 100.0
    )
    series.name = f"treasury_{tenor_key.lower()}_short_rate_proxy"
    if series.empty:
        raise ValueError("No short-rate history was available from Treasury for the requested range")
    return series.sort_index()



def build_short_rate_history_from_reference_rate(
    client: NewYorkFedReferenceRateClient | None = None,
    *,
    rate_name: str = "sofr",
    start_date: str | date | pd.Timestamp,
    end_date: str | date | pd.Timestamp,
) -> pd.Series:
    client = client or NewYorkFedReferenceRateClient()
    rate_key = normalize_short_rate_source(rate_name)
    if rate_key == "treasury":
        raise ValueError("Use build_short_rate_history_from_treasury for Treasury tenor proxies")
    df = client.fetch_range(rate_key, start_date, end_date)
    series = df[["date", "rate"]].dropna().set_index("date")["rate"].astype(float) / 100.0
    series.name = f"{rate_key}_short_rate_proxy"
    if series.empty:
        raise ValueError(f"No {rate_key} history was available for the requested range")
    return series.sort_index()



def fetch_treasury_summary(
    client: TreasuryGovClient | None = None,
    *,
    as_of: str | date | pd.Timestamp | None = None,
    history_years: float = 10.0,
) -> TreasuryFetchSummary:
    client = client or TreasuryGovClient()
    as_of_ts = _normalize_date(as_of)
    start_ts = as_of_ts - pd.Timedelta(days=int(round(history_years * 366)))
    history_frame = client.fetch_range(start_ts, as_of_ts)
    latest_row = history_frame.sort_values("date").iloc[-1] if not history_frame.empty else client.fetch_latest_available(as_of_ts)
    curve = ZeroCurve.from_treasury_par_yields(latest_row, as_of=pd.Timestamp(latest_row["date"]))
    return TreasuryFetchSummary(zero_curve=curve, latest_row=latest_row, history_frame=history_frame)



def build_market_data_from_gov_sources(
    *,
    history_years: float = 10.0,
    short_rate_tenor: str = "3M",
    short_rate_source: str = "treasury",
    as_of: str | date | pd.Timestamp | None = None,
    treasury_client: TreasuryGovClient | None = None,
    reference_rate_client: NewYorkFedReferenceRateClient | None = None,
    regime_features: dict[str, float] | None = None,
):
    from .meta import MarketData  # local import to avoid circular dependency

    treasury_client = treasury_client or TreasuryGovClient()
    reference_rate_client = reference_rate_client or NewYorkFedReferenceRateClient()
    as_of_ts = _normalize_date(as_of)
    start_ts = as_of_ts - pd.Timedelta(days=int(round(history_years * 366)))
    source_key = normalize_short_rate_source(short_rate_source)

    if source_key == "auto":
        candidates = ["sofr", "effr", "treasury"]
        last_error: Exception | None = None
        short_rates = None
        selected_source = None
        for candidate in candidates:
            try:
                if candidate == "treasury":
                    short_rates = build_short_rate_history_from_treasury(
                        treasury_client,
                        start_date=start_ts,
                        end_date=as_of_ts,
                        tenor=short_rate_tenor,
                    )
                else:
                    short_rates = build_short_rate_history_from_reference_rate(
                        reference_rate_client,
                        rate_name=candidate,
                        start_date=start_ts,
                        end_date=as_of_ts,
                    )
                selected_source = candidate
                break
            except Exception as exc:  # pragma: no cover - only on live failures
                last_error = exc
        if short_rates is None:
            raise ValueError(f"Auto data-source selection failed: {last_error}")
        source_key = selected_source or source_key
    elif source_key == "treasury":
        short_rates = build_short_rate_history_from_treasury(
            treasury_client,
            start_date=start_ts,
            end_date=as_of_ts,
            tenor=short_rate_tenor,
        )
    else:
        short_rates = build_short_rate_history_from_reference_rate(
            reference_rate_client,
            rate_name=source_key,
            start_date=start_ts,
            end_date=as_of_ts,
        )

    treasury_summary = fetch_treasury_summary(treasury_client, as_of=as_of_ts, history_years=history_years)
    md = MarketData(short_rates=short_rates, current_curve=treasury_summary.zero_curve, regime_features=regime_features)
    md.metadata = {
        "selected_short_rate_source": source_key,
        "treasury_latest_row": treasury_summary.latest_row,
        "treasury_history_frame": treasury_summary.history_frame,
        "curve_source": "treasury",
    }
    return md


# ---------------------------------------------------------------------------
# Optional Financial Datasets MCP adapter
# ---------------------------------------------------------------------------


@dataclass
class FinancialDatasetsMCPConfig:
    command: str = "uv"
    server_dir: str = "./mcp-server"
    args: tuple[str, ...] = ("run", "server.py")
    env: dict[str, str] | None = None


class FinancialDatasetsMCPClient:
    def __init__(self, config: FinancialDatasetsMCPConfig) -> None:
        self.config = config
        if ClientSession is None or StdioServerParameters is None or stdio_client is None:
            raise ImportError("The optional 'mcp' package is required. Install with: pip install mcp")

    async def _call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        server_params = StdioServerParameters(command=self.config.command, args=["--directory", self.config.server_dir, *self.config.args], env=self.config.env)
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(name, arguments=arguments)

        structured = getattr(result, "structuredContent", None)
        if structured not in (None, {}):
            return structured

        content = getattr(result, "content", None)
        if not content:
            return result
        first = content[0]
        text = getattr(first, "text", None)
        if text is None:
            return content
        try:
            return json.loads(text)
        except Exception:
            return text

    async def get_historical_stock_prices(self, ticker: str, start_date: str, end_date: str, interval: str = "day", interval_multiplier: int = 1) -> Any:
        return await self._call_tool(
            "get_historical_stock_prices",
            {
                "ticker": ticker,
                "start_date": start_date,
                "end_date": end_date,
                "interval": interval,
                "interval_multiplier": interval_multiplier,
            },
        )

    async def get_company_news(self, ticker: str) -> Any:
        return await self._call_tool("get_company_news", {"ticker": ticker})


async def fetch_regime_proxy_features_via_mcp(
    client: FinancialDatasetsMCPClient,
    equity_ticker: str = "SPY",
    duration_ticker: str = "TLT",
    credit_ticker: str = "HYG",
    lookback_days: int = 90,
) -> dict[str, float]:
    end = date.today()
    start = end - pd.Timedelta(days=lookback_days)
    start_s = start.isoformat()
    end_s = end.isoformat()

    eq_raw, dur_raw, cr_raw = await asyncio.gather(
        client.get_historical_stock_prices(equity_ticker, start_s, end_s),
        client.get_historical_stock_prices(duration_ticker, start_s, end_s),
        client.get_historical_stock_prices(credit_ticker, start_s, end_s),
    )

    def _frame(raw: Any) -> pd.DataFrame:
        if isinstance(raw, dict):
            for key in ("prices", "data", "results"):
                if key in raw and isinstance(raw[key], list):
                    raw = raw[key]
                    break
        if not isinstance(raw, list) or not raw:
            raise ValueError(f"Unexpected MCP payload format: {type(raw)!r}")
        df = pd.DataFrame(raw)
        candidates = [c for c in ("time", "date", "datetime", "timestamp") if c in df.columns]
        if candidates:
            df[candidates[0]] = pd.to_datetime(df[candidates[0]])
            df = df.sort_values(candidates[0])
        price_col = next((c for c in ("close", "adj_close", "price") if c in df.columns), None)
        if price_col is None:
            raise ValueError(f"Could not find a close-like column in: {df.columns.tolist()}")
        return df[[price_col]].rename(columns={price_col: "close"})

    eq = _frame(eq_raw)
    dur = _frame(dur_raw)
    cr = _frame(cr_raw)

    def _features(df: pd.DataFrame) -> tuple[float, float]:
        returns = df["close"].pct_change().dropna()
        trailing_return = float(df["close"].iloc[-1] / df["close"].iloc[0] - 1.0)
        trailing_vol = float(returns.std(ddof=0) * np.sqrt(252.0))
        return trailing_return, trailing_vol

    eq_ret, eq_vol = _features(eq)
    dur_ret, dur_vol = _features(dur)
    cr_ret, cr_vol = _features(cr)

    risk_off_score = max(0.0, -eq_ret) + max(0.0, dur_ret) + max(0.0, -cr_ret)
    return {
        "risk_off_score": float(risk_off_score),
        "equity_return": eq_ret,
        "equity_vol": eq_vol,
        "duration_return": dur_ret,
        "duration_vol": dur_vol,
        "credit_return": cr_ret,
        "credit_vol": cr_vol,
    }
