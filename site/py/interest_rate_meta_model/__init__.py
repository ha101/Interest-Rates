"""Public package exports with optional-dependency safety.

Core modeling modules are always imported. API/server and live-data helpers are
best-effort so that environments without web/runtime dependencies (for example
Pyodide Pages mode) can still import the package submodules safely.
"""

from __future__ import annotations

from .cache import HTTPCache, HTTPCacheConfig
from .curves import (
    DEFAULT_DASHBOARD_MATURITIES,
    TENOR_TO_TAG,
    TREASURY_TAG_TO_MATURITY_YEARS,
    ZeroCurve,
    bootstrap_zero_curve_from_par_yields,
    bootstrap_zero_curve_from_treasury_par_row,
    zero_curve_from_points,
)
from .meta import (
    MODEL_DESCRIPTIONS,
    WEIGHTING_LABELS,
    EnsemblePrediction,
    InterestRateMetaModel,
    MarketData,
    RuntimeModelConfig,
    model_catalog,
    recommended_model_names,
)
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

__all__ = [
    "HTTPCache",
    "HTTPCacheConfig",
    "ZeroCurve",
    "zero_curve_from_points",
    "bootstrap_zero_curve_from_par_yields",
    "bootstrap_zero_curve_from_treasury_par_row",
    "DEFAULT_DASHBOARD_MATURITIES",
    "TENOR_TO_TAG",
    "TREASURY_TAG_TO_MATURITY_YEARS",
    "MODEL_DESCRIPTIONS",
    "WEIGHTING_LABELS",
    "RuntimeModelConfig",
    "EnsemblePrediction",
    "InterestRateMetaModel",
    "MarketData",
    "model_catalog",
    "recommended_model_names",
    "InterestRateModel",
    "VasicekModel",
    "DothanModel",
    "ExponentiatedVasicekModel",
    "CIRModel",
    "HoLeeModel",
    "HullWhiteModel",
    "CIRPlusPlusModel",
    "GaussianHJMModel",
]


try:  # Optional runtime API surface (requires fastapi stack).
    from .api import app, create_app

    __all__.extend(["app", "create_app"])
except ModuleNotFoundError:  # pragma: no cover
    pass


try:  # Optional dashboard service (requires runtime/web dependencies).
    from .dashboard_service import DashboardService

    __all__.append("DashboardService")
except ModuleNotFoundError:  # pragma: no cover
    pass


try:  # Optional live data clients (requests/mcp integration surface).
    from .data_sources import (
        FREDClient,
        FinancialDatasetsMCPClient,
        FinancialDatasetsMCPConfig,
        NewYorkFedReferenceRateClient,
        TreasuryGovClient,
        build_market_data_from_gov_sources,
        fetch_regime_proxy_features_via_mcp,
    )

    __all__.extend(
        [
            "TreasuryGovClient",
            "NewYorkFedReferenceRateClient",
            "FREDClient",
            "build_market_data_from_gov_sources",
            "FinancialDatasetsMCPClient",
            "FinancialDatasetsMCPConfig",
            "fetch_regime_proxy_features_via_mcp",
        ]
    )
except ModuleNotFoundError:  # pragma: no cover
    pass
