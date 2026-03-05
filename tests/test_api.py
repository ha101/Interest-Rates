from fastapi.testclient import TestClient

from interest_rate_meta_model.api import create_app


class FakeService:
    def run_dashboard(self, params):
        return {"ok": True, "params": params}

    def scenario(self, payload):
        return {"scenario_type": payload["scenario_type"], "deltas_bps": []}

    def source_status(self, **kwargs):
        return {"supported_sources": [], "cache_dir": "/tmp/cache", "recent_logs": []}

    def clear_cache(self, namespace=None):
        return {"ok": True, "namespace": namespace}

    def available_models(self):
        return {"models": [{"name": "vasicek", "label": "Vasicek"}], "recommended": ["vasicek"], "weighting_methods": []}

    def run_single_model(self, params, model_name):
        return {"model": model_name, "params": params}


class ErrorService(FakeService):
    def __init__(self, exc):
        self.exc = exc

    def run_dashboard(self, params):
        raise self.exc


def test_root_renders_html():
    app = create_app(FakeService())
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert "Interest Rate Meta Model Dashboard" in response.text


def test_models_endpoint():
    app = create_app(FakeService())
    client = TestClient(app)
    response = client.get("/models")
    assert response.status_code == 200
    body = response.json()
    assert body["recommended"] == ["vasicek"]


def test_run_and_scenario_endpoints():
    app = create_app(FakeService())
    client = TestClient(app)

    run_response = client.post(
        "/run",
        json={
            "data_source": "auto",
            "horizon": "1m",
            "target": "10y",
        },
    )
    assert run_response.status_code == 200
    assert run_response.json()["ok"] is True

    scenario_response = client.post(
        "/scenario",
        json={
            "maturities": [1.0, 2.0],
            "zero_curve": [0.03, 0.032],
            "scenario_type": "parallel_25bp",
        },
    )
    assert scenario_response.status_code == 200
    assert scenario_response.json()["scenario_type"] == "parallel_25bp"


def test_run_returns_sanitized_500_for_unexpected_errors():
    app = create_app(ErrorService(RuntimeError("sensitive stack detail")))
    client = TestClient(app)
    response = client.post(
        "/run",
        json={
            "data_source": "auto",
            "horizon": "1m",
            "target": "10y",
        },
    )
    assert response.status_code == 500
    assert response.json()["detail"] == "Run failed due to an internal error."


def test_run_preserves_400_for_user_errors():
    app = create_app(ErrorService(ValueError("bad input")))
    client = TestClient(app)
    response = client.post(
        "/run",
        json={
            "data_source": "auto",
            "horizon": "1m",
            "target": "10y",
        },
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "bad input"
