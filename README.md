# Interest Rate Meta Model Dashboard

This package extends the Chapter 3 interest-rate meta-model with a full browser dashboard and a small FastAPI backend.

## What is included

- The original short-rate / term-structure models:
  - Vasicek
  - Dothan
  - Exponentiated Vasicek
  - CIR
  - Ho-Lee
  - Hull-White extended Vasicek
  - CIR++
  - one-factor Gaussian HJM
- Government data loaders:
  - U.S. Treasury par-curve XML feed
  - New York Fed SOFR / EFFR loaders
  - FRED fallback
  - local file-backed cache
- A browser dashboard with:
  - Simple mode for one-click runs
  - Advanced mode for scenarios, diagnostics, and cache controls
  - Overview / Curve Explorer / Scenario / Diagnostics / Data & Cache tabs
  - shareable URL state
  - CSV / JSON downloads
- Backend endpoints:
  - `POST /run`
  - `POST /scenario`
  - `GET /sources/status`
  - `POST /cache/clear`
  - `GET /models`
  - `POST /models/run_one`

## Quick start

```bash
cd interest_rate_meta_model_dashboard
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
python examples/run_api.py
```

Then open:

```text
http://127.0.0.1:8000
```

## Default first-run behavior

The dashboard is set up so the first run works without touching advanced settings:

- Source: `Auto`
- Horizon: `1 month`
- Target: `10Y yield`
- History window: `10 years`
- Models: recommended subset
- Cache: on
- FRED fallback: on

## Notes

- The current curve is always bootstrapped from U.S. Treasury par yields.
- When the short-rate source is `SOFR` or `Fed Funds`, the short-rate history is sourced from official New York Fed data when available and falls back to FRED only if needed.
- The dashboard keeps a local cache under `~/.cache/interest-rate-meta-model` by default.
- The front end is plain HTML/CSS/JavaScript, so there is no Node build step.

## Testing

```bash
pytest -q
```


## GitHub Pages mode (Option 2: prefetch + static JSON + in-browser models)


This repo supports a **no-backend** deployment pattern suitable for **GitHub Pages**:

- A scheduled GitHub Action (`.github/workflows/prefetch_pages_data.yml`) fetches:
  - U.S. Treasury Daily Par Yield Curve history,
  - SOFR and EFFR from official New York Fed endpoints,
  - optional FRED fallback.
- The Action writes normalized JSON into `site/data/` and commits it back to the repo.
- A deploy GitHub Action (`.github/workflows/deploy_pages.yml`) publishes `site/` to GitHub Pages.
- GitHub Pages serves `site/` (UI + JSON) from the **same origin**, avoiding CORS and avoiding exposing API keys.
- The browser runs the model engine using **Pyodide** (`site/py/*` + `site/static/app.js`).

### Enable Pages mode

1. In GitHub repo settings:
   - Pages → Build and deployment → Source: **GitHub Actions**
   - Actions → General → Workflow permissions: **Read and write permissions**

2. Trigger the workflow:
   - Actions → “Prefetch rate data for Pages” → Run workflow

3. Verify both workflows complete:
   - “Prefetch rate data for Pages”
   - “Deploy GitHub Pages”

4. Verify `site/data/` contains:
   - `treasury_par_history.json`
   - `ref_rates.json`
   - `models.json`
   - `status.json`

5. Open your Pages URL and click **Run**.

### Local preflight

No local prefetch step is required for GitHub Pages deployment. The scheduled/manual GitHub Action refreshes `site/data` and validates non-empty payloads before commit.

### Switch between API and Pages mode

- **Pages (default):** no query string needed
- **API mode:** append `?api=https://your-api-host` to the Pages URL
