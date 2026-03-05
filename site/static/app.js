/* Hybrid dashboard:
   - If ?api=<url> is present, uses remote FastAPI backend (same endpoints as before).
   - Otherwise (GitHub Pages mode), loads cached JSON from ./data and runs the model engine in-browser via Pyodide.
*/
const qs = new URLSearchParams(window.location.search);
const API_BASE = qs.get("api") || window.API_BASE || null;
const LOCAL_MODE = !API_BASE;

async function fetchJSON(url){
  const r = await fetch(url, {cache:"no-cache"});
  if(!r.ok) throw new Error(`Fetch failed ${r.status}: ${url}`);
  return await r.json();
}

function normalizeLocalModels(payload) {
  const models = Array.isArray(payload?.models) ? payload.models : [];
  const recommended = Array.isArray(payload?.recommended)
    ? payload.recommended
    : models.filter((m) => m?.recommended).map((m) => m.name);
  const weightingMethods = Array.isArray(payload?.weighting_methods)
    ? payload.weighting_methods
    : [
        { key: 'performance', label: 'Performance-based' },
        { key: 'curve_fit', label: 'More curve-fit emphasis' },
        { key: 'forecast', label: 'More forecast emphasis' },
      ];
  return { models, recommended, weighting_methods: weightingMethods };
}

function normalizeLocalStatus(payload, query = new URLSearchParams()) {
  const allowFredFallback = query.get('allow_fred_fallback') !== 'false';
  const cacheEnabled = query.get('cache_enabled') !== 'false';
  return {
    supported_sources: payload?.supported_sources || [
      { key: 'treasury', label: 'Treasury curve (official)' },
      { key: 'sofr', label: 'SOFR (official)' },
      { key: 'effr', label: 'Fed Funds (official)' },
      { key: 'auto', label: 'Auto (official then fallback)' },
    ],
    cache_dir: payload?.cache_dir || 'site/data',
    cache_enabled: cacheEnabled,
    allow_fred_fallback: allowFredFallback,
    cache_summary: payload?.cache_summary || {},
    recent_logs: Array.isArray(payload?.recent_logs) ? payload.recent_logs : [],
    last_updated: payload?.last_updated || null,
    latest_date: payload?.latest_date || null,
    fallback_used: Boolean(payload?.fallback_used),
    mode: payload?.mode || 'pages_static_json',
  };
}

// --- Pyodide local engine ---
let _pyodidePromise = null;
async function getPyodide(){
  if(_pyodidePromise) return _pyodidePromise;
  _pyodidePromise = (async ()=>{
    setStatus('Loading engine…', 'muted');
    const pyodideMod = await import("https://cdn.jsdelivr.net/pyodide/v0.25.1/full/pyodide.mjs");
    const pyodide = await pyodideMod.loadPyodide({ indexURL: "https://cdn.jsdelivr.net/pyodide/v0.25.1/full/" });
    await pyodide.loadPackage(["numpy", "scipy", "pandas"]);
    const manifest = await fetchJSON("./py/manifest.json");
    for(const f of manifest.files){
      const txt = await (await fetch(`./py/${f}`)).text();
      const path = `/py/${f}`;
      const dir = path.split("/").slice(0,-1).join("/");
      try { pyodide.FS.mkdirTree(dir); } catch(e){}
      pyodide.FS.writeFile(path, txt, {encoding:"utf8"});
    }
    pyodide.runPython(`
import sys
if "/py" not in sys.path:
    sys.path.insert(0, "/py")
`);
    setStatus('Engine ready', 'muted');
    return pyodide;
  })();
  return _pyodidePromise;
}

async function localRun(runPayload){
  let treasury;
  let refRates;
  let status;
  try {
    [treasury, refRates, status] = await Promise.all([
      fetchJSON("./data/treasury_par_history.json"),
      fetchJSON("./data/ref_rates.json"),
      fetchJSON("./data/status.json"),
    ]);
  } catch (error) {
    throw new Error("Pages data files are missing. Run the GitHub Action 'Prefetch rate data for Pages' to regenerate site/data JSON.");
  }
  const pyodide = await getPyodide();
  pyodide.globals.set("PAYLOAD_JSON", JSON.stringify(runPayload));
  pyodide.globals.set("DATA_JSON", JSON.stringify({treasury_par_history: treasury, ref_rates: refRates, status}));
  let out;
  try {
    out = pyodide.runPython(`
import json
from browser_engine import run_local
res = run_local(json.loads(PAYLOAD_JSON), json.loads(DATA_JSON))
json.dumps(res)
`);
  } catch (error) {
    throw new Error("Pages data is empty or invalid. Re-run the GitHub Action 'Prefetch rate data for Pages' and refresh.");
  }
  const res = JSON.parse(out);
  return res;
}

async function localScenario(payload) {
  const pyodide = await getPyodide();
  pyodide.globals.set("SCENARIO_JSON", JSON.stringify(payload));
  let out;
  try {
    out = pyodide.runPython(`
import json
from browser_engine import run_scenario
res = run_scenario(json.loads(SCENARIO_JSON))
json.dumps(res)
`);
  } catch (error) {
    throw new Error("Local scenario engine failed. Refresh and run the dashboard again.");
  }
  return JSON.parse(out);
}

const state = {
  advancedMode: false,
  modelsMeta: null,
  sourceStatus: null,
  result: null,
  curveType: 'zero',
  curveSelection: { market: true, ensemble: true, models: false },
  advancedDrawerOpen: true,
};

const palette = ['#1b5cff', '#0f8a5f', '#ef6820', '#7a5af8', '#06aed5', '#f04438', '#5c6c80', '#ca8504'];

function $(id) {
  return document.getElementById(id);
}

function formatPct(value) {
  if (value === null || value === undefined || Number.isNaN(value)) return '-';
  return `${(value * 100).toFixed(2)}%`;
}

function formatBps(value) {
  if (value === null || value === undefined || Number.isNaN(value)) return '-';
  const sign = value > 0 ? '+' : '';
  return `${sign}${value.toFixed(1)} bp`;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#39;');
}

async function apiFetch(path, options = {}) {
  if (!LOCAL_MODE) {
    const url = `${API_BASE.replace(/\/$/, '')}${path}`;
    const response = await fetch(url, {
      headers: { 'Content-Type': 'application/json', ...(options.headers || {}) },
      ...options,
    });
    const payload = await response.text();
    if (!response.ok) {
      throw new Error(payload || `Request failed (${response.status})`);
    }
    try {
      return payload ? JSON.parse(payload) : null;
    } catch (error) {
      return payload;
    }
  }

  // --- Pages local mode shim ---
  if (path === '/models' && (!options.method || options.method === 'GET')) {
    return normalizeLocalModels(await fetchJSON('./data/models.json'));
  }
  if (path.startsWith('/sources/status') && (!options.method || options.method === 'GET')) {
    const query = new URLSearchParams(path.split('?')[1] || '');
    return normalizeLocalStatus(await fetchJSON('./data/status.json'), query);
  }
  if (path === '/cache/clear') {
    // No server-side cache to clear in Pages mode
    return { ok: true, mode: 'pages_static_json' };
  }
  if (path === '/run' && (options.method || 'GET') === 'POST') {
    const body = options.body ? JSON.parse(options.body) : {};
    return await localRun(body);
  }
  if (path === '/models/run_one' && (options.method || 'GET') === 'POST') {
    const body = options.body ? JSON.parse(options.body) : {};
    const runPayload = { ...body };
    delete runPayload.model_name;
    if (body.model_name) {
      runPayload.selected_models = [body.model_name];
    }
    return await localRun(runPayload);
  }
  if (path === '/scenario' && (options.method || 'GET') === 'POST') {
    const body = options.body ? JSON.parse(options.body) : {};
    return await localScenario(body);
  }

  throw new Error(`Endpoint not available in Pages mode: ${path}`);
}

function todayIso() {
  return new Date().toISOString().slice(0, 10);
}

function parseQuery() {
  const params = new URLSearchParams(window.location.search);
  const setIf = (id, name = id) => {
    if (params.has(name)) $(id).value = params.get(name);
  };
  setIf('dataSource', 'data_source');
  setIf('asOfDate', 'as_of_date');
  setIf('horizon');
  setIf('target');
  setIf('historyYears', 'history_years');
  setIf('shortRateTenor', 'short_rate_tenor');
  setIf('weightingMethod', 'weighting_method');
  setIf('optimizationMode', 'optimization_mode');
  setIf('mcPaths', 'mc_paths');
  setIf('randomSeed', 'random_seed');
  if (params.get('mode') === 'advanced') {
    $('modeToggle').checked = true;
    state.advancedMode = true;
  }
  if (params.has('allow_fred_fallback')) {
    const checked = params.get('allow_fred_fallback') !== 'false';
    $('allowFredFallback').checked = checked;
    $('allowFredFallbackCacheTab').checked = checked;
  }
  if (params.has('cache_enabled')) {
    $('cacheEnabled').checked = params.get('cache_enabled') !== 'false';
  }
}

function buildShareUrl() {
  const params = new URLSearchParams();
  if (API_BASE) params.set('api', API_BASE);
  params.set('data_source', $('dataSource').value);
  if ($('asOfDate').value) params.set('as_of_date', $('asOfDate').value);
  params.set('horizon', $('horizon').value);
  params.set('target', $('target').value);
  params.set('history_years', $('historyYears').value);
  params.set('short_rate_tenor', $('shortRateTenor').value);
  params.set('weighting_method', $('weightingMethod').value);
  params.set('optimization_mode', $('optimizationMode').value);
  if ($('mcPaths').value) params.set('mc_paths', $('mcPaths').value);
  if ($('randomSeed').value) params.set('random_seed', $('randomSeed').value);
  params.set('allow_fred_fallback', $('allowFredFallback').checked ? 'true' : 'false');
  params.set('cache_enabled', $('cacheEnabled').checked ? 'true' : 'false');
  params.set('mode', state.advancedMode ? 'advanced' : 'simple');
  const selectedModels = getSelectedModels();
  if (selectedModels.length) params.set('selected_models', selectedModels.join(','));
  return `${window.location.origin}${window.location.pathname}?${params.toString()}`;
}

function syncUrl() {
  window.history.replaceState({}, '', buildShareUrl());
}

function setStatus(text, kind = 'muted') {
  const pill = $('statusPill');
  pill.textContent = text;
  pill.className = `status-pill ${kind === 'muted' ? 'muted' : ''}`;
}

function getSelectedModels() {
  return Array.from(document.querySelectorAll('#modelChecklist input[type="checkbox"]:checked')).map((el) => el.value);
}

function gatherRunPayload(forceRefresh = false) {
  return {
    data_source: $('dataSource').value,
    as_of_date: $('asOfDate').value || null,
    horizon: $('horizon').value,
    target: $('target').value,
    history_years: Number($('historyYears').value),
    short_rate_tenor: $('shortRateTenor').value,
    allow_fred_fallback: $('allowFredFallback').checked,
    cache_enabled: $('cacheEnabled').checked,
    force_refresh: forceRefresh,
    weighting_method: $('weightingMethod').value,
    optimization_mode: $('optimizationMode').value,
    mc_paths: $('mcPaths').value ? Number($('mcPaths').value) : null,
    random_seed: $('randomSeed').value ? Number($('randomSeed').value) : null,
    selected_models: getSelectedModels(),
  };
}

function renderModelChecklist(meta) {
  const container = $('modelChecklist');
  container.innerHTML = '';
  const recommended = new Set(meta.recommended || []);
  meta.models.forEach((model) => {
    const row = document.createElement('label');
    row.className = 'checkbox-row';
    row.innerHTML = `
      <input type="checkbox" value="${escapeHtml(model.name)}" ${recommended.has(model.name) ? 'checked' : ''} />
      <span><strong>${escapeHtml(model.label)}</strong> - ${escapeHtml(model.short_description)}</span>
    `;
    row.querySelector('input').addEventListener('change', syncUrl);
    container.appendChild(row);
  });
}

function setAdvancedMode(enabled) {
  state.advancedMode = enabled;
  document.querySelectorAll('.advanced-only').forEach((el) => {
    el.classList.toggle('hidden', !enabled);
  });
  $('advancedDrawer').classList.toggle('hidden', !enabled);
  if (!enabled) {
    activateTab('overview');
  }
  syncUrl();
}

function activateTab(tabName) {
  document.querySelectorAll('.tab-button').forEach((button) => {
    button.classList.toggle('active', button.dataset.tab === tabName);
  });
  document.querySelectorAll('.tab-panel').forEach((panel) => {
    panel.classList.toggle('active', panel.id === `tab-${tabName}`);
  });
}

function renderTopSummary(result) {
  const overview = result.overview;
  $('predictionCardValue').textContent = overview.predicted_level == null ? 'Curve forecast ready' : formatPct(overview.predicted_level);
  $('predictionCardChange').textContent = overview.change_bps == null ? 'Current and predicted curves are shown below.' : `${formatBps(overview.change_bps)} vs current`;
  $('confidenceLabel').textContent = overview.confidence.label.toUpperCase();
  $('confidenceRange').textContent = overview.confidence.lower == null ? '-' : `${formatPct(overview.confidence.lower)} to ${formatPct(overview.confidence.upper)}`;
  $('dataHealthSource').textContent = overview.data_health.source_used;
  $('dataHealthInfo').textContent = `Last obs: ${overview.data_health.last_observation_date || '-'} | Missing days filled: ${overview.data_health.missing_days_filled ? 'yes' : 'no'}`;
  const drivers = $('topDrivers');
  drivers.innerHTML = '';
  overview.top_drivers.forEach((driver) => {
    const div = document.createElement('div');
    div.className = 'driver-card';
    div.innerHTML = `
      <strong>${escapeHtml(driver.label)} <span class="badge">${(driver.weight * 100).toFixed(1)}%</span></strong>
      <div>${escapeHtml(driver.description)}</div>
      <div class="metric-subtext">${escapeHtml(driver.note)}</div>
    `;
    drivers.appendChild(div);
  });
}

function renderOverviewChart(result) {
  const target = result.request.target;
  if (target === 'curve') {
    renderCurveChart($('overviewChart'), result.charts.current_curve, result.charts.predicted_curve, state.curveType, { market: true, ensemble: true, models: false }, {});
    return;
  }
  const history = result.charts.target_history || [];
  const forecast = result.charts.target_forecast;
  const series = [];
  if (history.length) {
    series.push({ name: 'History', color: palette[0], values: history.map((row) => ({ x: row.date, y: row.value })) });
  }
  if (forecast.value != null) {
    series.push({
      name: 'Forecast',
      color: palette[1],
      values: [{ x: forecast.date, y: forecast.value }],
      pointsOnly: true,
      band: forecast.lower != null ? [{ x: forecast.date, y0: forecast.lower, y1: forecast.upper }] : null,
    });
  }
  renderLineChart($('overviewChart'), series, { yFormatter: formatPct });
}

function renderCurveChart(container, market, ensemble, curveType, selection, modelCurves) {
  const series = [];
  if (selection.market) {
    series.push({ name: 'Market', color: palette[0], values: market.maturities.map((x, i) => ({ x, y: market[curveType][i] })) });
  }
  if (selection.ensemble) {
    series.push({ name: 'Ensemble', color: palette[1], values: ensemble.maturities.map((x, i) => ({ x, y: ensemble[curveType][i] })) });
  }
  if (selection.models) {
    Object.entries(modelCurves || {}).forEach(([name, bundle], idx) => {
      series.push({ name, color: palette[(idx + 2) % palette.length], values: bundle.maturities.map((x, i) => ({ x, y: bundle[curveType][i] })) });
    });
  }
  renderLineChart(container, series, { xFormatter: (value) => `${value}y`, yFormatter: formatPct, xType: 'numeric' });
}

function renderCurveExplorer(result) {
  const market = result.curve_explorer.market;
  const ensemble = result.curve_explorer.ensemble;
  const models = result.curve_explorer.models;
  const chips = $('tenorChips');
  chips.innerHTML = '';
  result.curve_explorer.tenor_labels.forEach((label) => {
    const chip = document.createElement('span');
    chip.className = 'tenor-chip';
    chip.textContent = label;
    chips.appendChild(chip);
  });
  renderCurveChart($('curveChart'), market, ensemble, state.curveType, state.curveSelection, models);
}

function renderWeightsTable(result) {
  const rows = result.diagnostics.ensemble_health.weights || [];
  $('weightsTable').innerHTML = makeTable(
    ['Model', 'Weight', 'Hist RMSE (bp)', 'Curve RMSE (bp)'],
    rows.map((row) => [row.model, `${(row.weight * 100).toFixed(1)}%`, (row.historical_rmse * 10000).toFixed(1), Number.isFinite(row.curve_rmse) ? (row.curve_rmse * 10000).toFixed(1) : '-'])
  );
}

function renderRollingErrorChart(result) {
  const rolling = result.diagnostics.ensemble_health.rolling_backtest_error || [];
  const series = [{
    name: 'Rolling RMSE',
    color: palette[2],
    values: rolling.map((row) => ({ x: row.date, y: row.rolling_rmse })),
  }];
  renderLineChart($('rollingErrorChart'), series, { yFormatter: (v) => `${(v * 10000).toFixed(1)} bp` });
}

function renderModelAccordions(result) {
  const container = $('modelAccordions');
  container.innerHTML = '';
  (result.diagnostics.per_model || []).forEach((model) => {
    const item = document.createElement('div');
    item.className = 'accordion-item';
    const paramHtml = Object.entries(model.parameters || {}).map(([k, v]) => `<div><strong>${escapeHtml(k)}:</strong> ${escapeHtml(v)}</div>`).join('');
    item.innerHTML = `
      <div class="accordion-header">
        <div>
          <strong>${escapeHtml(model.label)} <span class="badge">${(model.weight * 100).toFixed(1)}%</span></strong>
          <div class="metric-subtext">Curve fit: ${model.curve_fit_rmse_bps == null ? '-' : model.curve_fit_rmse_bps.toFixed(1) + ' bp'} | Forecast error: ${model.forecast_rmse_bps.toFixed(1)} bp</div>
        </div>
        <div class="metric-subtext">Details</div>
      </div>
      <div class="accordion-body">
        <div>${escapeHtml(model.notes)}</div>
        <div class="metric-subtext" style="margin-top:8px;">${paramHtml || 'No explicit scalar parameters exposed.'}</div>
      </div>
    `;
    item.querySelector('.accordion-header').addEventListener('click', () => item.classList.toggle('open'));
    container.appendChild(item);
  });
}

function renderDataCache(result) {
  $('cacheDir').textContent = result.data_cache.cache_dir;
  $('sourceStatus').textContent = result.status.source_used;
  $('fetchLogs').innerHTML = makeTable(
    ['Time', 'Message'],
    (result.data_cache.logs || []).map((row) => [row.timestamp || '-', row.message || '-'])
  );
}

function makeTable(headers, rows) {
  const thead = headers.map((h) => `<th>${escapeHtml(h)}</th>`).join('');
  const tbody = rows.map((row) => `<tr>${row.map((cell) => `<td>${escapeHtml(cell)}</td>`).join('')}</tr>`).join('');
  return `<table class="simple-table"><thead><tr>${thead}</tr></thead><tbody>${tbody}</tbody></table>`;
}

function renderScenarioTable(result) {
  $('scenarioTable').innerHTML = makeTable(
    ['Tenor', 'Delta (bp)'],
    (result.deltas_bps || []).map((row) => [String(row.maturity), row.delta_bps.toFixed(1)])
  );
}

function renderScenarioChart(result) {
  const base = result.base;
  const shocked = result.shocked;
  const curveType = state.curveType;
  const series = [
    { name: 'Base', color: palette[0], values: base.maturities.map((x, i) => ({ x, y: base[curveType][i] })) },
    { name: 'Shocked', color: palette[5], values: shocked.maturities.map((x, i) => ({ x, y: shocked[curveType][i] })) },
  ];
  renderLineChart($('scenarioChart'), series, { xFormatter: (value) => `${value}y`, yFormatter: formatPct, xType: 'numeric' });
}

function renderLineChart(container, series, options = {}) {
  if (!series.length || !series.some((s) => s.values.length)) {
    container.innerHTML = '<div class="metric-subtext">No data to display.</div>';
    return;
  }
  const width = container.clientWidth || 700;
  const height = Math.max(260, container.clientHeight || 320);
  const margin = { top: 20, right: 20, bottom: 40, left: 68 };
  const allPoints = [];
  series.forEach((s) => {
    s.values.forEach((p) => allPoints.push({ x: normalizeX(p.x, options.xType), y: p.y }));
    if (s.band) {
      s.band.forEach((b) => {
        allPoints.push({ x: normalizeX(b.x, options.xType), y: b.y0 });
        allPoints.push({ x: normalizeX(b.x, options.xType), y: b.y1 });
      });
    }
  });
  const xValues = allPoints.map((p) => p.x);
  const yValues = allPoints.map((p) => p.y);
  const xMin = Math.min(...xValues);
  const xMax = Math.max(...xValues);
  const yMinRaw = Math.min(...yValues);
  const yMaxRaw = Math.max(...yValues);
  const yPad = (yMaxRaw - yMinRaw || 0.002) * 0.12;
  const yMin = yMinRaw - yPad;
  const yMax = yMaxRaw + yPad;
  const xScale = (x) => margin.left + ((x - xMin) / Math.max(xMax - xMin || 1, 1e-9)) * (width - margin.left - margin.right);
  const yScale = (y) => height - margin.bottom - ((y - yMin) / Math.max(yMax - yMin || 1, 1e-9)) * (height - margin.top - margin.bottom);

  const axisTicks = 4;
  const yTicks = Array.from({ length: axisTicks + 1 }, (_, i) => yMin + (i / axisTicks) * (yMax - yMin));
  const xTicks = Array.from({ length: axisTicks + 1 }, (_, i) => xMin + (i / axisTicks) * (xMax - xMin));

  const legendHtml = `<div class="legend-row">${series.map((s) => `<span class="legend-item"><span class="legend-swatch" style="background:${s.color}"></span>${escapeHtml(s.name)}</span>`).join('')}</div>`;
  let svg = `<svg class="svg-chart" viewBox="0 0 ${width} ${height}" preserveAspectRatio="none">`;
  svg += `<rect x="0" y="0" width="${width}" height="${height}" fill="white" rx="12"></rect>`;

  yTicks.forEach((tick) => {
    svg += `<line x1="${margin.left}" y1="${yScale(tick)}" x2="${width - margin.right}" y2="${yScale(tick)}" stroke="#eef2f7"></line>`;
    svg += `<text x="${margin.left - 10}" y="${yScale(tick) + 4}" text-anchor="end" font-size="11" fill="#5c6c80">${escapeHtml(options.yFormatter ? options.yFormatter(tick) : tick.toFixed(2))}</text>`;
  });

  xTicks.forEach((tick) => {
    svg += `<line x1="${xScale(tick)}" y1="${margin.top}" x2="${xScale(tick)}" y2="${height - margin.bottom}" stroke="#f5f7fa"></line>`;
    svg += `<text x="${xScale(tick)}" y="${height - 12}" text-anchor="middle" font-size="11" fill="#5c6c80">${escapeHtml(options.xFormatter ? options.xFormatter(denormalizeX(tick, options.xType)) : formatXTick(tick, options.xType))}</text>`;
  });

  svg += `<line x1="${margin.left}" y1="${height - margin.bottom}" x2="${width - margin.right}" y2="${height - margin.bottom}" stroke="#cbd5e1"></line>`;
  svg += `<line x1="${margin.left}" y1="${margin.top}" x2="${margin.left}" y2="${height - margin.bottom}" stroke="#cbd5e1"></line>`;

  series.forEach((s) => {
    if (s.band) {
      const points = s.band.map((b) => `${xScale(normalizeX(b.x, options.xType))},${yScale(b.y0)}`).join(' ');
      const reverse = [...s.band].reverse().map((b) => `${xScale(normalizeX(b.x, options.xType))},${yScale(b.y1)}`).join(' ');
      svg += `<polygon points="${points} ${reverse}" fill="${hexToRgba(s.color, 0.15)}"></polygon>`;
    }
    const path = s.values.map((p, idx) => `${idx === 0 ? 'M' : 'L'} ${xScale(normalizeX(p.x, options.xType))} ${yScale(p.y)}`).join(' ');
    if (!s.pointsOnly && s.values.length > 1) {
      svg += `<path d="${path}" fill="none" stroke="${s.color}" stroke-width="2.5"></path>`;
    }
    s.values.forEach((p) => {
      svg += `<circle cx="${xScale(normalizeX(p.x, options.xType))}" cy="${yScale(p.y)}" r="${s.pointsOnly ? 4.5 : 3}" fill="${s.color}"></circle>`;
    });
  });

  svg += '</svg>';
  container.innerHTML = `${legendHtml}${svg}`;
}

function formatXTick(value, type) {
  if (type === 'numeric') return Number(value).toFixed(1);
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? String(value) : date.toISOString().slice(0, 10);
}

function normalizeX(value, type) {
  if (type === 'numeric') return Number(value);
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? Number(value) : date.getTime();
}

function denormalizeX(value, type) {
  if (type === 'numeric') return value;
  return new Date(value).toISOString().slice(0, 10);
}

function hexToRgba(hex, alpha) {
  const normalized = hex.replace('#', '');
  const bigint = Number.parseInt(normalized.length === 3 ? normalized.split('').map((c) => c + c).join('') : normalized, 16);
  const r = (bigint >> 16) & 255;
  const g = (bigint >> 8) & 255;
  const b = bigint & 255;
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

async function runDashboard(forceRefresh = false) {
  setStatus('Running...', 'muted');
  syncUrl();
  try {
    const result = await apiFetch('/run', { method: 'POST', body: JSON.stringify(gatherRunPayload(forceRefresh)) });
    state.result = result;
    setStatus(result.status.pill, 'ok');
    renderResult();
  } catch (error) {
    setStatus('Run failed', 'muted');
    alert(error.message);
  }
}

function renderResult() {
  if (!state.result) return;
  renderTopSummary(state.result);
  renderOverviewChart(state.result);
  renderCurveExplorer(state.result);
  renderWeightsTable(state.result);
  renderRollingErrorChart(state.result);
  renderModelAccordions(state.result);
  renderDataCache(state.result);
}

async function refreshSourcesStatus() {
  try {
    state.sourceStatus = await apiFetch(`/sources/status?allow_fred_fallback=${$('allowFredFallback').checked}&cache_enabled=${$('cacheEnabled').checked}`);
    if (!state.result) {
      $('cacheDir').textContent = state.sourceStatus.cache_dir;
      $('sourceStatus').textContent = 'Ready';
      $('fetchLogs').innerHTML = makeTable(['Time', 'Message'], (state.sourceStatus.recent_logs || []).map((row) => [row.timestamp || '-', row.message || '-']));
    }
  } catch (error) {
    console.error(error);
  }
}

async function applyScenario() {
  if (!state.result) {
    alert('Run the dashboard first.');
    return;
  }
  const selected = document.querySelector('input[name="scenarioPreset"]:checked').value;
  const payload = {
    maturities: state.result.charts.predicted_curve.maturities,
    zero_curve: state.result.charts.predicted_curve.zero,
    scenario_type: selected,
    short_end_bps: Number($('shortShock').value),
    long_end_bps: Number($('longShock').value),
  };
  try {
    const result = await apiFetch('/scenario', { method: 'POST', body: JSON.stringify(payload) });
    renderScenarioChart(result);
    renderScenarioTable(result);
  } catch (error) {
    alert(error.message);
  }
}

function downloadBlob(filename, text, mimeType = 'text/plain') {
  const blob = new Blob([text], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

function handleDownload(kind) {
  if (!state.result) {
    alert('Run the dashboard first.');
    return;
  }
  if (kind === 'predictions.csv') {
    const rows = [
      ['target', state.result.overview.target_label],
      ['current_level', state.result.overview.current_level],
      ['predicted_level', state.result.overview.predicted_level],
      ['change_bps', state.result.overview.change_bps],
      ['confidence_label', state.result.overview.confidence.label],
      ['confidence_lower', state.result.overview.confidence.lower],
      ['confidence_upper', state.result.overview.confidence.upper],
    ];
    downloadBlob(kind, rows.map((row) => row.join(',')).join('\n'), 'text/csv');
    return;
  }
  if (kind === 'curve_points.csv') {
    const curve = state.result.charts.predicted_curve;
    const current = state.result.charts.current_curve;
    const header = ['maturity', 'current_zero', 'predicted_zero', 'current_par', 'predicted_par', 'current_forward', 'predicted_forward'];
    const rows = curve.maturities.map((m, i) => [m, current.zero[i], curve.zero[i], current.par[i], curve.par[i], current.forward[i], curve.forward[i]]);
    downloadBlob(kind, [header.join(','), ...rows.map((row) => row.join(','))].join('\n'), 'text/csv');
    return;
  }
  downloadBlob(kind, JSON.stringify(state.result.diagnostics, null, 2), 'application/json');
}

function wireEvents() {
  $('runButton').addEventListener('click', () => runDashboard(false));
  $('refreshNow').addEventListener('click', () => runDashboard(true));
  $('modeToggle').addEventListener('change', (event) => setAdvancedMode(event.target.checked));
  $('advancedDrawerToggle').addEventListener('click', () => {
    state.advancedDrawerOpen = !state.advancedDrawerOpen;
    $('advancedDrawerBody').classList.toggle('hidden', !state.advancedDrawerOpen);
    $('advancedDrawerToggle').textContent = state.advancedDrawerOpen ? 'Collapse' : 'Expand';
  });
  $('shareLinkButton').addEventListener('click', async () => {
    const url = buildShareUrl();
    await navigator.clipboard.writeText(url);
    setStatus('Share link copied', 'ok');
  });
  $('downloadToggle').addEventListener('click', () => $('downloadMenu').classList.toggle('hidden'));
  document.querySelectorAll('#downloadMenu button').forEach((button) => {
    button.addEventListener('click', () => {
      handleDownload(button.dataset.download);
      $('downloadMenu').classList.add('hidden');
    });
  });
  document.querySelectorAll('.tab-button').forEach((button) => {
    button.addEventListener('click', () => activateTab(button.dataset.tab));
  });
  document.querySelectorAll('#curveTypeSwitch button').forEach((button) => {
    button.addEventListener('click', () => {
      state.curveType = button.dataset.curveType;
      document.querySelectorAll('#curveTypeSwitch button').forEach((b) => b.classList.toggle('active', b === button));
      if (state.result) {
        renderCurveExplorer(state.result);
        renderOverviewChart(state.result);
      }
    });
  });
  $('showMarketCurve').addEventListener('change', (e) => { state.curveSelection.market = e.target.checked; if (state.result) renderCurveExplorer(state.result); });
  $('showEnsembleCurve').addEventListener('change', (e) => { state.curveSelection.ensemble = e.target.checked; if (state.result) renderCurveExplorer(state.result); });
  $('showModelCurves').addEventListener('change', (e) => { state.curveSelection.models = e.target.checked; if (state.result) renderCurveExplorer(state.result); });
  document.querySelectorAll('input[name="scenarioPreset"]').forEach((input) => input.addEventListener('change', applyScenario));
  $('shortShock').addEventListener('input', (e) => { $('shortShockValue').textContent = e.target.value; if (document.querySelector('input[name="scenarioPreset"]:checked').value === 'custom') applyScenario(); });
  $('longShock').addEventListener('input', (e) => { $('longShockValue').textContent = e.target.value; if (document.querySelector('input[name="scenarioPreset"]:checked').value === 'custom') applyScenario(); });
  $('applyScenario').addEventListener('click', applyScenario);
  $('clearCache').addEventListener('click', async () => {
    if (!confirm('Clear the local cache?')) return;
    await apiFetch('/cache/clear', { method: 'POST', body: JSON.stringify({}) });
    await refreshSourcesStatus();
    if (state.result) await runDashboard(false);
  });
  $('allowFredFallback').addEventListener('change', (e) => { $('allowFredFallbackCacheTab').checked = e.target.checked; syncUrl(); });
  $('allowFredFallbackCacheTab').addEventListener('change', (e) => { $('allowFredFallback').checked = e.target.checked; syncUrl(); refreshSourcesStatus(); });
  ['dataSource', 'asOfDate', 'horizon', 'target', 'historyYears', 'shortRateTenor', 'weightingMethod', 'optimizationMode', 'mcPaths', 'randomSeed', 'cacheEnabled'].forEach((id) => {
    $(id).addEventListener('change', syncUrl);
  });
}

async function initialize() {
  $('asOfDate').value = todayIso();
  parseQuery();
  wireEvents();
  state.modelsMeta = await apiFetch('/models');
  renderModelChecklist(state.modelsMeta);
  if (new URLSearchParams(window.location.search).has('selected_models')) {
    const selected = new Set(new URLSearchParams(window.location.search).get('selected_models').split(','));
    document.querySelectorAll('#modelChecklist input[type="checkbox"]').forEach((el) => { el.checked = selected.has(el.value); });
  }
  setAdvancedMode($('modeToggle').checked);
  await refreshSourcesStatus();
  if (window.location.search) {
    runDashboard(false);
  }
}

initialize().catch((error) => {
  console.error(error);
  alert(error.message);
});
