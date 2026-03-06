const state = {
  advancedMode: false,
  modelsMeta: null,
  sourceStatus: null,
  result: null,
  curveType: 'zero',
  curveSelection: { market: true, ensemble: true, models: false },
  selectedTenorIndex: null,
  running: false,
  runCounter: 0,
  pendingRun: null,
  autoRunTimer: null,
  advancedDrawerOpen: true,
  runProgressTimer: null,
  runProgressHideTimer: null,
  runProgressValue: 0,
};

const palette = ['#1b5cff', '#0f8a5f', '#ef6820', '#7a5af8', '#06aed5', '#f04438', '#5c6c80', '#ca8504'];
const HORIZON_LABELS = { '1d': '1D', '1w': '1W', '1m': '1M', '3m': '3M', '1y': '1Y' };

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

function formatMaturityTick(value) {
  const num = Number(value);
  if (!Number.isFinite(num)) return String(value);
  if (num < 1.0) {
    const months = Math.round(num * 12);
    return `${months}M`;
  }
  if (Math.abs(num - Math.round(num)) < 1e-9) {
    return `${Math.round(num)}Y`;
  }
  return `${num.toFixed(1)}Y`;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#39;');
}

async function apiFetch(url, options = {}) {
  const response = await fetch(url, {
    headers: { 'Content-Type': 'application/json', ...(options.headers || {}) },
    ...options,
  });
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.detail || 'Request failed');
  }
  return payload;
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

function setRunControlsDisabled(disabled) {
  const runButton = $('runButton');
  if (runButton) {
    runButton.disabled = disabled;
    runButton.textContent = disabled ? 'Running...' : 'Run';
  }
  const refreshButton = $('refreshNow');
  if (refreshButton) {
    refreshButton.disabled = disabled;
  }
}

function startRunProgress() {
  const shell = $('rerunProgress');
  const bar = $('rerunProgressBar');
  if (!shell || !bar) return;

  const wasFinishing = state.runProgressHideTimer !== null;
  if (state.runProgressHideTimer !== null) {
    clearTimeout(state.runProgressHideTimer);
    state.runProgressHideTimer = null;
  }

  if (!shell.classList.contains('visible') || wasFinishing || state.runProgressValue >= 95) {
    state.runProgressValue = 10;
    bar.style.width = `${state.runProgressValue}%`;
    shell.classList.remove('hidden');
    shell.classList.add('visible');
    shell.setAttribute('aria-hidden', 'false');
  } else {
    state.runProgressValue = Math.max(state.runProgressValue, 18);
    bar.style.width = `${state.runProgressValue}%`;
  }

  if (state.runProgressTimer !== null) return;
  state.runProgressTimer = window.setInterval(() => {
    const remaining = 94 - state.runProgressValue;
    if (remaining <= 0) {
      clearInterval(state.runProgressTimer);
      state.runProgressTimer = null;
      return;
    }
    const increment = state.runProgressValue < 40 ? 7 : state.runProgressValue < 70 ? 3 : 1;
    state.runProgressValue = Math.min(94, state.runProgressValue + increment);
    bar.style.width = `${state.runProgressValue}%`;
  }, 170);
}

function finishRunProgress() {
  const shell = $('rerunProgress');
  const bar = $('rerunProgressBar');
  if (!shell || !bar) return;

  if (state.runProgressTimer !== null) {
    clearInterval(state.runProgressTimer);
    state.runProgressTimer = null;
  }
  if (state.runProgressHideTimer !== null) {
    clearTimeout(state.runProgressHideTimer);
    state.runProgressHideTimer = null;
  }

  state.runProgressValue = 100;
  bar.style.width = '100%';
  state.runProgressHideTimer = window.setTimeout(() => {
    shell.classList.remove('visible');
    shell.classList.add('hidden');
    shell.setAttribute('aria-hidden', 'true');
    bar.style.width = '0%';
    state.runProgressValue = 0;
    state.runProgressHideTimer = null;
  }, 220);
}

function renderHorizonQuickButtons() {
  const container = $('horizonQuickButtons');
  if (!container) return;
  const current = $('horizon').value;
  container.innerHTML = '';
  Object.entries(HORIZON_LABELS).forEach(([value, label]) => {
    const button = document.createElement('button');
    button.type = 'button';
    button.textContent = label;
    button.classList.toggle('active', value === current);
    button.addEventListener('click', () => {
      if ($('horizon').value === value && state.result) return;
      $('horizon').value = value;
      renderHorizonQuickButtons();
      scheduleAutoRun('horizon');
    });
    container.appendChild(button);
  });
}

function scheduleAutoRun(reason = 'control') {
  syncUrl();
  if (state.autoRunTimer !== null) {
    clearTimeout(state.autoRunTimer);
    state.autoRunTimer = null;
  }
  state.autoRunTimer = window.setTimeout(() => {
    state.autoRunTimer = null;
    runDashboard(false, reason);
  }, 220);
}

function queuePendingRun(forceRefresh, reason) {
  const nextForceRefresh = Boolean(forceRefresh);
  if (!state.pendingRun) {
    state.pendingRun = { forceRefresh: nextForceRefresh, reason };
    return;
  }
  const mergedForceRefresh = Boolean(state.pendingRun.forceRefresh || nextForceRefresh);
  let mergedReason = reason || state.pendingRun.reason;
  if (mergedForceRefresh && (state.pendingRun.reason === 'refresh' || reason === 'refresh')) {
    mergedReason = 'refresh';
  }
  state.pendingRun = { forceRefresh: mergedForceRefresh, reason: mergedReason };
}

function getSelectedModels() {
  return Array.from(document.querySelectorAll('#modelChecklist input[type="checkbox"]:checked')).map((el) => el.value);
}

function modelLabel(modelName) {
  const rows = state.modelsMeta?.models || [];
  const hit = rows.find((row) => row?.name === modelName);
  return hit?.label || modelName;
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
    row.querySelector('input').addEventListener('change', (event) => {
      if (!getSelectedModels().length) {
        event.target.checked = true;
        alert('At least one model must remain selected.');
        return;
      }
      scheduleAutoRun('models');
    });
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

function renderCurveChart(container, market, ensemble, curveType, selection, modelCurves, tenorHighlight = null) {
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
  renderLineChart(container, series, {
    xFormatter: formatMaturityTick,
    yFormatter: formatPct,
    xType: 'numeric',
    highlightX: tenorHighlight?.maturity ?? null,
    highlightLabel: tenorHighlight?.label ?? null,
  });
}

function renderCurveExplorer(result) {
  const market = result.curve_explorer.market;
  const ensemble = result.curve_explorer.ensemble;
  const models = result.curve_explorer.models;
  const chips = $('tenorChips');
  chips.innerHTML = '';
  result.curve_explorer.tenor_labels.forEach((label, idx) => {
    const chip = document.createElement('button');
    chip.type = 'button';
    chip.className = 'tenor-chip';
    chip.classList.toggle('active', state.selectedTenorIndex === idx);
    chip.textContent = label;
    chip.addEventListener('click', () => {
      state.selectedTenorIndex = state.selectedTenorIndex === idx ? null : idx;
      renderCurveExplorer(result);
    });
    chips.appendChild(chip);
  });

  let tenorHighlight = null;
  if (state.selectedTenorIndex != null && market?.maturities?.[state.selectedTenorIndex] != null) {
    tenorHighlight = {
      maturity: Number(market.maturities[state.selectedTenorIndex]),
      label: result.curve_explorer.tenor_labels[state.selectedTenorIndex],
    };
  }

  renderCurveChart($('curveChart'), market, ensemble, state.curveType, state.curveSelection, models, tenorHighlight);
}

function renderWeightsTable(result) {
  const rows = result.diagnostics.ensemble_health.weights || [];
  const histBps = rows
    .map((row) => Number(row.historical_rmse) * 10000)
    .filter((value) => Number.isFinite(value));
  const histSpread = histBps.length ? Math.max(...histBps) - Math.min(...histBps) : null;
  const tableHtml = makeTable(
    [
      { label: 'Model', help: 'Short-rate model included in the current ensemble run.' },
      { label: 'Weight', help: 'Current ensemble contribution after combining performance, curve fit, and model traits.' },
      { label: 'Backtest 1-step RMSE (bp)', help: 'One-step-ahead error on historical short-rate data. Lower is better.' },
      { label: 'As-of Curve RMSE (bp)', help: "Fit error to today's Treasury zero curve. Lower is better." },
    ],
    rows.map((row) => [
      modelLabel(row.model),
      `${(row.weight * 100).toFixed(1)}%`,
      Number.isFinite(row.historical_rmse) ? (row.historical_rmse * 10000).toFixed(2) : '-',
      Number.isFinite(row.curve_rmse) ? (row.curve_rmse * 10000).toFixed(2) : '-',
    ])
  );
  const spreadText = histSpread == null
    ? ''
    : `<div class="table-note">Backtest RMSE spread across models: ${histSpread.toFixed(2)} bp. If this spread is small, identical-looking values are usually a display-rounding effect.</div>`;
  $('weightsTable').innerHTML = `${tableHtml}${spreadText}`;
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
          <div class="metric-subtext">As-of curve RMSE: ${model.curve_fit_rmse_bps == null ? '-' : model.curve_fit_rmse_bps.toFixed(2) + ' bp'} | Backtest 1-step RMSE: ${model.forecast_rmse_bps.toFixed(2)} bp</div>
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
  const thead = headers.map((h) => {
    if (typeof h === 'string') {
      return `<th>${escapeHtml(h)}</th>`;
    }
    const label = escapeHtml(h?.label ?? '');
    const help = h?.help ? escapeHtml(h.help) : '';
    if (!help) {
      return `<th>${label}</th>`;
    }
    return `<th><span class="th-wrap">${label}<button type="button" class="help-tip" title="${help}" aria-label="${help}">i</button></span></th>`;
  }).join('');
  const tbody = rows.map((row) => `<tr>${row.map((cell) => `<td>${escapeHtml(cell)}</td>`).join('')}</tr>`).join('');
  return `<table class="simple-table"><thead><tr>${thead}</tr></thead><tbody>${tbody}</tbody></table>`;
}

function renderScenarioTable(result) {
  $('scenarioTable').innerHTML = makeTable(
    ['Tenor', 'Delta (bp)'],
    (result.deltas_bps || []).map((row) => [formatMaturityTick(row.maturity), row.delta_bps.toFixed(1)])
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
  renderLineChart($('scenarioChart'), series, { xFormatter: formatMaturityTick, yFormatter: formatPct, xType: 'numeric' });
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
  const normalizedSeries = series.map((s) => ({
    name: s.name,
    color: s.color,
    points: s.values
      .map((p) => ({ x: normalizeX(p.x, options.xType), y: Number(p.y) }))
      .filter((p) => Number.isFinite(p.x) && Number.isFinite(p.y))
      .sort((a, b) => a.x - b.x),
  }));
  const xCandidates = Array.from(new Set(normalizedSeries.flatMap((s) => s.points.map((p) => p.x)))).sort((a, b) => a - b);

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

  if (Number.isFinite(options.highlightX)) {
    const hx = normalizeX(options.highlightX, options.xType);
    const xPos = xScale(hx);
    svg += `<line x1="${xPos}" y1="${margin.top}" x2="${xPos}" y2="${height - margin.bottom}" stroke="#344054" stroke-dasharray="4 4" stroke-width="1.2"></line>`;
    if (options.highlightLabel) {
      svg += `<text x="${xPos + 6}" y="${margin.top + 14}" font-size="11" fill="#344054">${escapeHtml(options.highlightLabel)}</text>`;
    }
    series.forEach((s) => {
      const hit = s.values.find((p) => Math.abs(normalizeX(p.x, options.xType) - hx) < 1e-9);
      if (!hit) return;
      svg += `<circle cx="${xScale(normalizeX(hit.x, options.xType))}" cy="${yScale(hit.y)}" r="5.5" fill="#ffffff" stroke="${s.color}" stroke-width="2.2"></circle>`;
    });
  }

  svg += '<g class="hover-layer"></g>';
  svg += '</svg>';
  container.innerHTML = `${legendHtml}${svg}`;

  const svgEl = container.querySelector('.svg-chart');
  const hoverLayer = container.querySelector('.hover-layer');
  if (!svgEl || !hoverLayer || !xCandidates.length) return;

  const nearestValue = (sortedValues, target) => {
    if (!sortedValues.length) return null;
    let lo = 0;
    let hi = sortedValues.length - 1;
    while (lo <= hi) {
      const mid = (lo + hi) >> 1;
      if (sortedValues[mid] < target) lo = mid + 1;
      else hi = mid - 1;
    }
    if (lo <= 0) return sortedValues[0];
    if (lo >= sortedValues.length) return sortedValues[sortedValues.length - 1];
    return Math.abs(sortedValues[lo] - target) < Math.abs(sortedValues[lo - 1] - target) ? sortedValues[lo] : sortedValues[lo - 1];
  };

  const interpolateYAtX = (points, targetX) => {
    if (!points.length) return null;
    if (targetX < points[0].x || targetX > points[points.length - 1].x) return null;
    let lo = 0;
    let hi = points.length - 1;
    while (lo <= hi) {
      const mid = (lo + hi) >> 1;
      const midX = points[mid].x;
      if (midX === targetX) return points[mid].y;
      if (midX < targetX) lo = mid + 1;
      else hi = mid - 1;
    }
    const right = Math.min(lo, points.length - 1);
    const left = Math.max(0, right - 1);
    const p0 = points[left];
    const p1 = points[right];
    if (!p0 || !p1) return null;
    if (p1.x === p0.x) return p1.y;
    const t = (targetX - p0.x) / (p1.x - p0.x);
    return p0.y + t * (p1.y - p0.y);
  };

  const xLabelFor = (normalizedX) => {
    const denorm = denormalizeX(normalizedX, options.xType);
    return options.xFormatter ? options.xFormatter(denorm) : formatXTick(normalizedX, options.xType);
  };

  const clearHover = () => {
    hoverLayer.innerHTML = '';
  };

  const renderHoverAtClientX = (clientX) => {
    const rect = svgEl.getBoundingClientRect();
    if (!rect.width) return;
    const localX = ((clientX - rect.left) / rect.width) * width;
    if (!Number.isFinite(localX)) return;
    const plotLeft = margin.left;
    const plotRight = width - margin.right;
    if (localX < plotLeft || localX > plotRight) {
      clearHover();
      return;
    }

    const unsnapped = xMin + ((localX - plotLeft) / Math.max(plotRight - plotLeft, 1e-9)) * (xMax - xMin);
    const snappedX = nearestValue(xCandidates, unsnapped);
    if (!Number.isFinite(snappedX)) {
      clearHover();
      return;
    }

    const hoverPoints = normalizedSeries
      .map((s) => {
        const y = interpolateYAtX(s.points, snappedX);
        if (!Number.isFinite(y)) return null;
        return { name: s.name, color: s.color, y, screenY: yScale(y) };
      })
      .filter(Boolean)
      .sort((a, b) => a.screenY - b.screenY);

    if (!hoverPoints.length) {
      clearHover();
      return;
    }

    const clusters = [];
    const clusterThresholdPx = 14;
    hoverPoints.forEach((point) => {
      if (!clusters.length) {
        clusters.push({ points: [point], centerY: point.screenY });
        return;
      }
      const last = clusters[clusters.length - 1];
      if (Math.abs(point.screenY - last.centerY) <= clusterThresholdPx) {
        last.points.push(point);
        last.centerY = last.points.reduce((sum, p) => sum + p.screenY, 0) / last.points.length;
      } else {
        clusters.push({ points: [point], centerY: point.screenY });
      }
    });

    const xPos = xScale(snappedX);
    const preferRight = xPos < width * 0.62;
    const topBound = margin.top + 2;
    const bottomBound = height - margin.bottom - 2;
    const labels = clusters.map((cluster) => {
      const header = String(xLabelFor(snappedX));
      const lines = cluster.points.map((p) => ({
        text: `${p.name}: ${options.yFormatter ? options.yFormatter(p.y) : p.y.toFixed(4)}`,
        color: p.color,
      }));
      const maxChars = Math.max(header.length, ...lines.map((l) => l.text.length));
      const boxWidth = Math.min(290, Math.max(140, 16 + maxChars * 6.6));
      const boxHeight = 12 + 14 + lines.length * 14;
      const boxX = preferRight
        ? Math.min(width - margin.right - boxWidth - 4, xPos + 10)
        : Math.max(margin.left + 4, xPos - boxWidth - 10);
      const rawY = cluster.centerY - boxHeight / 2;
      return {
        x: boxX,
        y: Math.max(topBound, Math.min(bottomBound - boxHeight, rawY)),
        w: boxWidth,
        h: boxHeight,
        header,
        lines,
      };
    }).sort((a, b) => a.y - b.y);

    for (let i = 1; i < labels.length; i += 1) {
      labels[i].y = Math.max(labels[i].y, labels[i - 1].y + labels[i - 1].h + 4);
    }
    if (labels.length) {
      const overflow = labels[labels.length - 1].y + labels[labels.length - 1].h - bottomBound;
      if (overflow > 0) {
        labels.forEach((label) => {
          label.y -= overflow;
        });
      }
      if (labels[0].y < topBound) {
        const underflow = topBound - labels[0].y;
        labels.forEach((label) => {
          label.y += underflow;
        });
      }
    }

    let hoverSvg = '';
    hoverSvg += `<line x1="${xPos}" y1="${margin.top}" x2="${xPos}" y2="${height - margin.bottom}" stroke="#475467" stroke-dasharray="3 4" stroke-width="1.2"></line>`;
    hoverPoints.forEach((point) => {
      hoverSvg += `<circle cx="${xPos}" cy="${point.screenY}" r="4.2" fill="#ffffff" stroke="${point.color}" stroke-width="2"></circle>`;
    });
    labels.forEach((label) => {
      hoverSvg += `<rect x="${label.x}" y="${label.y}" width="${label.w}" height="${label.h}" rx="8" fill="rgba(255,255,255,0.96)" stroke="#d8e0ea"></rect>`;
      let textY = label.y + 14;
      hoverSvg += `<text x="${label.x + 8}" y="${textY}" font-size="11" font-weight="600" fill="#344054">${escapeHtml(label.header)}</text>`;
      label.lines.forEach((line) => {
        textY += 14;
        hoverSvg += `<text x="${label.x + 8}" y="${textY}" font-size="11" fill="${line.color}">${escapeHtml(line.text)}</text>`;
      });
    });
    hoverLayer.innerHTML = hoverSvg;
  };

  svgEl.addEventListener('mousemove', (event) => renderHoverAtClientX(event.clientX));
  svgEl.addEventListener('mouseleave', clearHover);
  svgEl.addEventListener('touchstart', (event) => {
    if (!event.touches || !event.touches.length) return;
    renderHoverAtClientX(event.touches[0].clientX);
  }, { passive: true });
  svgEl.addEventListener('touchmove', (event) => {
    if (!event.touches || !event.touches.length) return;
    renderHoverAtClientX(event.touches[0].clientX);
  }, { passive: true });
  svgEl.addEventListener('touchend', clearHover);
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

async function runDashboard(forceRefresh = false, reason = 'manual') {
  if (state.autoRunTimer !== null) {
    clearTimeout(state.autoRunTimer);
    state.autoRunTimer = null;
  }
  if (state.running) {
    queuePendingRun(forceRefresh, reason);
    return;
  }
  if (!getSelectedModels().length) {
    setStatus('Select at least one model', 'muted');
    alert('Select at least one model before running.');
    return;
  }
  const runId = ++state.runCounter;
  state.running = true;
  setRunControlsDisabled(true);
  startRunProgress();
  const horizonLabel = HORIZON_LABELS[$('horizon').value] || $('horizon').value;
  const statusByReason = {
    horizon: `Running ${horizonLabel} forecast...`,
    models: 'Updating model weights...',
    refresh: 'Refreshing data and rerunning...',
    startup: 'Running initial forecast...',
    manual: 'Running...',
  };
  setStatus(statusByReason[reason] || 'Running...', 'muted');
  syncUrl();
  try {
    const result = await apiFetch('/run', { method: 'POST', body: JSON.stringify(gatherRunPayload(forceRefresh)) });
    if (runId !== state.runCounter) return;
    state.result = result;
    setStatus(result.status.pill, 'ok');
    renderResult();
  } catch (error) {
    if (runId !== state.runCounter) return;
    setStatus('Run failed', 'muted');
    alert(error.message);
  } finally {
    if (runId === state.runCounter) {
      state.running = false;
      setRunControlsDisabled(false);
      renderHorizonQuickButtons();
      if (state.pendingRun) {
        const nextRun = state.pendingRun;
        state.pendingRun = null;
        runDashboard(nextRun.forceRefresh, nextRun.reason);
      } else {
        finishRunProgress();
      }
    }
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
  $('runButton').addEventListener('click', () => {
    if (state.autoRunTimer !== null) {
      clearTimeout(state.autoRunTimer);
      state.autoRunTimer = null;
    }
    runDashboard(false, 'manual');
  });
  $('refreshNow').addEventListener('click', () => runDashboard(true, 'refresh'));
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
  $('horizon').addEventListener('change', () => {
    renderHorizonQuickButtons();
    scheduleAutoRun('horizon');
  });
  document.querySelectorAll('input[name="scenarioPreset"]').forEach((input) => input.addEventListener('change', applyScenario));
  $('shortShock').addEventListener('input', (e) => { $('shortShockValue').textContent = e.target.value; if (document.querySelector('input[name="scenarioPreset"]:checked').value === 'custom') applyScenario(); });
  $('longShock').addEventListener('input', (e) => { $('longShockValue').textContent = e.target.value; if (document.querySelector('input[name="scenarioPreset"]:checked').value === 'custom') applyScenario(); });
  $('applyScenario').addEventListener('click', applyScenario);
  $('clearCache').addEventListener('click', async () => {
    if (!confirm('Clear the local cache?')) return;
    await apiFetch('/cache/clear', { method: 'POST', body: JSON.stringify({}) });
    await refreshSourcesStatus();
    if (state.result) await runDashboard(false, 'refresh');
  });
  $('allowFredFallback').addEventListener('change', (e) => { $('allowFredFallbackCacheTab').checked = e.target.checked; syncUrl(); });
  $('allowFredFallbackCacheTab').addEventListener('change', (e) => { $('allowFredFallback').checked = e.target.checked; syncUrl(); refreshSourcesStatus(); });
  ['dataSource', 'asOfDate', 'target', 'historyYears', 'shortRateTenor', 'weightingMethod', 'optimizationMode', 'mcPaths', 'randomSeed', 'cacheEnabled'].forEach((id) => {
    $(id).addEventListener('change', syncUrl);
  });
}

async function initialize() {
  $('asOfDate').value = todayIso();
  parseQuery();
  renderHorizonQuickButtons();
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
    runDashboard(false, 'startup');
  }
}

initialize().catch((error) => {
  console.error(error);
  alert(error.message);
});
