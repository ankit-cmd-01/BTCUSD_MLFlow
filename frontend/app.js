const API = {
    overview: "/api/dashboard/overview",
    drift: "/api/model-drift",
};

const el = {
    predictionValue: document.getElementById("predictionValue"),
    predictionDelta: document.getElementById("predictionDelta"),
    predictionTimestamp: document.getElementById("predictionTimestamp"),
    perfRmse: document.getElementById("perfRmse"),
    perfMae: document.getElementById("perfMae"),
    perfR2: document.getElementById("perfR2"),
    driftRatio: document.getElementById("driftRatio"),
    driftPerfAlert: document.getElementById("driftPerfAlert"),
    driftFeatureCount: document.getElementById("driftFeatureCount"),
    lrRmse: document.getElementById("lrRmse"),
    lrMae: document.getElementById("lrMae"),
    lrR2: document.getElementById("lrR2"),
    lrRmseBar: document.getElementById("lrRmseBar"),
    lrMaeBar: document.getElementById("lrMaeBar"),
    lrR2Bar: document.getElementById("lrR2Bar"),
    arimaRmse: document.getElementById("arimaRmse"),
    arimaMae: document.getElementById("arimaMae"),
    arimaRmseBar: document.getElementById("arimaRmseBar"),
    arimaMaeBar: document.getElementById("arimaMaeBar"),
    diffRmse: document.getElementById("diffRmse"),
    diffMae: document.getElementById("diffMae"),
    betterModel: document.getElementById("betterModel"),
    driftFeatures: document.getElementById("driftFeatures"),
    runsTableBody: document.getElementById("runsTableBody"),
    refreshBtn: document.getElementById("refreshBtn"),
    checkDriftBtn: document.getElementById("checkDriftBtn"),
    refWindow: document.getElementById("refWindow"),
    curWindow: document.getElementById("curWindow"),
    rmseThreshold: document.getElementById("rmseThreshold"),
};

function clampPercent(value) {
    return Math.max(8, Math.min(100, value));
}

function setBarWidth(target, value, maxValue, higherIsBetter = false) {
    if (!target || value == null || maxValue == null || Number.isNaN(Number(value)) || Number.isNaN(Number(maxValue)) || Number(maxValue) <= 0) {
        if (target) {
            target.style.width = "0%";
        }
        return;
    }

    const numericValue = Number(value);
    const numericMax = Number(maxValue);
    const percent = higherIsBetter
        ? clampPercent(numericValue * 100)
        : clampPercent((1 - (numericValue / numericMax)) * 100);

    target.style.width = `${percent}%`;
}

function asMoney(value) {
    return Number(value).toLocaleString(undefined, { maximumFractionDigits: 4 });
}

function setDriftColor(target, value, badIfHigh = true) {
    if (value == null) {
        target.style.color = "#9ec3d8";
        return;
    }

    const isBad = badIfHigh ? Number(value) > 0 : Number(value) < 0;
    target.style.color = isBad ? "#ff6c7a" : "#6be3a5";
}

function formatMetric(value, digits = 6) {
    if (value === null || value === undefined || Number.isNaN(Number(value))) {
        return "-";
    }
    return Number(value).toFixed(digits);
}

function setDiffColor(target, value, lowerIsBetter = true) {
    if (value === null || value === undefined || Number.isNaN(Number(value))) {
        target.className = "indicator-neutral";
        return;
    }

    const numeric = Number(value);
    const isBetter = lowerIsBetter ? numeric < 0 : numeric > 0;
    target.className = isBetter ? "indicator-good" : "indicator-bad";
}

async function fetchJson(url) {
    const response = await fetch(url);
    if (!response.ok) {
        const payload = await response.json().catch(() => ({}));
        throw new Error(payload.detail || "Request failed");
    }
    return response.json();
}

function renderOverview(data) {
    const { prediction, performance, drift, runs, model_metrics: modelMetrics } = data;

    el.predictionValue.textContent = `$${asMoney(prediction.predicted_next_hour_close)}`;
    const deltaSign = prediction.delta >= 0 ? "+" : "";
    el.predictionDelta.textContent = `${deltaSign}${asMoney(prediction.delta)} (${deltaSign}${prediction.delta_pct}%)`;
    el.predictionDelta.style.color = prediction.delta >= 0 ? "#6be3a5" : "#ff6c7a";
    el.predictionTimestamp.textContent = `From: ${new Date(prediction.last_timestamp).toLocaleString()}`;

    el.perfRmse.textContent = performance.rmse;
    el.perfMae.textContent = performance.mae;
    el.perfR2.textContent = performance.r2;

    el.driftRatio.textContent = `${(drift.rmse_drift_ratio * 100).toFixed(2)}%`;
    setDriftColor(el.driftRatio, drift.rmse_drift_ratio, true);

    el.driftPerfAlert.textContent = drift.performance_drift_alert ? "ALERT" : "OK";
    el.driftPerfAlert.style.color = drift.performance_drift_alert ? "#ff6c7a" : "#6be3a5";

    el.driftFeatureCount.textContent = drift.feature_drift_alert_features.length;
    el.driftFeatureCount.style.color = drift.feature_drift_alert_features.length > 0 ? "#ffae4f" : "#6be3a5";

    renderModelMetrics(modelMetrics);
    renderDriftFeatures(drift.feature_psi);
    renderRuns(runs.runs);
}

function renderModelMetrics(modelMetrics) {
    if (!modelMetrics) {
        return;
    }

    const linear = modelMetrics.linear_regression || {};
    const arima = modelMetrics.arima || {};
    const delta = modelMetrics.delta_arima_minus_linear || {};

    el.lrRmse.textContent = formatMetric(linear.rmse);
    el.lrMae.textContent = formatMetric(linear.mae);
    el.lrR2.textContent = formatMetric(linear.r2);

    el.arimaRmse.textContent = formatMetric(arima.rmse);
    el.arimaMae.textContent = formatMetric(arima.mae);

    const rmseMax = Math.max(Number(linear.rmse || 0), Number(arima.rmse || 0));
    const maeMax = Math.max(Number(linear.mae || 0), Number(arima.mae || 0));

    setBarWidth(el.lrRmseBar, linear.rmse, rmseMax, false);
    setBarWidth(el.arimaRmseBar, arima.rmse, rmseMax, false);
    setBarWidth(el.lrMaeBar, linear.mae, maeMax, false);
    setBarWidth(el.arimaMaeBar, arima.mae, maeMax, false);
    setBarWidth(el.lrR2Bar, linear.r2, 1, true);

    el.diffRmse.textContent = formatMetric(delta.rmse);
    el.diffMae.textContent = formatMetric(delta.mae);

    setDiffColor(el.diffRmse, delta.rmse, true);
    setDiffColor(el.diffMae, delta.mae, true);

    if (delta.rmse == null && delta.mae == null) {
        el.betterModel.textContent = "-";
        el.betterModel.className = "indicator-neutral";
        return;
    }

    const score = [delta.rmse, delta.mae]
        .filter((value) => value !== null && value !== undefined)
        .reduce((acc, value) => acc + Number(value), 0);

    if (score > 0) {
        el.betterModel.textContent = "Linear Regression";
        el.betterModel.className = "indicator-good";
    } else if (score < 0) {
        el.betterModel.textContent = "ARIMA";
        el.betterModel.className = "indicator-good";
    } else {
        el.betterModel.textContent = "Tie";
        el.betterModel.className = "indicator-neutral";
    }
}

function switchTab(button, tabName) {
    document.querySelectorAll(".tabs .tab").forEach((tab) => {
        tab.classList.remove("active");
    });

    if (button) {
        button.classList.add("active");
    }

    const lrTab = document.getElementById("tabLr");
    const arimaTab = document.getElementById("tabArima");

    if (lrTab) {
        lrTab.style.display = tabName === "lr" ? "grid" : "none";
    }

    if (arimaTab) {
        arimaTab.style.display = tabName === "arima" ? "grid" : "none";
    }
}

window.switchTab = switchTab;

function renderDriftFeatures(featurePsi) {
    const sortedEntries = Object.entries(featurePsi)
        .filter(([, value]) => value !== null)
        .sort((a, b) => b[1] - a[1]);

    if (!sortedEntries.length) {
        el.driftFeatures.innerHTML = "<p class='muted'>No drift features available.</p>";
        return;
    }

    el.driftFeatures.innerHTML = sortedEntries
        .map(([feature, psi]) => {
            const color = psi >= 0.2 ? "#ff6c7a" : psi >= 0.1 ? "#ffae4f" : "#6be3a5";
            return `<div class="drift-item"><span>${feature}</span><strong style="color:${color}">${psi}</strong></div>`;
        })
        .join("");
}

function renderRuns(runs) {
    if (!runs.length) {
        el.runsTableBody.innerHTML = "<tr><td colspan='6'>No runs found</td></tr>";
        return;
    }

    el.runsTableBody.innerHTML = runs
        .map((run) => {
            const rmse = run.metrics.rmse || run.metrics.linear_rmse || "-";
            const mae = run.metrics.mae || run.metrics.linear_mae || "-";
            const r2 = run.metrics.r2 || run.metrics.linear_r2 || "-";

            return `
        <tr>
          <td>${run.experiment_name}</td>
          <td>${run.run_id.slice(0, 8)}...</td>
          <td>${run.status}</td>
          <td>${rmse}</td>
          <td>${mae}</td>
          <td>${r2}</td>
        </tr>
      `;
        })
        .join("");
}

async function checkDrift() {
    try {
        const ref = Number(el.refWindow?.value ?? 720);
        const cur = Number(el.curWindow?.value ?? 168);
        const threshold = Number(el.rmseThreshold?.value ?? 0.15);

        const drift = await fetchJson(`${API.drift}?reference_window=${ref}&current_window=${cur}&rmse_alert_threshold=${threshold}`);

        el.driftRatio.textContent = `${(drift.rmse_drift_ratio * 100).toFixed(2)}%`;
        setDriftColor(el.driftRatio, drift.rmse_drift_ratio, true);
        el.driftPerfAlert.textContent = drift.performance_drift_alert ? "ALERT" : "OK";
        el.driftPerfAlert.style.color = drift.performance_drift_alert ? "#ff6c7a" : "#6be3a5";
        el.driftFeatureCount.textContent = drift.feature_drift_alert_features.length;
        renderDriftFeatures(drift.feature_psi);
    } catch (err) {
        alert(`Drift check failed: ${err.message}`);
    }
}

async function loadAll() {
    try {
        const overview = await fetchJson(API.overview);

        renderOverview(overview);
    } catch (err) {
        alert(`Dashboard load failed: ${err.message}`);
    }
}

if (el.refreshBtn) {
    el.refreshBtn.addEventListener("click", loadAll);
}

if (el.checkDriftBtn) {
    el.checkDriftBtn.addEventListener("click", checkDrift);
}

loadAll();
