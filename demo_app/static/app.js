/**
 * Frontend Application Logic for Neuro-Symbolic Italian Dialect Identification Demo
 */

const STATE = {
  initData: null,
  activeRegion: null,
  activeExpectedCode: null,
  activePresetId: null,
  socialOffset: 0,
  socialLimit: 20,
  socialTotal: 0,
  currentSocialPosts: [],
  socialPredictionsCache: {},
};

document.addEventListener("DOMContentLoaded", async () => {
  setupTabs();
  setupComposerEvents();
  setupSocialExplorerEvents();
  await initializeApp();
});

function setupTabs() {
  const tabBtns = document.querySelectorAll(".tab-btn");
  tabBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      const target = btn.getAttribute("data-tab");
      switchTab(target);
    });
  });

  document.getElementById("jumpExplainBtn").addEventListener("click", () => {
    switchTab("explain");
  });
}

function switchTab(tabName) {
  document.querySelectorAll(".tab-btn").forEach((b) => {
    b.classList.toggle("active", b.getAttribute("data-tab") === tabName);
  });
  document.querySelectorAll(".tab-panel").forEach((p) => {
    p.classList.toggle("active", p.id === `tab-${tabName}`);
  });
}

async function initializeApp() {
  try {
    const res = await fetch("/api/init");
    const data = await res.json();
    STATE.initData = data;

    // Topbar Device Indicator
    const devMap = {
      mps: "Apple Silicon MPS · 3 Models Live",
      cuda: "NVIDIA CUDA GPU · 3 Models Live",
      cpu: "CPU Engine · 3 Models Live",
    };
    document.getElementById("deviceLabel").textContent =
      devMap[data.device] || `Device: ${data.device}`;
    document.getElementById("socialCountBadge").textContent =
      data.total_social_posts.toLocaleString();

    // Render Presets
    renderPresetButtons(data.presets);

    // Render Benchmark & Methodology Tab
    renderBenchmarkTab(data.benchmark, data.dialects);

    // Render SHAP Embeds
    if (data.shap_html) {
      document.getElementById("shapBaseContainer").innerHTML =
        data.shap_html.comparison_base || "<em>SHAP report not found</em>";
      document.getElementById("shapFusionContainer").innerHTML =
        data.shap_html.comparison_fusion || "<em>SHAP report not found</em>";
    }

    // Populate Social Region Filter
    populateSocialRegions(data.social_regions, data.dialects);
    await loadSocialPosts();

    // Automatically select and run first code-switching preset
    if (data.presets && data.presets.length > 0) {
      selectPreset(data.presets[0]);
    }
  } catch (err) {
    console.error("Initialization error:", err);
    document.getElementById("deviceLabel").textContent = "Server Connection Error";
  }
}

function renderPresetButtons(presets) {
  const container = document.getElementById("presetButtons");
  container.innerHTML = "";
  presets.forEach((p) => {
    const btn = document.createElement("button");
    btn.className = "preset-chip";
    btn.dataset.id = p.id;
    btn.innerHTML = `<span class="preset-code">${p.expected_dialect}</span> <span>${escapeHtml(p.title)}</span>`;
    btn.addEventListener("click", () => selectPreset(p));
    container.appendChild(btn);
  });
}

function selectPreset(preset) {
  STATE.activePresetId = preset.id;
  document.querySelectorAll(".preset-chip").forEach((c) => {
    c.classList.toggle("active", c.dataset.id === preset.id);
  });

  const inputEl = document.getElementById("inputText");
  inputEl.value = preset.text;

  setRegionConstraint(preset.region, preset.expected_dialect);

  const noteBox = document.getElementById("presetNoteBox");
  if (preset.note) {
    noteBox.innerHTML = `<strong>Linguistic Context:</strong> ${escapeHtml(preset.note)}`;
    noteBox.classList.remove("hidden");
  } else {
    noteBox.classList.add("hidden");
  }

  runPrediction();
}

function setRegionConstraint(regionName, expectedCode) {
  STATE.activeRegion = regionName;
  STATE.activeExpectedCode = expectedCode;
  const pill = document.getElementById("activeRegionPill");
  if (regionName && expectedCode) {
    document.getElementById("activeRegionName").textContent = regionName;
    document.getElementById("activeRegionCode").textContent = expectedCode;
    pill.classList.remove("hidden");
  } else {
    pill.classList.add("hidden");
  }
}

function setupComposerEvents() {
  const analyzeBtn = document.getElementById("analyzeBtn");
  const inputEl = document.getElementById("inputText");

  analyzeBtn.addEventListener("click", () => runPrediction());

  inputEl.addEventListener("keydown", (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
      e.preventDefault();
      runPrediction();
    }
  });

  inputEl.addEventListener("input", () => {
    // Deselect preset highlight if user edits text manually
    document.querySelectorAll(".preset-chip").forEach((c) => c.classList.remove("active"));
    document.getElementById("presetNoteBox").classList.add("hidden");
  });

  document.getElementById("clearRegionBtn").addEventListener("click", () => {
    setRegionConstraint(null, null);
    runPrediction();
  });
}

async function runPrediction() {
  const text = document.getElementById("inputText").value.trim();
  if (!text) return;

  const btnText = document.getElementById("analyzeBtnText");
  btnText.textContent = "Analyzing…";

  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text: text,
        region: STATE.activeRegion,
      }),
    });
    const data = await res.json();
    if (data.error) {
      alert(data.error);
      return;
    }

    renderModelCards(data);
    renderTokenFragmentation(data.tokenization);
    renderExplainabilityTab(data);
  } catch (err) {
    console.error("Prediction error:", err);
  } finally {
    btnText.textContent = "Run Live Inference";
  }
}

function renderModelCards(data) {
  const grid = document.getElementById("modelsGrid");
  const order = [
    { key: "logreg", tier: "Baseline 1 · Linear Model", proposed: false },
    { key: "baseline_bert", tier: "Baseline 2 · Pure Transformer", proposed: false },
    { key: "fusion_bert", tier: "Proposed · Neuro-Symbolic Architecture", proposed: true },
  ];

  const regEval = data.region_evaluation;

  grid.innerHTML = order
    .map((cfg) => {
      const m = data.models[cfg.key];
      const pred = m.predicted;

      let regionBadgeHtml = "";
      if (regEval && regEval.matches) {
        const isMatch = regEval.matches[cfg.key];
        if (isMatch) {
          regionBadgeHtml = `
            <div class="region-match-tag match">
              <span>✓ Matches Geolocated Origin (${escapeHtml(regEval.region)} → <code>${regEval.expected_code}</code>)</span>
            </div>`;
        } else {
          regionBadgeHtml = `
            <div class="region-match-tag mismatch">
              <span>✕ Mismatch with Origin (${escapeHtml(regEval.region)} expects <code>${regEval.expected_code}</code>)</span>
            </div>`;
        }
      }

      const barsHtml = m.top_classes
        .map(
          (c) => `
          <div class="prob-row">
            <span class="prob-lbl"><code>${c.code}</code> ${escapeHtml(c.name)}</span>
            <div class="prob-bar-track">
              <div class="prob-bar-fill" style="width: ${Math.max(c.prob, 1.2)}%; background: ${c.color};"></div>
            </div>
            <span class="prob-pct">${c.prob.toFixed(1)}%</span>
          </div>`
        )
        .join("");

      return `
        <div class="model-card ${cfg.proposed ? "proposed" : ""}">
          ${cfg.proposed ? '<div class="model-ribbon">Proposed Model</div>' : ""}
          <div>
            <div class="model-header">
              <div>
                <div class="model-tier">${cfg.tier}</div>
                <div class="model-name">${escapeHtml(m.title)}</div>
              </div>
              <span class="model-latency">${m.latency_ms} ms</span>
            </div>

            <div class="pred-hero">
              <div class="pred-top-row">
                <div class="pred-dialect-name">
                  <span class="iso-pill" style="background: ${pred.color};">${pred.code}</span>
                  <span>${escapeHtml(pred.name)}</span>
                </div>
                <span class="pred-conf" style="color: ${pred.color};">${pred.confidence.toFixed(1)}%</span>
              </div>
              <div class="pred-sub">
                <span>Native: <em>${escapeHtml(pred.native)}</em></span>
                <span>${escapeHtml(pred.region)}</span>
              </div>
              ${regionBadgeHtml}
            </div>
          </div>

          <div class="prob-list">
            ${barsHtml}
          </div>
        </div>
      `;
    })
    .join("");
}

function renderTokenFragmentation(tok) {
  document.getElementById("metricWords").textContent = tok.num_words;
  document.getElementById("metricSubwords").textContent = tok.num_subwords;
  document.getElementById("metricFertility").textContent = tok.fertility_rate.toFixed(2);
  document.getElementById("fertilityBadgeVal").textContent = `${tok.fertility_rate.toFixed(3)} (${tok.num_subwords}/${tok.num_words})`;

  const strip = document.getElementById("tokenFragmentationStrip");
  strip.innerHTML = tok.words
    .map((w) => {
      const piecesHtml = w.pieces
        .map((p) => `<span class="subword-pill">${escapeHtml(p)}</span>`)
        .join("");
      return `
        <div class="word-tok-chip ${w.is_fragmented ? "fragmented" : ""}">
          <span class="word-surface">${escapeHtml(w.word)}</span>
          <div class="word-pieces">${piecesHtml}</div>
        </div>
      `;
    })
    .join("");
}

function renderExplainabilityTab(data) {
  const exp = data.explainability;
  document.getElementById("explainCurrentText").textContent = `"${data.text}"`;
  const badge = document.getElementById("explainTargetBadge");
  badge.textContent = `${exp.target_dialect.code.toUpperCase()} · ${exp.target_dialect.name}`;
  badge.style.background = exp.target_dialect.color;

  // Render Word Attribution Boxes
  document.getElementById("attrFusionBox").innerHTML = renderWordAttributionsHtml(
    exp.attributions_fusion
  );
  document.getElementById("attrBaseBox").innerHTML = renderWordAttributionsHtml(
    exp.attributions_baseline
  );

  // Render Top-15 Symbolic Features
  const auxContainer = document.getElementById("auxFeaturesChart");
  const maxImpact = Math.max(
    ...exp.top_symbolic_features.map((f) => Math.abs(f.projected_impact)),
    0.01
  );

  auxContainer.innerHTML = exp.top_symbolic_features
    .map((f) => {
      const pct = Math.min((Math.abs(f.projected_impact) / maxImpact) * 100, 100);
      const isPos = f.projected_impact >= 0;
      const barColor = isPos ? "#0D9488" : "#E11D48";
      return `
        <div class="aux-bar-row" title="Raw Feature: ${escapeHtml(f.raw_name)} | Standardized z-score: ${f.z_score}">
          <span class="aux-feat-name">${escapeHtml(f.label)}</span>
          <span class="aux-cat-badge">${escapeHtml(f.category)}</span>
          <div class="aux-bar-track">
            <div class="aux-bar-fill" style="width: ${pct}%; background: ${barColor};"></div>
          </div>
          <span class="aux-impact-val" style="color: ${barColor};">${f.projected_impact >= 0 ? "+" : ""}${f.projected_impact.toFixed(3)}</span>
        </div>
      `;
    })
    .join("");

  // Active Dialect Markers
  const markersEl = document.getElementById("activeMarkersList");
  if (exp.active_dialect_markers && exp.active_dialect_markers.length > 0) {
    markersEl.innerHTML = exp.active_dialect_markers
      .map(
        (m) => `
        <div class="marker-pill">
          <strong>${escapeHtml(m.label)}</strong> <span>(z = +${m.z_score})</span>
        </div>`
      )
      .join("");
  } else {
    markersEl.innerHTML = `<span style="color: var(--text-muted); font-size: 13px;">No explicit dialect regex shibboleths triggered (classified via character/word-edge n-grams &amp; contextual embeddings).</span>`;
  }
}

function renderWordAttributionsHtml(words) {
  return words
    .map((w) => {
      const ns = w.norm_score;
      const alpha = Math.min(Math.max(Math.abs(ns) * 0.78, 0.08), 0.78).toFixed(2);
      const bg =
        ns >= 0
          ? `rgba(16, 185, 129, ${alpha})`
          : `rgba(244, 63, 94, ${alpha})`;
      const subInfo = w.subwords.join(" + ");
      return `<span class="attr-word" style="background: ${bg};" title="Subwords: [${escapeHtml(subInfo)}] | Score: ${w.score}">${escapeHtml(w.word)}</span>`;
    })
    .join(" ");
}

/* =========================================================================
   TAB 3: SOCIAL MEDIA REGION EXPLORER
   ========================================================================= */
function populateSocialRegions(regions, dialects) {
  const sel = document.getElementById("regionSelect");
  sel.innerHTML = `<option value="all">All Italian Regions (4,262 Geotagged Posts)</option>`;
  regions.forEach((r) => {
    const dMeta = dialects[r.expected_dialect] || { name: r.expected_dialect };
    const opt = document.createElement("option");
    opt.value = r.region;
    opt.textContent = `${r.region} — ${r.count} posts (Target: ${r.expected_dialect} / ${dMeta.name})`;
    sel.appendChild(opt);
  });
}

function setupSocialExplorerEvents() {
  document.getElementById("regionSelect").addEventListener("change", () => {
    STATE.socialOffset = 0;
    loadSocialPosts();
  });

  let debounceTimer = null;
  document.getElementById("socialSearchInput").addEventListener("input", () => {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => {
      STATE.socialOffset = 0;
      loadSocialPosts();
    }, 220);
  });

  document.getElementById("prevPageBtn").addEventListener("click", () => {
    if (STATE.socialOffset >= STATE.socialLimit) {
      STATE.socialOffset -= STATE.socialLimit;
      loadSocialPosts();
    }
  });

  document.getElementById("nextPageBtn").addEventListener("click", () => {
    if (STATE.socialOffset + STATE.socialLimit < STATE.socialTotal) {
      STATE.socialOffset += STATE.socialLimit;
      loadSocialPosts();
    }
  });

  document.getElementById("batchEvalBtn").addEventListener("click", async () => {
    const btn = document.getElementById("batchEvalBtn");
    btn.textContent = "Running Live Inference on Table…";
    btn.disabled = true;
    try {
      for (const post of STATE.currentSocialPosts) {
        if (!STATE.socialPredictionsCache[post.id]) {
          const res = await fetch("/api/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text: post.text, region: post.region }),
          });
          const predData = await res.json();
          STATE.socialPredictionsCache[post.id] = predData;
          updateSocialRowPrediction(post.id, predData, post.expected_dialect);
        }
      }
    } finally {
      btn.textContent = "Run Live Predictions on Visible Posts";
      btn.disabled = false;
    }
  });
}

async function loadSocialPosts() {
  const reg = document.getElementById("regionSelect").value;
  const q = document.getElementById("socialSearchInput").value.trim();

  const url = `/api/social-posts?region=${encodeURIComponent(reg)}&q=${encodeURIComponent(q)}&limit=${STATE.socialLimit}&offset=${STATE.socialOffset}`;
  const res = await fetch(url);
  const data = await res.json();

  STATE.socialTotal = data.total;
  STATE.currentSocialPosts = data.posts;

  const start = data.total === 0 ? 0 : STATE.socialOffset + 1;
  const end = Math.min(STATE.socialOffset + STATE.socialLimit, data.total);
  document.getElementById("socialShowingText").textContent =
    `Showing ${start}–${end} of ${data.total.toLocaleString()} geolocated posts`;

  document.getElementById("prevPageBtn").disabled = STATE.socialOffset === 0;
  document.getElementById("nextPageBtn").disabled =
    STATE.socialOffset + STATE.socialLimit >= data.total;

  const tbody = document.getElementById("socialTableBody");
  const dialects = STATE.initData ? STATE.initData.dialects : {};

  tbody.innerHTML = data.posts
    .map((p) => {
      const dMeta = dialects[p.expected_dialect] || {
        color: "#555",
        name: p.expected_dialect,
      };
      const cached = STATE.socialPredictionsCache[p.id];
      const predCellHtml = cached
        ? formatMiniPreds(cached, p.expected_dialect)
        : `<span style="color: var(--text-muted); font-size: 12px;">Click 'Test in Arena' or 'Run Live Predictions'</span>`;

      return `
        <tr id="social-row-${escapeHtml(p.id)}">
          <td>
            <strong>${escapeHtml(p.region)}</strong>
            ${p.city ? `<div style="font-size: 11.5px; color: var(--text-muted);">📍 ${escapeHtml(p.city)}</div>` : ""}
          </td>
          <td>
            <span class="iso-pill" style="background: ${dMeta.color};">${p.expected_dialect}</span>
            <div style="font-size: 11.5px; color: var(--text-secondary); margin-top: 2px;">${escapeHtml(dMeta.name)}</div>
          </td>
          <td style="font-size: 13.5px; line-height: 1.45;">${escapeHtml(p.text)}</td>
          <td class="pred-cell">${predCellHtml}</td>
          <td style="text-align: right;">
            <button class="btn-table-action" data-post-id="${escapeHtml(p.id)}">Test in Arena →</button>
          </td>
        </tr>
      `;
    })
    .join("");

  tbody.querySelectorAll(".btn-table-action").forEach((btn) => {
    btn.addEventListener("click", () => {
      const pid = btn.getAttribute("data-post-id");
      const post = STATE.currentSocialPosts.find((x) => x.id === pid);
      if (post) {
        document.getElementById("inputText").value = post.text;
        setRegionConstraint(post.region, post.expected_dialect);
        document.getElementById("presetNoteBox").innerHTML =
          `<strong>Real Social Media Post (ID: ${escapeHtml(post.id)}):</strong> Geolocated in <strong>${escapeHtml(post.city || post.region)}</strong> (${escapeHtml(post.region)}). Expected regional dialect: <code>${post.expected_dialect}</code>.`;
        document.getElementById("presetNoteBox").classList.remove("hidden");
        switchTab("arena");
        runPrediction();
      }
    });
  });
}

function updateSocialRowPrediction(postId, predData, expectedDialect) {
  const row = document.getElementById(`social-row-${postId}`);
  if (!row) return;
  const cell = row.querySelector(".pred-cell");
  if (cell) {
    cell.innerHTML = formatMiniPreds(predData, expectedDialect);
  }
}

function formatMiniPreds(predData, expectedDialect) {
  const lr = predData.models.logreg.predicted.code;
  const bb = predData.models.baseline_bert.predicted.code;
  const ff = predData.models.fusion_bert.predicted.code;
  return `
    <div class="mini-pred-pills">
      <span class="mini-pill ${lr === expectedDialect ? "match" : ""}" title="Logistic Regression">LR: ${lr}</span>
      <span class="mini-pill ${bb === expectedDialect ? "match" : ""}" title="Baseline BERT">BERT: ${bb}</span>
      <span class="mini-pill ${ff === expectedDialect ? "match" : ""}" title="Feature Fusion BERT">Fusion: ${ff}</span>
    </div>
  `;
}

/* =========================================================================
   TAB 4: BENCHMARK & METHODOLOGY
   ========================================================================= */
function renderBenchmarkTab(bmark, dialects) {
  // Feature breakdown in Explainability sidebar
  const fbMini = document.getElementById("featureBlocksMini");
  if (fbMini && bmark.aux_breakdown) {
    fbMini.innerHTML = Object.entries(bmark.aux_breakdown)
      .map(
        ([k, v]) => `
        <div class="fblock-row">
          <span>${escapeHtml(k)}</span>
          <code>${v} dims</code>
        </div>`
      )
      .join("");
  }

  // Benchmark Summary Cards
  const cardsRow = document.getElementById("benchmarkCardsRow");
  cardsRow.innerHTML = bmark.models
    .map((m, idx) => {
      const isProp = idx === 2;
      return `
        <div class="bmark-stat-card ${isProp ? "highlight" : ""}">
          <div class="card-kicker">${escapeHtml(m.short)}</div>
          <div class="bmark-Big">${(m.macro_f1 * 100).toFixed(2)}% <span style="font-size: 13px; font-weight: 500; color: var(--text-secondary);">Macro F1</span></div>
          <div style="font-size: 12.5px; color: var(--text-secondary);">
            Accuracy: <strong>${(m.accuracy * 100).toFixed(2)}%</strong> · Weighted F1: <strong>${(m.weighted_f1 * 100).toFixed(2)}%</strong>
          </div>
        </div>
      `;
    })
    .join("");

  // Benchmark Table
  const tbody = document.getElementById("benchmarkTableBody");
  tbody.innerHTML = bmark.models
    .map(
      (m, idx) => `
      <tr style="${idx === 2 ? "background: var(--accent-fusion-bg); font-weight: 600;" : ""}">
        <td>${escapeHtml(m.name)}</td>
        <td style="font-size: 12.5px; color: var(--text-secondary);">${escapeHtml(m.architecture)}</td>
        <td style="font-family: var(--font-mono);">${m.val_macro_f1.toFixed(4)}</td>
        <td style="font-family: var(--font-mono); font-weight: 700;">${m.macro_f1.toFixed(4)}</td>
        <td style="font-family: var(--font-mono);">${m.weighted_f1.toFixed(4)}</td>
        <td style="font-family: var(--font-mono);">${m.accuracy.toFixed(4)}</td>
        <td style="font-family: var(--font-mono);">${m.eval_time}</td>
      </tr>`
    )
    .join("");

  // 12 Dialects Reference Grid
  const refGrid = document.getElementById("dialectsRefGrid");
  refGrid.innerHTML = Object.values(dialects)
    .map((d) => {
      const lrStat = bmark.per_dialect_logreg_val.find((x) => x.code === d.code);
      return `
        <div class="dialect-ref-item">
          <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
              <span class="iso-pill" style="background: ${d.color};">${d.code}</span>
              <strong style="margin-left: 6px;">${escapeHtml(d.name)}</strong>
            </div>
            <span style="font-size: 11.5px; color: var(--text-muted);">${escapeHtml(d.native)}</span>
          </div>
          <div style="font-size: 12px; color: var(--text-secondary); margin-top: 4px;">
            Region: ${escapeHtml(d.region)} · Group: ${escapeHtml(d.group)}
          </div>
          ${
            lrStat
              ? `<div style="font-family: var(--font-mono); font-size: 11px; color: var(--text-muted); margin-top: 3px;">
                   Val Support: ${lrStat.support.toLocaleString()} · Baseline F1: ${lrStat.f1.toFixed(4)}
                 </div>`
              : ""
          }
        </div>
      `;
    })
    .join("");
}

function escapeHtml(str) {
  return String(str || "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

