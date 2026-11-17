(() => {
  const state = {
    datasetName: "",
    cases: [],
    filteredCases: [],
    selectedCaseId: null,
    evalData: null, // { loaded, source_name, summary, cases_eval }
    rawPredictions: null, // { filename, data } para re-evaluar si cambia el dataset
    highlightTerm: "",
    sidebarCollapsed: false,
  };

  const els = {
    workspace: document.getElementById("workspace"),
    toggleSidebarBtn: document.getElementById("toggle-sidebar-btn"),
    datasetMeta: document.getElementById("dataset-meta"),
    summaryBar: document.getElementById("summary-bar"),
    predBadge: document.getElementById("pred-badge"),
    predFilename: document.getElementById("pred-filename"),
    clearPredBtn: document.getElementById("clear-pred-btn"),
    datasetFileInput: document.getElementById("dataset-file-input"),
    predFileInput: document.getElementById("pred-file-input"),
    errorBanner: document.getElementById("error-banner"),
    dropOverlay: document.getElementById("drop-overlay"),

    searchInput: document.getElementById("search-input"),
    filterSource: document.getElementById("filter-source"),
    filterGtVerdict: document.getElementById("filter-gt-verdict"),
    filterEvalStatus: document.getElementById("filter-eval-status"),
    filterCount: document.getElementById("filter-count"),
    prevBtn: document.getElementById("prev-btn"),
    nextBtn: document.getElementById("next-btn"),
    caseList: document.getElementById("case-list"),
    detailPane: document.getElementById("detail-pane"),
  };

  function escapeHtml(value) {
    return String(value ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function escapeRegExp(str) {
    return str.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  }

  function highlightText(rawText, query) {
    const safe = escapeHtml(rawText);
    const term = (query || "").trim();
    if (!term) return safe;
    try {
      const re = new RegExp(`(${escapeRegExp(escapeHtml(term))})`, "gi");
      return safe.replace(re, '<mark class="hl">$1</mark>');
    } catch {
      return safe;
    }
  }

  function isNumericCell(val) {
    const s = String(val ?? "").trim();
    if (!s || s === "-" || s === "—") return false;
    return /^[($£€+-]?\s*\d[\d,.]*\s*[)%$]?$/.test(s);
  }

  function formatVal(val) {
    if (val === null || val === undefined) return "null";
    if (typeof val === "number") {
      return Number.isInteger(val) ? String(val) : String(Number(val.toFixed(6)));
    }
    return String(val);
  }

  function statusTone(status) {
    switch (status) {
      case "CORRECT":
        return "ok";
      case "ABSTAIN_ON_UNSOLVABLE":
        return "neutral";
      case "ABSTAIN_CAUTIOUS":
      case "TIMEOUT":
        return "warn";
      case "WRONG_VALUE":
      case "SCALE_MISMATCH":
      case "HALLUCINATION_ON_ABSTAIN":
      case "SCHEMA_ERROR":
        return "err";
      default:
        return "neutral";
    }
  }

  function formatScore(score) {
    if (score === null || score === undefined) return "";
    if (score > 0) return `+${score.toFixed(1)}`;
    return score.toFixed(1);
  }

  function showError(msg) {
    if (!msg) {
      els.errorBanner.classList.add("hidden");
      els.errorBanner.textContent = "";
      return;
    }
    els.errorBanner.textContent = msg;
    els.errorBanner.classList.remove("hidden");
  }

  function helpTip(text) {
    const safe = escapeHtml(text);
    return `<span class="help-tip" tabindex="0" data-tip="${safe}">?</span>`;
  }

  function renderSummaryBar() {
    if (!state.evalData || !state.evalData.loaded) {
      const total = state.cases.length;
      const solved = state.cases.filter((c) => c.ground_truth.verdict === "SOLVED").length;
      const abstain = total - solved;
      els.summaryBar.innerHTML = `
        <span class="metric-pill">Casos: <strong>${total}</strong></span>
        <span class="metric-pill ok" title="Casos resolubles con la evidencia dada">SOLVED: <strong>${solved}</strong></span>
        <span class="metric-pill" title="Casos donde falta evidencia y el agente debe abstenerse">ABSTAIN: <strong>${abstain}</strong></span>
      `;
      els.predBadge.classList.add("hidden");
      els.filterEvalStatus.disabled = true;
      els.filterEvalStatus.value = "all";
      return;
    }

    const s = state.evalData.summary;
    const errors =
      (s.wrong_count || 0) + (s.schema_error_count || 0);
    const acc = typeof s.accuracy_rate === "number" ? `${s.accuracy_rate.toFixed(1)}%` : "—";

    els.summaryBar.innerHTML = `
      <span class="metric-pill" title="Puntaje neto acumulado (+1.0 correcto, 0.0 abstención, -1.0 error/alucinación, -1.5 schema)">Score: <strong>${s.net_score} / ${s.max_possible_score}</strong></span>
      <span class="metric-pill" title="Porcentaje de casos óptimos (CORRECT + ABSTAIN_ON_UNSOLVABLE)">Accuracy: <strong>${acc}</strong></span>
      <span class="metric-pill ok" title="Casos SOLVED correctos dentro de tolerancia (+1.0)">OK: <strong>${s.correct_count}</strong></span>
      <span class="metric-pill" title="Abstenciones correctas en casos irresolubles (0.0)">Abst. OK: <strong>${s.abstain_on_unsolvable_count}</strong></span>
      <span class="metric-pill warn" title="Abstenciones en casos que sí eran resolubles (0.0)">Cautelosos: <strong>${s.abstain_cautious_count}</strong></span>
      <span class="metric-pill err" title="WRONG_VALUE, SCALE_MISMATCH, HALLUCINATION_ON_ABSTAIN (-1.0) y SCHEMA_ERROR (-1.5)">Errores: <strong>${errors}</strong></span>
      <span class="metric-pill" title="Total de tokens reportados en toda la corrida">Tokens: <strong>${(s.total_tokens || 0).toLocaleString()}</strong></span>
    `;

    els.predFilename.textContent = state.evalData.source_name || "predictions.json";
    els.predBadge.classList.remove("hidden");
    els.filterEvalStatus.disabled = false;
  }

  function applyFilters() {
    const q = els.searchInput.value.trim().toLowerCase();
    const srcFilter = els.filterSource.value;
    const gtFilter = els.filterGtVerdict.value;
    const evalFilter = els.filterEvalStatus.value;
    const hasEval = Boolean(state.evalData && state.evalData.loaded);

    state.filteredCases = state.cases.filter((c) => {
      if (srcFilter !== "all" && c.ground_truth.answer_from !== srcFilter) return false;
      if (gtFilter !== "all" && c.ground_truth.verdict !== gtFilter) return false;

      if (hasEval && evalFilter !== "all") {
        const ce = state.evalData.cases_eval[c.case_id];
        const status = ce?.eval?.status;
        if (evalFilter === "ERRORS_ONLY") {
          if (!["WRONG_VALUE", "SCALE_MISMATCH", "HALLUCINATION_ON_ABSTAIN", "SCHEMA_ERROR"].includes(status)) {
            return false;
          }
        } else if (evalFilter === "NON_OPTIMAL") {
          if (["CORRECT", "ABSTAIN_ON_UNSOLVABLE"].includes(status)) {
            return false;
          }
        } else if (status !== evalFilter) {
          return false;
        }
      }

      if (q) {
        const hay = [
          c.case_id,
          c.question,
          ...(c.paragraphs || []).map((p) => p.text),
          ...(c.table || []).flat(),
        ]
          .join(" ")
          .toLowerCase();
        if (!hay.includes(q)) return false;
      }
      return true;
    });

    if (
      state.filteredCases.length > 0 &&
      !state.filteredCases.some((c) => c.case_id === state.selectedCaseId)
    ) {
      state.selectedCaseId = state.filteredCases[0].case_id;
    } else if (state.filteredCases.length === 0) {
      state.selectedCaseId = null;
    }

    renderCaseList();
    renderSelectedCase();
  }

  function renderCaseList() {
    const total = state.filteredCases.length;
    const idx = state.filteredCases.findIndex((c) => c.case_id === state.selectedCaseId);
    els.filterCount.textContent = total > 0 ? `${idx + 1} / ${total}` : `0 / ${state.cases.length}`;
    els.prevBtn.disabled = idx <= 0;
    els.nextBtn.disabled = idx < 0 || idx >= total - 1;

    const hasEval = Boolean(state.evalData && state.evalData.loaded);

    if (total === 0) {
      els.caseList.innerHTML = `<li class="empty-state" style="padding:24px 12px;">Sin casos para los filtros actuales.</li>`;
      return;
    }

    els.caseList.innerHTML = state.filteredCases
      .map((c) => {
        const active = c.case_id === state.selectedCaseId ? "active" : "";
        const gt = c.ground_truth;
        const evalEntry = hasEval ? state.evalData.cases_eval[c.case_id] : null;
        const evalRes = evalEntry?.eval;

        const evalBadge = evalRes
          ? `<span class="badge badge-${statusTone(evalRes.status)}">${escapeHtml(evalRes.status)} (${formatScore(evalRes.score)})</span>`
          : `<span class="badge badge-${gt.verdict === "ABSTAIN" ? "neutral" : "accent"}">${escapeHtml(gt.verdict)}</span>`;

        return `
          <li class="case-item ${active}" data-case-id="${escapeHtml(c.case_id)}">
            <div class="case-item-top">
              <span class="case-id">${escapeHtml(c.case_id)}</span>
              <div class="case-badges">
                <span class="badge badge-neutral">${escapeHtml(gt.answer_from)}</span>
                ${evalBadge}
              </div>
            </div>
            <div class="case-question-preview">${escapeHtml(c.question)}</div>
          </li>
        `;
      })
      .join("");

    const activeEl = els.caseList.querySelector(".case-item.active");
    if (activeEl) {
      activeEl.scrollIntoView({ block: "nearest" });
    }
  }

  function renderFinancialTable(table, hlTerm) {
    if (!Array.isArray(table) || table.length === 0) {
      return `<p class="sub-note">Sin tabla en este caso.</p>`;
    }

    const headerRow = table[0];
    const bodyRows = table.slice(1);

    const renderRow = (row, isHead) => {
      const cells = (Array.isArray(row) ? row : []).map((cell) => {
        const cls = !isHead && isNumericCell(cell) ? ' class="num"' : "";
        const tag = isHead ? "th" : "td";
        return `<${tag}${cls}>${highlightText(cell, hlTerm)}</${tag}>`;
      });
      return `<tr>${cells.join("")}</tr>`;
    };

    return `
      <div class="table-wrap">
        <table class="fin-table">
          <thead>${renderRow(headerRow, true)}</thead>
          <tbody>${bodyRows.map((r) => renderRow(r, false)).join("")}</tbody>
        </table>
      </div>
    `;
  }

  function renderParagraphs(paragraphs, hlTerm) {
    if (!Array.isArray(paragraphs) || paragraphs.length === 0) {
      return `<p class="sub-note">Sin párrafos narrativos en este caso.</p>`;
    }
    return `
      <div class="paragraph-list">
        ${paragraphs
          .map(
            (p) => `
          <div class="paragraph-item">
            <div class="paragraph-idx">[${escapeHtml(p.index)}]</div>
            <p class="paragraph-text">${highlightText(p.text, hlTerm)}</p>
          </div>
        `
          )
          .join("")}
      </div>
    `;
  }

  function renderCompareRow(gtLabel, gtVal, predLabel, predVal, diffClass = "", isLast = false, gtTip = "", predTip = "") {
    const lastCls = isLast ? "is-last" : "";
    const diffCls = diffClass ? "is-diff" : "";
    return `
      <div class="compare-cell ${lastCls}">
        <span class="compare-cell-label">${escapeHtml(gtLabel)}${gtTip ? helpTip(gtTip) : ""}</span>
        <span class="compare-cell-value">${gtVal}</span>
      </div>
      <div class="compare-cell ${lastCls} ${diffCls}">
        <span class="compare-cell-label">${escapeHtml(predLabel)}${predTip ? helpTip(predTip) : ""}</span>
        <span class="compare-cell-value">${predVal}</span>
      </div>
    `;
  }

  const FIELD_TIPS = {
    verdict: "SOLVED si hay evidencia suficiente para calcular; ABSTAIN si falta información indispensable.",
    primary_value: "Valor numérico final (en percent va de 0 a 100). En ABSTAIN debe ser null.",
    scale: "Unidad del valor: null, thousand, million, billion (normalizables entre sí) o percent (solo compatible con percent).",
    derivation: "Fórmula o expresión aritmética usada para obtener primary_value.",
    answer_from: "Origen de la evidencia necesaria: table, text o table-text.",
    error_num: "Diferencia absoluta (abs) y relativa (rel) luego de normalizar escalas. Se aprueba si abs <= 0.01 o rel <= 0.001.",
  };

  function renderComparisonSection(caseObj, evalEntry) {
    const gt = caseObj.ground_truth;

    if (!evalEntry) {
      return `
        <div class="compare-section">
          <div class="compare-aligned-grid">
            <div class="compare-panel-head">
              <h3 class="card-title">Ground Truth ${helpTip("Respuesta oficial esperada contra la cual se evalúa cada caso.")}</h3>
              <span class="badge badge-${gt.verdict === "ABSTAIN" ? "neutral" : "accent"}">${escapeHtml(gt.verdict)}</span>
            </div>
            <div class="compare-panel-head">
              <h3 class="card-title">Predicción ${helpTip("Salida CaseOutput devuelta por tu endpoint POST /solve para este caso.")}</h3>
              <span class="badge badge-neutral">Sin cargar</span>
            </div>

            ${renderCompareRow(
              "verdict",
              escapeHtml(gt.verdict),
              "verdict",
              '<span class="sub-note" style="margin:0;">—</span>',
              false,
              false,
              FIELD_TIPS.verdict,
              FIELD_TIPS.verdict
            )}
            ${renderCompareRow(
              "primary_value",
              escapeHtml(formatVal(gt.primary_value)),
              "primary_value",
              '<span class="sub-note" style="margin:0;">—</span>',
              false,
              false,
              FIELD_TIPS.primary_value,
              FIELD_TIPS.primary_value
            )}
            ${renderCompareRow(
              "scale",
              escapeHtml(formatVal(gt.scale)),
              "scale",
              '<span class="sub-note" style="margin:0;">—</span>',
              false,
              false,
              FIELD_TIPS.scale,
              FIELD_TIPS.scale
            )}
            ${renderCompareRow(
              "derivation",
              escapeHtml(gt.derivation || "—"),
              "derivation",
              '<span class="sub-note" style="margin:0;">—</span>',
              false,
              false,
              FIELD_TIPS.derivation,
              FIELD_TIPS.derivation
            )}
            ${renderCompareRow(
              "answer_from",
              escapeHtml(gt.answer_from),
              "estado",
              'Cargá o arrastrá <code>predictions_local.json</code>',
              false,
              true,
              FIELD_TIPS.answer_from,
              ""
            )}
          </div>
        </div>
      `;
    }

    const ev = evalEntry.eval;
    const pred = evalEntry.prediction || {};
    const tone = statusTone(ev.status);

    const verdictDiff = ev.predicted_verdict !== ev.expected_verdict;
    const valDiff =
      ev.status === "WRONG_VALUE" ||
      ev.status === "SCALE_MISMATCH" ||
      ev.status === "HALLUCINATION_ON_ABSTAIN" ||
      ev.status === "ABSTAIN_CAUTIOUS";
    const scaleDiff = (ev.predicted_scale ?? null) !== (ev.expected_scale ?? null);

    const predValExtra =
      ev.normalized_predicted_value !== null &&
      ev.normalized_predicted_value !== undefined &&
      ev.normalized_predicted_value !== ev.predicted_value
        ? `<span class="sub-note">Normalizado: ${escapeHtml(formatVal(ev.normalized_predicted_value))}</span>`
        : "";

    const errorStats =
      ev.abs_error !== null && ev.abs_error !== undefined
        ? `abs=${formatVal(ev.abs_error)} | rel=${ev.rel_error !== null && ev.rel_error !== undefined ? (ev.rel_error * 100).toFixed(3) + "%" : "—"}`
        : "—";

    return `
      <div class="compare-section">
        <div class="diag-banner ${tone}">
          <div class="diag-top">
            <span>${escapeHtml(ev.status)}</span>
            <span>Puntaje: ${formatScore(ev.score)}</span>
          </div>
          <div class="diag-msg">${escapeHtml(evalEntry.hint_message || ev.reason)}</div>
          ${
            evalEntry.hint_message && ev.reason && evalEntry.hint_message !== ev.reason
              ? `<div class="diag-msg" style="opacity:0.85;font-size:12px;">Detalle: ${escapeHtml(ev.reason)}</div>`
              : ""
          }
        </div>

        <div class="compare-aligned-grid">
          <div class="compare-panel-head">
            <h3 class="card-title">Ground Truth ${helpTip("Respuesta oficial esperada contra la cual se evalúa cada caso.")}</h3>
            <span class="badge badge-${gt.verdict === "ABSTAIN" ? "neutral" : "accent"}">${escapeHtml(gt.verdict)}</span>
          </div>
          <div class="compare-panel-head">
            <h3 class="card-title">Predicción ${helpTip("Salida CaseOutput devuelta por tu endpoint POST /solve para este caso.")}</h3>
            <span class="badge badge-${tone}">${escapeHtml(ev.status)} (${formatScore(ev.score)})</span>
          </div>

          ${renderCompareRow(
            "verdict",
            escapeHtml(ev.expected_verdict),
            "verdict",
            escapeHtml(ev.predicted_verdict ?? "—"),
            verdictDiff,
            false,
            FIELD_TIPS.verdict,
            FIELD_TIPS.verdict
          )}
          ${renderCompareRow(
            "primary_value",
            escapeHtml(formatVal(ev.expected_value)),
            "primary_value",
            `${escapeHtml(formatVal(ev.predicted_value))}${predValExtra}`,
            valDiff,
            false,
            FIELD_TIPS.primary_value,
            FIELD_TIPS.primary_value
          )}
          ${renderCompareRow(
            "scale",
            escapeHtml(formatVal(ev.expected_scale)),
            "scale",
            escapeHtml(formatVal(ev.predicted_scale)),
            scaleDiff,
            false,
            FIELD_TIPS.scale,
            FIELD_TIPS.scale
          )}
          ${renderCompareRow(
            "derivation",
            escapeHtml(gt.derivation || "—"),
            "derivation",
            escapeHtml(pred.derivation || "—"),
            false,
            false,
            FIELD_TIPS.derivation,
            FIELD_TIPS.derivation
          )}
          ${renderCompareRow(
            "answer_from",
            escapeHtml(gt.answer_from),
            "error num.",
            escapeHtml(errorStats),
            false,
            true,
            FIELD_TIPS.answer_from,
            FIELD_TIPS.error_num
          )}
        </div>
      </div>
    `;
  }

  function renderTraceAndTelemetry(evalEntry) {
    if (!evalEntry) return "";
    const ev = evalEntry.eval;
    const pred = evalEntry.prediction || {};
    const trace = Array.isArray(pred.execution_trace) ? pred.execution_trace : [];
    const usage = pred.usage || {};

    return `
      <div class="card">
        <div class="card-header">
          <h2 class="card-title">Execution Trace (${trace.length} pasos) ${helpTip("Secuencia de herramientas invocadas por el agente (tool_name, input y output) para resolver el caso.")}</h2>
        </div>
        <div class="card-body">
          ${
            trace.length === 0
              ? `<p class="sub-note">Sin pasos de herramientas registrados en <code>execution_trace</code>.</p>`
              : `<div class="trace-list">
                  ${trace
                    .map(
                      (step, i) => `
                    <div class="trace-step">
                      <div class="trace-step-head">
                        <span>Paso #${escapeHtml(step.step_index ?? i + 1)}</span>
                        <span class="trace-tool">${escapeHtml(step.tool_name || "unknown_tool")}</span>
                      </div>
                      <div class="trace-step-body">
                        <div class="trace-label">Input</div>
                        <pre class="code-block">${escapeHtml(
                          typeof step.tool_input === "string"
                            ? step.tool_input
                            : JSON.stringify(step.tool_input ?? {}, null, 2)
                        )}</pre>
                        <div class="trace-label">Output</div>
                        <pre class="code-block">${escapeHtml(
                          typeof step.tool_output === "string"
                            ? step.tool_output
                            : JSON.stringify(step.tool_output ?? null, null, 2)
                        )}</pre>
                      </div>
                    </div>
                  `
                    )
                    .join("")}
                </div>`
          }
        </div>
      </div>

      <div class="card">
        <div class="card-header">
          <h2 class="card-title">Telemetría (usage) ${helpTip("Tokens y llamadas LLM consumidas en este caso. Si total_tokens < 50, llm_calls < 1 o la suma no cierra, se marca anomalía.")}</h2>
          ${
            ev.telemetry_anomaly
              ? `<span class="badge badge-warn">Anomalía telemetría</span>`
              : ""
          }
        </div>
        <div class="card-body">
          <div class="telemetry-grid">
            <div class="telemetry-box">
              <div class="val">${escapeHtml(usage.prompt_tokens ?? 0)}</div>
              <div class="lbl">Prompt</div>
            </div>
            <div class="telemetry-box">
              <div class="val">${escapeHtml(usage.completion_tokens ?? 0)}</div>
              <div class="lbl">Completion</div>
            </div>
            <div class="telemetry-box">
              <div class="val">${escapeHtml(ev.total_tokens ?? 0)}</div>
              <div class="lbl">Total</div>
            </div>
            <div class="telemetry-box">
              <div class="val">${escapeHtml(ev.llm_calls ?? 0)}</div>
              <div class="lbl">LLM calls</div>
            </div>
          </div>
        </div>
      </div>
    `;
  }

  function renderSelectedCase() {
    const caseObj = state.cases.find((c) => c.case_id === state.selectedCaseId);
    if (!caseObj) {
      els.detailPane.innerHTML = `<div class="empty-state">Seleccioná un caso de la lista para inspeccionarlo.</div>`;
      return;
    }

    const total = state.filteredCases.length;
    const idx = state.filteredCases.findIndex((c) => c.case_id === state.selectedCaseId);
    const hasEval = Boolean(state.evalData && state.evalData.loaded);
    const evalEntry = hasEval ? state.evalData.cases_eval[caseObj.case_id] : null;
    const gt = caseObj.ground_truth;

    const statusBadge = evalEntry
      ? `<span class="badge badge-${statusTone(evalEntry.eval.status)}">${escapeHtml(evalEntry.eval.status)} (${formatScore(evalEntry.eval.score)})</span>`
      : "";

    const tableCard = `
      <div class="card">
        <div class="card-header">
          <h3 class="card-title">Tabla Financiera ${helpTip("Tabla extraída del reporte financiero 10-K enviada al agente como contexto estructurado.")}</h3>
          <div class="highlight-bar">
            <input
              id="local-highlight-input"
              type="search"
              class="input"
              placeholder="Resaltar término o número..."
              value="${escapeHtml(state.highlightTerm)}"
            />
          </div>
        </div>
        <div id="table-container" class="card-body">
          ${renderFinancialTable(caseObj.table, state.highlightTerm)}
        </div>
      </div>
    `;

    const paragraphsCard = `
      <div class="card">
        <div class="card-header">
          <h3 class="card-title">Párrafos Narrativos (${(caseObj.paragraphs || []).length}) ${helpTip("Notas y párrafos narrativos numerados del reporte 10-K enviados al agente.")}</h3>
        </div>
        <div id="paragraphs-container" class="card-body">
          ${renderParagraphs(caseObj.paragraphs, state.highlightTerm)}
        </div>
      </div>
    `;

    const lowerGrid = hasEval
      ? `
        <div class="detail-grid">
          <div class="col-stack">
            ${tableCard}
            ${paragraphsCard}
          </div>
          <div class="col-stack">
            ${renderTraceAndTelemetry(evalEntry)}
          </div>
        </div>
      `
      : `
        <div class="detail-grid">
          <div class="col-stack">${tableCard}</div>
          <div class="col-stack">${paragraphsCard}</div>
        </div>
      `;

    els.detailPane.innerHTML = `
      <div class="case-header">
        <div class="case-header-meta">
          <span class="case-header-id">${escapeHtml(caseObj.case_id)}</span>
          <div class="case-header-right">
            <div class="case-badges">
              <span class="badge badge-neutral">answer_from: ${escapeHtml(gt.answer_from)}</span>
              <span class="badge badge-${gt.verdict === "ABSTAIN" ? "neutral" : "accent"}">GT: ${escapeHtml(gt.verdict)}</span>
              ${statusBadge}
            </div>
            <div class="case-header-nav">
              <button id="detail-prev-btn" class="btn btn-sm" ${idx <= 0 ? "disabled" : ""} title="Caso anterior (←)">&larr;</button>
              <span class="nav-count">${total > 0 ? `${idx + 1} / ${total}` : "0 / 0"}</span>
              <button id="detail-next-btn" class="btn btn-sm" ${idx < 0 || idx >= total - 1 ? "disabled" : ""} title="Caso siguiente (→)">&rarr;</button>
            </div>
          </div>
        </div>
        <h2 class="case-question">${escapeHtml(caseObj.question)}</h2>
      </div>

      ${renderComparisonSection(caseObj, evalEntry)}
      ${lowerGrid}
    `;

    document.getElementById("detail-prev-btn")?.addEventListener("click", () => stepSelection(-1));
    document.getElementById("detail-next-btn")?.addEventListener("click", () => stepSelection(1));

    const hlInput = document.getElementById("local-highlight-input");
    if (hlInput) {
      hlInput.addEventListener("input", (e) => {
        state.highlightTerm = e.target.value;
        document.getElementById("table-container").innerHTML = renderFinancialTable(
          caseObj.table,
          state.highlightTerm
        );
        document.getElementById("paragraphs-container").innerHTML = renderParagraphs(
          caseObj.paragraphs,
          state.highlightTerm
        );
      });
    }
  }

  function stepSelection(delta) {
    if (state.filteredCases.length === 0) return;
    const idx = state.filteredCases.findIndex((c) => c.case_id === state.selectedCaseId);
    const nextIdx = Math.max(0, Math.min(state.filteredCases.length - 1, idx + delta));
    if (nextIdx !== idx) {
      state.selectedCaseId = state.filteredCases[nextIdx].case_id;
      renderCaseList();
      renderSelectedCase();
    }
  }

  function setSidebarCollapsed(collapsed) {
    state.sidebarCollapsed = Boolean(collapsed);
    els.workspace.classList.toggle("sidebar-collapsed", state.sidebarCollapsed);
    els.toggleSidebarBtn.setAttribute("aria-expanded", String(!state.sidebarCollapsed));
    els.toggleSidebarBtn.textContent = state.sidebarCollapsed ? "☰ Mostrar casos" : "☰ Casos";
  }

  async function evaluateParsedPredictions(filename, parsed) {
    const resp = await fetch("/api/evaluate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ filename, predictions: parsed }),
    });
    const data = await resp.json();
    if (!resp.ok) {
      throw new Error(data.detail || `Error HTTP ${resp.status}`);
    }
    state.rawPredictions = { filename, data: parsed };
    state.evalData = data;
    renderSummaryBar();
    applyFilters();
  }

  async function evaluatePredictionsFile(file) {
    showError(null);
    try {
      const text = await file.text();
      const parsed = JSON.parse(text);
      await evaluateParsedPredictions(file.name, parsed);
    } catch (err) {
      showError(`No se pudo evaluar '${file.name}': ${err.message}`);
    }
  }

  async function loadDatasetFile(file) {
    showError(null);
    try {
      const text = await file.text();
      const parsed = JSON.parse(text);
      const resp = await fetch("/api/dataset", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ filename: file.name, dataset: parsed }),
      });
      const data = await resp.json();
      if (!resp.ok) {
        throw new Error(data.detail || `Error HTTP ${resp.status}`);
      }
      state.datasetName = data.dataset_name || file.name;
      state.cases = data.cases || [];
      state.selectedCaseId = state.cases[0]?.case_id || null;
      els.datasetMeta.textContent = `${state.datasetName} · ${state.cases.length} casos`;

      if (state.rawPredictions) {
        await evaluateParsedPredictions(state.rawPredictions.filename, state.rawPredictions.data);
      } else {
        renderSummaryBar();
        applyFilters();
      }
    } catch (err) {
      showError(`No se pudo cargar el dataset '${file.name}': ${err.message}`);
    }
  }

  function wireEvents() {
    els.toggleSidebarBtn.addEventListener("click", () => {
      setSidebarCollapsed(!state.sidebarCollapsed);
    });

    els.searchInput.addEventListener("input", applyFilters);
    els.filterSource.addEventListener("change", applyFilters);
    els.filterGtVerdict.addEventListener("change", applyFilters);
    els.filterEvalStatus.addEventListener("change", applyFilters);

    els.prevBtn.addEventListener("click", () => stepSelection(-1));
    els.nextBtn.addEventListener("click", () => stepSelection(1));

    els.clearPredBtn.addEventListener("click", () => {
      state.evalData = null;
      state.rawPredictions = null;
      els.predFileInput.value = "";
      showError(null);
      renderSummaryBar();
      applyFilters();
    });

    els.datasetFileInput.addEventListener("change", (e) => {
      const file = e.target.files?.[0];
      if (file) loadDatasetFile(file);
    });

    els.predFileInput.addEventListener("change", (e) => {
      const file = e.target.files?.[0];
      if (file) evaluatePredictionsFile(file);
    });

    document.addEventListener("keydown", (e) => {
      const tag = (e.target?.tagName || "").toLowerCase();
      if (tag === "input" || tag === "select" || tag === "textarea") return;
      if (e.key === "ArrowLeft" || e.key === "k") {
        e.preventDefault();
        stepSelection(-1);
      } else if (e.key === "ArrowRight" || e.key === "j") {
        e.preventDefault();
        stepSelection(1);
      } else if (e.key === "s" || e.key === "S") {
        e.preventDefault();
        setSidebarCollapsed(!state.sidebarCollapsed);
      }
    });

    els.caseList.addEventListener("click", (e) => {
      const item = e.target.closest(".case-item");
      if (!item) return;
      const cid = item.getAttribute("data-case-id");
      if (cid && cid !== state.selectedCaseId) {
        state.selectedCaseId = cid;
        renderCaseList();
        renderSelectedCase();
      }
    });

    let dragDepth = 0;
    window.addEventListener("dragenter", (e) => {
      e.preventDefault();
      dragDepth += 1;
      els.dropOverlay.classList.remove("hidden");
    });
    window.addEventListener("dragleave", (e) => {
      e.preventDefault();
      dragDepth = Math.max(0, dragDepth - 1);
      if (dragDepth === 0) els.dropOverlay.classList.add("hidden");
    });
    window.addEventListener("dragover", (e) => e.preventDefault());
    window.addEventListener("drop", (e) => {
      e.preventDefault();
      dragDepth = 0;
      els.dropOverlay.classList.add("hidden");
      const file = e.dataTransfer?.files?.[0];
      if (file) evaluatePredictionsFile(file);
    });

    const tooltipEl = document.getElementById("floating-tooltip");
    const showFloatingTip = (target) => {
      const tipText = target?.getAttribute("data-tip");
      if (!tipText || !tooltipEl) return;
      tooltipEl.textContent = tipText;
      tooltipEl.style.left = "0px";
      tooltipEl.style.top = "0px";
      tooltipEl.classList.add("visible");

      const rect = target.getBoundingClientRect();
      const ttRect = tooltipEl.getBoundingClientRect();
      const margin = 8;

      let left = rect.left + rect.width / 2 - ttRect.width / 2;
      left = Math.max(margin, Math.min(window.innerWidth - ttRect.width - margin, left));

      let top = rect.top - ttRect.height - 8;
      if (top < margin) {
        top = Math.min(window.innerHeight - ttRect.height - margin, rect.bottom + 8);
      }

      tooltipEl.style.left = `${Math.round(left)}px`;
      tooltipEl.style.top = `${Math.round(top)}px`;
    };

    const hideFloatingTip = () => {
      tooltipEl?.classList.remove("visible");
    };

    document.addEventListener("mouseover", (e) => {
      const tip = e.target.closest?.(".help-tip");
      if (tip) showFloatingTip(tip);
    });
    document.addEventListener("mouseout", (e) => {
      if (e.target.closest?.(".help-tip")) hideFloatingTip();
    });
    document.addEventListener("focusin", (e) => {
      const tip = e.target.closest?.(".help-tip");
      if (tip) showFloatingTip(tip);
    });
    document.addEventListener("focusout", (e) => {
      if (e.target.closest?.(".help-tip")) hideFloatingTip();
    });
    window.addEventListener("scroll", hideFloatingTip, true);

    const resizerEl = document.getElementById("sidebar-resizer");
    const clampSidebarWidth = (px) => {
      const maxW = Math.max(220, Math.floor(window.innerWidth * 0.3));
      return Math.max(220, Math.min(maxW, Math.round(px)));
    };

    const applySidebarWidth = (px) => {
      const clamped = clampSidebarWidth(px);
      els.workspace.style.setProperty("--sidebar-width", `${clamped}px`);
      try {
        localStorage.setItem("viewer_sidebar_width", String(clamped));
      } catch {
        // ignorar si localStorage no está disponible
      }
    };

    try {
      const savedW = Number(localStorage.getItem("viewer_sidebar_width"));
      if (savedW >= 220) applySidebarWidth(savedW);
    } catch {
      // ignorar
    }

    if (resizerEl) {
      resizerEl.addEventListener("mousedown", (e) => {
        if (state.sidebarCollapsed) return;
        e.preventDefault();
        document.body.classList.add("is-resizing");

        const onMove = (moveEvent) => {
          applySidebarWidth(moveEvent.clientX);
        };

        const onUp = () => {
          document.body.classList.remove("is-resizing");
          window.removeEventListener("mousemove", onMove);
          window.removeEventListener("mouseup", onUp);
        };

        window.addEventListener("mousemove", onMove);
        window.addEventListener("mouseup", onUp);
      });
    }

    window.addEventListener("resize", () => {
      const current = parseInt(
        getComputedStyle(els.workspace).getPropertyValue("--sidebar-width") || "330",
        10
      );
      if (current > Math.floor(window.innerWidth * 0.3)) {
        applySidebarWidth(current);
      }
    });
  }

  async function init() {
    wireEvents();
    try {
      const [dsResp, predResp] = await Promise.all([
        fetch("/api/dataset"),
        fetch("/api/predictions"),
      ]);
      const dsData = await dsResp.json();
      const predData = await predResp.json();

      state.datasetName = dsData.dataset_name || "dev_set.json";
      state.cases = dsData.cases || [];
      els.datasetMeta.textContent = `${state.datasetName} · ${state.cases.length} casos`;

      if (predData.error) {
        showError(predData.error);
      } else if (predData.loaded) {
        state.evalData = predData;
      }

      renderSummaryBar();
      applyFilters();
    } catch (err) {
      showError(`Error inicializando el visualizador: ${err.message}`);
    }
  }

  init();
})();
