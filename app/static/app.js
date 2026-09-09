// Shared helpers
async function apiGet(path) {
  const res = await fetch(path);
  if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
  return res.json();
}
async function apiPost(path, body) {
  const res = await fetch(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
  return res.json();
}
function esc(s) {
  if (s === null || s === undefined) return '';
  return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}
function fmtDate(iso) {
  if (!iso) return '\u2014';
  return new Date(iso).toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' });
}
function badge(text, cls) {
  return '<span class="badge badge-' + esc(cls) + '">' + esc(text) + '</span>';
}
function toast(msg, isError) {
  const el = document.createElement('div');
  el.className = 'toast' + (isError ? ' error' : '');
  el.textContent = msg;
  document.body.appendChild(el);
  setTimeout(function () { el.remove(); }, 4000);
}

async function initDashboard() {
  const tbody = document.getElementById('cases-tbody');
  const typeSelect = document.getElementById('filter-exception-type');
  const stateSelect = document.getElementById('filter-state');

  // Populate the type dropdown from what's ACTUALLY in the database,
  // not a hardcoded list — found necessary directly: the API schema
  // documents payment/inventory/carrier/return/fraud as valid values,
  // but not every one of those has ever been assigned by real code,
  // so a hardcoded dropdown would offer options that always return
  // zero results.
  try {
    const types = await apiGet('/api/v1/cases/meta/exception-types');
    types.forEach(function (t) {
      const opt = document.createElement('option');
      opt.value = t;
      opt.textContent = t;
      typeSelect.appendChild(opt);
    });
  } catch (e) { /* non-fatal — filter still works with just "All" */ }

  async function render() {
    tbody.innerHTML = '<tr><td colspan="7"><div class="empty-state">Loading...</div></td></tr>';
    try {
      const params = new URLSearchParams();
      if (typeSelect.value) params.set('exception_type', typeSelect.value);
      if (stateSelect.value) params.set('state', stateSelect.value);
      const query = params.toString();
      const cases = await apiGet('/api/v1/cases' + (query ? '?' + query : ''));
      if (!cases.length) {
        tbody.innerHTML = '<tr><td colspan="7"><div class="empty-state">No cases match this filter.</div></td></tr>';
        return;
      }
      tbody.innerHTML = cases.map(function (c) {
        return '<tr>' +
          '<td><a class="id" href="/cases/' + esc(c.id) + '">' + esc(c.id.slice(0, 8)) + '</a></td>' +
          '<td class="id">' + esc(c.order_id) + '</td>' +
          '<td class="id">' + esc(c.customer_id) + '</td>' +
          '<td>' + esc(c.exception_type) + '</td>' +
          '<td>' + badge(c.state, c.state) + '</td>' +
          '<td>' + (c.fraud_flag ? badge('flagged', 'flagged') : '\u2014') + '</td>' +
          '<td>' + fmtDate(c.created_at) + '</td>' +
          '</tr>';
      }).join('');
    } catch (e) {
      tbody.innerHTML = '<tr><td colspan="7"><div class="empty-state">Error loading cases: ' + esc(e.message) + '</div></td></tr>';
    }
  }

  typeSelect.addEventListener('change', render);
  stateSelect.addEventListener('change', render);
  await render();
}

async function initCaseDetail() {
  const caseId = document.body.dataset.caseId;
  const root = document.getElementById('case-detail-root');
  try {
    const c = await apiGet('/api/v1/cases/' + caseId);
    let html = '';
    html += '<div class="panel"><h2>Case Overview</h2>';
    html += '<div class="field-row"><div class="field-label">Case ID</div><div class="field-value id">' + esc(c.id) + '</div></div>';
    html += '<div class="field-row"><div class="field-label">Order ID</div><div class="field-value id">' + esc(c.order_id) + '</div></div>';
    html += '<div class="field-row"><div class="field-label">Customer ID</div><div class="field-value id">' + esc(c.customer_id) + '</div></div>';
    html += '<div class="field-row"><div class="field-label">Channel</div><div class="field-value">' + esc(c.channel) + '</div></div>';
    html += '<div class="field-row"><div class="field-label">Exception type</div><div class="field-value">' + esc(c.exception_type) + '</div></div>';
    html += '<div class="field-row"><div class="field-label">State</div><div class="field-value">' + badge(c.state, c.state) + '</div></div>';
    html += '<div class="field-row"><div class="field-label">Fraud risk score</div><div class="field-value">' + (c.fraud_risk_score !== null && c.fraud_risk_score !== undefined ? c.fraud_risk_score : '\u2014') + (c.fraud_flag ? ' ' + badge('flagged', 'flagged') : '') + '</div></div>';
    html += '<div class="field-row"><div class="field-label">Created</div><div class="field-value">' + fmtDate(c.created_at) + '</div></div>';
    html += '<div class="field-row"><div class="field-label">Trace</div><div class="field-value"><a href="/traces/' + esc(c.id) + '">View full trace \u2192</a></div></div>';
    html += '</div>';

    if (c.diagnosis) {
      html += '<div class="panel"><h2>Diagnosis</h2>';
      html += '<div class="field-row"><div class="field-label">Terminated</div><div class="field-value">' + esc(c.diagnosis.terminated_reason) + '</div></div>';
      html += '<div class="field-row"><div class="field-label">Root causes</div><div class="field-value">' + ((c.diagnosis.root_causes || []).map(esc).join('<br>') || '\u2014') + '</div></div>';
      html += '</div>';
    }

    if (c.resolution_decision) {
      html += '<div class="panel"><h2>Resolution Decision</h2>';
      html += '<div class="field-row"><div class="field-label">Action</div><div class="field-value">' + esc(c.resolution_decision.action) + '</div></div>';
      html += '<div class="field-row"><div class="field-label">Amount</div><div class="field-value">$' + esc(c.resolution_decision.amount_usd) + '</div></div>';
      html += '<div class="field-row"><div class="field-label">Confidence</div><div class="field-value">' + esc(c.resolution_decision.confidence) + '</div></div>';
      html += '<div class="field-row"><div class="field-label">Reasoning</div><div class="field-value">' + esc(c.resolution_decision.reasoning) + '</div></div>';
      if (c.resolution_decision.cited_policy) {
        html += '<div class="field-row"><div class="field-label">Cited policy</div><div class="field-value id">' + esc(c.resolution_decision.cited_policy.doc_id) + ' v' + esc(c.resolution_decision.cited_policy.version) + '</div></div>';
      }
      html += '</div>';
    }

    if (c.execution_result) {
      html += '<div class="panel"><h2>Execution</h2>';
      html += '<div class="field-row"><div class="field-label">Status</div><div class="field-value">' + esc(c.execution_result.status) + '</div></div>';
      html += '<div class="field-row"><div class="field-label">Result</div><div class="field-value"><pre>' + esc(JSON.stringify(c.execution_result.result, null, 2)) + '</pre></div></div>';
      html += '</div>';
    }

    root.innerHTML = html;
  } catch (e) {
    root.innerHTML = '<div class="empty-state">Error loading case: ' + esc(e.message) + '</div>';
  }
}

async function initEscalations() {
  const tbody = document.getElementById('escalations-tbody');
  async function render() {
    const items = await apiGet('/api/v1/escalations');
    if (!items.length) {
      tbody.closest('table').style.display = 'none';
      document.getElementById('escalations-empty').style.display = 'block';
      return;
    }
    tbody.closest('table').style.display = '';
    document.getElementById('escalations-empty').style.display = 'none';
    tbody.innerHTML = items.map(function (c) {
      return '<tr>' +
        '<td><a class="id" href="/cases/' + esc(c.case_id) + '">' + esc(c.case_id.slice(0, 8)) + '</a></td>' +
        '<td class="id">' + esc(c.order_id) + '</td>' +
        '<td>' + esc(c.exception_type) + '</td>' +
        '<td>' + (c.fraud_flag ? badge('flagged', 'flagged') : '\u2014') + '</td>' +
        '<td>' + (c.proposed_resolution ? esc(c.proposed_resolution.action) + ' \u2014 $' + esc(c.proposed_resolution.amount_usd) : '\u2014') + '</td>' +
        '<td>' + esc(c.priority_score) + '</td>' +
        '<td><button class="primary" onclick="decideEscalation(\'' + esc(c.case_id) + '\', \'approve\')">Approve</button> ' +
        '<button class="danger" onclick="decideEscalation(\'' + esc(c.case_id) + '\', \'reject\')">Reject</button></td>' +
        '</tr>';
    }).join('');
  }
  window._reloadEscalations = render;
  await render();
}

async function decideEscalation(caseId, action) {
  const decidedBy = prompt('Your identity (e.g. human:you@company.com):', 'human:reviewer@company.com');
  if (!decidedBy) return;
  try {
    const result = await apiPost('/api/v1/escalations/' + caseId + '/decision', { action: action, decided_by: decidedBy });
    toast('Case ' + caseId.slice(0, 8) + ': ' + result.outcome);
    window._reloadEscalations();
  } catch (e) {
    toast('Error: ' + e.message, true);
  }
}

async function initPolicies() {
  const root = document.getElementById('policies-list');
  async function render() {
    const data = await apiGet('/api/v1/policies');
    if (!data.files_on_disk.length) {
      root.innerHTML = '<div class="empty-state">No policy documents uploaded yet.</div>';
      return;
    }
    let rows = data.files_on_disk.map(function (f) {
      const stem = f.replace(/\.pdf$/i, '');
      const indexed = data.indexed_doc_ids.indexOf(stem) !== -1;
      const info = (data.file_status || {})[f] || {};
      const statusCell = indexed
        ? badge('indexed', 'resolved')
        : badge('not indexed', 'blocked') + '<div class="rationale-text">' + esc(info.detail || '') + '</div>' +
          '<button style="margin-top:4px;" onclick="retryIngestion(\'' + esc(f) + '\')">Retry ingestion</button>';
      return '<tr><td class="id">' + esc(f) + '</td><td>' + statusCell + '</td></tr>';
    }).join('');
    root.innerHTML = '<table><thead><tr><th>File</th><th>Indexed</th></tr></thead><tbody>' + rows + '</tbody></table>';
  }
  await render();
  window._reloadPolicies = render;

  document.getElementById('upload-form').addEventListener('submit', async function (ev) {
    ev.preventDefault();
    const fileInput = document.getElementById('policy-file');
    if (!fileInput.files.length) return;
    const formData = new FormData();
    formData.append('file', fileInput.files[0]);
    try {
      const res = await fetch('/api/v1/policies/upload', { method: 'POST', body: formData });
      const body = await res.json();
      if (!res.ok) throw new Error(body.detail || 'Upload failed');
      toast('Indexed: ' + body.filename + ' (' + (body.summary && body.summary.doc_id ? body.summary.doc_id : 'ready') + ')');
      fileInput.value = '';
      await render();
    } catch (e) {
      toast('Error: ' + e.message, true);
    }
  });
}

async function retryIngestion(filename) {
  try {
    const resp = await fetch('/api/v1/policies/' + encodeURIComponent(filename) + '/reingest', { method: 'POST' });
    const body = await resp.json();
    if (!resp.ok) throw new Error(body.detail || 'Retry failed');
    toast('Indexed: ' + body.filename);
    if (window._reloadPolicies) await window._reloadPolicies();
  } catch (e) {
    toast('Error: ' + e.message, true);
  }
}

async function initMetrics() {
  const tabs = document.querySelectorAll('.tab');
  const root = document.getElementById('metrics-root');

  function renderMetrics(data) {
    const entries = Object.entries(data);
    const cards = entries
      .filter(function (kv) { return typeof kv[1] === 'number' || typeof kv[1] === 'string' || kv[1] === null; })
      .map(function (kv) {
        const k = kv[0], v = kv[1];
        const display = v === null ? '\u2014' : (typeof v === 'number' ? (Number.isInteger(v) ? v : v.toFixed(3)) : esc(v));
        return '<div class="metric-card"><div class="value">' + display + '</div><div class="label">' + esc(k.replace(/_/g, ' ')) + '</div></div>';
      }).join('');

    let perToolHtml = '';
    const rest = entries.filter(function (kv) {
      if (kv[0] === 'per_tool_metrics') { perToolHtml = renderPerToolTable(kv[1]); return false; }
      return typeof kv[1] === 'object' && kv[1] !== null;
    });
    const restHtml = rest.map(function (kv) {
      return '<div class="panel"><h2>' + esc(kv[0].replace(/_/g, ' ')) + '</h2><pre>' + esc(JSON.stringify(kv[1], null, 2)) + '</pre></div>';
    }).join('');
    return '<div class="metric-grid">' + cards + '</div>' + perToolHtml + restHtml;
  }

  function renderPerToolTable(perTool) {
    const names = Object.keys(perTool);
    if (!names.length) return '';
    const rows = names.map(function (name) {
      const m = perTool[name];
      const failurePct = (m.failure_rate * 100).toFixed(1) + '%';
      const failureCls = m.failure_rate > 0 ? 'blocked' : 'resolved';
      const latency = m.avg_latency_ms === null ? '\u2014' : m.avg_latency_ms.toFixed(2) + ' ms';
      const rw = m.is_write ? badge('write', 'escalated') : badge('read', 'diagnosing');
      return '<tr><td class="id">' + esc(name) + '</td><td>' + m.call_count + '</td><td>' +
             badge(failurePct, failureCls) + '</td><td>' + esc(latency) + '</td><td>' + rw + '</td></tr>';
    }).join('');
    return '<div class="panel"><h2>Per-Tool Breakdown</h2><table><thead><tr>' +
           '<th>Tool</th><th>Calls</th><th>Failure Rate</th><th>Avg Latency</th><th>Read/Write</th>' +
           '</tr></thead><tbody>' + rows + '</tbody></table></div>';
  }

  async function loadScope(scope) {
    root.innerHTML = '<div class="empty-state">Loading\u2026</div>';
    try {
      const data = await apiGet('/api/v1/metrics/' + scope);
      root.innerHTML = renderMetrics(data);
    } catch (e) {
      root.innerHTML = '<div class="empty-state">Error: ' + esc(e.message) + '</div>';
    }
  }

  tabs.forEach(function (tab) {
    tab.addEventListener('click', function () {
      tabs.forEach(function (t) { t.classList.remove('active'); });
      tab.classList.add('active');
      loadScope(tab.dataset.scope);
    });
  });
  await loadScope('agent');
}

async function initThreshold() {
  const proposalsRoot = document.getElementById('proposals-root');
  const overridesRoot = document.getElementById('overrides-root');

  async function render() {
    const proposals = await apiGet('/api/v1/threshold-proposals?status=pending_review');
    proposalsRoot.innerHTML = proposals.length ? proposals.map(function (p) {
      return '<div class="panel"><h2>' + esc(p.cluster_key) + '</h2>' +
        '<div class="field-row"><div class="field-label">Sample size</div><div class="field-value">' + p.sample_size + '</div></div>' +
        '<div class="field-row"><div class="field-label">Overturn rate</div><div class="field-value">' + (p.overturn_rate * 100).toFixed(1) + '%</div></div>' +
        '<div class="field-row"><div class="field-label">Current \u2192 Proposed</div><div class="field-value">' + p.current_threshold + ' \u2192 <strong>' + p.proposed_threshold + '</strong></div></div>' +
        '<p class="rationale-text">' + esc(p.rationale) + '</p>' +
        '<div class="actions-row"><button class="primary" onclick="decideThreshold(\'' + p.id + '\', \'accept\')">Accept</button> ' +
        '<button onclick="decideThreshold(\'' + p.id + '\', \'reject\')">Reject</button></div></div>';
    }).join('') : '<div class="empty-state">No pending threshold proposals. Run the batch job below to generate one from recent resolution outcomes.</div>';

    const overrides = await apiGet('/api/v1/threshold-proposals/active-overrides');
    overridesRoot.innerHTML = overrides.length ? ('<table><thead><tr><th>Cluster</th><th>Active threshold</th><th>Accepted by</th><th>Accepted at</th></tr></thead><tbody>' +
      overrides.map(function (o) {
        return '<tr><td>' + esc(o.cluster_key) + '</td><td>' + o.active_threshold + '</td><td class="id">' + esc(o.accepted_by) + '</td><td>' + fmtDate(o.accepted_at) + '</td></tr>';
      }).join('') + '</tbody></table>') : '<div class="empty-state">No overrides active \u2014 all clusters use the default threshold.</div>';
  }
  window._reloadThreshold = render;
  await render();

  document.getElementById('run-batch-form').addEventListener('submit', async function (ev) {
    ev.preventDefault();
    const threshold = parseFloat(document.getElementById('current-threshold').value);
    try {
      const result = await apiPost('/api/v1/threshold-proposals/run-batch-job', { current_threshold: threshold, min_sample_size: 5 });
      toast('Batch job created ' + result.proposals_created + ' proposal(s)');
      await render();
    } catch (e) {
      toast('Error: ' + e.message, true);
    }
  });
}

async function decideThreshold(proposalId, action) {
  const decidedBy = prompt('Your identity (required \u2014 cannot be "system"):', 'human:ops-lead@company.com');
  if (!decidedBy) return;
  try {
    await apiPost('/api/v1/threshold-proposals/' + proposalId + '/' + action, { decided_by: decidedBy });
    toast('Proposal ' + action + 'ed');
    window._reloadThreshold();
  } catch (e) {
    toast('Error: ' + e.message, true);
  }
}

async function initAuditLog() {
  const tbody = document.getElementById('audit-tbody');
  async function render(caseId) {
    const path = caseId ? ('/api/v1/audit-log?case_id=' + encodeURIComponent(caseId)) : '/api/v1/audit-log';
    const items = await apiGet(path);
    if (!items.length) {
      tbody.innerHTML = '<tr><td colspan="5"><div class="empty-state">No audit entries found.</div></td></tr>';
      return;
    }
    tbody.innerHTML = items.map(function (e) {
      return '<tr>' +
        '<td>' + fmtDate(e.timestamp) + '</td>' +
        '<td class="id"><a href="/cases/' + esc(e.case_id) + '">' + esc((e.case_id || '').slice(0, 8)) + '</a></td>' +
        '<td>' + esc(e.actor) + '</td>' +
        '<td>' + esc(e.action) + '</td>' +
        '<td><pre>' + esc(JSON.stringify(e.detail, null, 2)) + '</pre></td>' +
        '</tr>';
    }).join('');
  }
  await render(null);
  document.getElementById('audit-filter-form').addEventListener('submit', function (ev) {
    ev.preventDefault();
    render(document.getElementById('audit-case-filter').value.trim() || null);
  });
}

async function initTrace() {
  const caseId = document.body.dataset.caseId;
  const root = document.getElementById('trace-root');
  try {
    const data = await apiGet('/api/v1/traces/' + caseId);
    root.innerHTML = data.spans.map(function (s) {
      return '<div class="trace-span">' +
        '<div class="span-name">' + esc(s.agent_or_tool_name) + '</div>' +
        '<div class="span-meta">' + fmtDate(s.created_at) + ' \u2014 span ' + esc(s.span_id.slice(0, 8)) + (s.parent_span_id ? ' \u00b7 parent ' + esc(s.parent_span_id.slice(0, 8)) : '') + '</div>' +
        '<pre>input: ' + esc(JSON.stringify(s.input, null, 2)) + '</pre>' +
        '<pre>output: ' + esc(JSON.stringify(s.output, null, 2)) + '</pre>' +
        '<pre>metadata: ' + esc(JSON.stringify(s.metadata, null, 2)) + '</pre>' +
        '</div>';
    }).join('');
  } catch (e) {
    root.innerHTML = '<div class="empty-state">No trace found for this case, or error: ' + esc(e.message) + '</div>';
  }
}

document.addEventListener('DOMContentLoaded', function () {
  const page = document.body.dataset.page;
  const dispatch = {
    dashboard: initDashboard,
    case_detail: initCaseDetail,
    escalations: initEscalations,
    policies: initPolicies,
    metrics: initMetrics,
    threshold: initThreshold,
    audit: initAuditLog,
    trace: initTrace,
  };
  if (dispatch[page]) dispatch[page]();
});
