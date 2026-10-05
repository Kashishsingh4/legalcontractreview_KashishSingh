import { useEffect, useRef, useState } from "react";

const titleCase = (s) =>
  (s || "").replace(/_/g, " ").replace(/\w\S*/g, (w) => w[0].toUpperCase() + w.slice(1).toLowerCase());
const pct = (x, d = 1) => `${((x || 0) * 100).toFixed(d)}%`;
const stem = (name) => (name || "contract").replace(/\.[^.]+$/, "");

const STEPS = ["Parse", "Classify", "Retrieve", "Analyze", "Review", "Report"];

function downloadJson(data, filename) {
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

function statusOf(finding) {
  if (finding.risk_assessment === "attention_required") return "risk";
  if (finding.risk_assessment === "insufficient_evidence") return "unknown";
  return "ok";
}
const STATUS_LABEL = { risk: "Attention required", unknown: "Insufficient evidence", ok: "No risk factor found" };

function Badge({ tone, children }) {
  return <span className={`badge badge-${tone}`}>{children}</span>;
}

function Note({ tone = "neutral", children }) {
  return <div className={`note note-${tone}`}>{children}</div>;
}

/* ---------------- Header & history drawer ---------------- */

function Header({ onHistory, historyCount, onReset, hasResult }) {
  return (
    <header className="topbar">
      <div className="brand" onClick={onReset}>
        <div className="logo">⚖</div>
        <div>
          <div className="brand-name">ClauseLens</div>
          <div className="brand-sub">AI Legal Contract Review</div>
        </div>
      </div>
      <div className="topbar-actions">
        {hasResult && (
          <button className="btn btn-ghost" onClick={onReset}>
            + New analysis
          </button>
        )}
        <button className="btn btn-ghost" onClick={onHistory}>
          History <span className="pill">{historyCount}</span>
        </button>
      </div>
    </header>
  );
}

function HistoryDrawer({ open, onClose, history }) {
  return (
    <>
      <div className={`scrim ${open ? "show" : ""}`} onClick={onClose} />
      <aside className={`drawer ${open ? "open" : ""}`}>
        <div className="drawer-head">
          <h3>Analysis history</h3>
          <button className="icon-btn" onClick={onClose} aria-label="Close">
            ✕
          </button>
        </div>
        <p className="muted small">
          New uploads are automatically compared against stored contracts by content similarity.
        </p>
        {history.length === 0 && <div className="empty">No contracts analyzed yet.</div>}
        {history.slice(0, 20).map((row) => (
          <div key={row.doc_id} className="history-item">
            <div className="file-icon">📄</div>
            <div className="history-meta">
              <div className="history-name">{row.filename}</div>
              <div className="muted small">
                {row.clause_count} clauses · {row.page_count} pages
                {row.created_at ? ` · ${String(row.created_at).slice(0, 10)}` : ""}
              </div>
            </div>
            {row.overall_assessment && <Badge tone="neutral">{titleCase(row.overall_assessment)}</Badge>}
          </div>
        ))}
      </aside>
    </>
  );
}

/* ---------------- Upload / landing ---------------- */

function Landing({ onFile, loading, error, fileName }) {
  const inputRef = useRef(null);
  const [drag, setDrag] = useState(false);
  const [step, setStep] = useState(0);

  useEffect(() => {
    if (!loading) return setStep(0);
    const t = setInterval(() => setStep((s) => Math.min(s + 1, STEPS.length - 1)), 2500);
    return () => clearInterval(t);
  }, [loading]);

  const pick = (file) => file && onFile(file);

  return (
    <div className="landing">
      <div className="landing-copy">
        <span className="eyebrow">Evidence-first analysis</span>
        <h1>
          Review contracts with <span className="grad">sourced evidence</span>, not guesswork.
        </h1>
        <p className="muted lead">
          Document parsing, LegalBERT clause classification, GraphRAG retrieval and a restricted local-LLM that only
          explains the evidence it is given. No arbitrary 0–100 risk score.
        </p>
      </div>

      <div
        className={`dropzone ${drag ? "drag" : ""} ${loading ? "busy" : ""}`}
        onClick={() => !loading && inputRef.current?.click()}
        onDragOver={(e) => {
          e.preventDefault();
          setDrag(true);
        }}
        onDragLeave={() => setDrag(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDrag(false);
          if (!loading) pick(e.dataTransfer.files?.[0]);
        }}
      >
        <input
          ref={inputRef}
          type="file"
          accept=".pdf,.docx"
          hidden
          onChange={(e) => {
            pick(e.target.files?.[0]);
            e.target.value = "";
          }}
        />
        {loading ? (
          <>
            <div className="loader" />
            <div className="dz-title">Analyzing {fileName}</div>
            <div className="stepper">
              {STEPS.map((s, i) => (
                <div key={s} className={`step ${i < step ? "done" : i === step ? "active" : ""}`}>
                  <div className="step-dot">{i < step ? "✓" : i + 1}</div>
                  <div className="step-label">{s}</div>
                </div>
              ))}
            </div>
          </>
        ) : (
          <>
            <div className="dz-icon">⇪</div>
            <div className="dz-title">Drop your contract here</div>
            <div className="muted">
              or <span className="link">browse files</span> · PDF or DOCX
            </div>
          </>
        )}
      </div>

      {error && <Note tone="risk">{error}</Note>}

      <div className="features">
        {[
          ["🏷", "Clause classification", "LegalBERT predicts each clause's type with a confidence score."],
          ["🔎", "GraphRAG evidence", "Retrieves supporting legal knowledge with graph paths."],
          ["🔄", "Version tracking", "Automatically diffs against earlier versions of the same contract."],
        ].map(([icon, t, d]) => (
          <div key={t} className="feature">
            <div className="feature-icon">{icon}</div>
            <div className="feature-title">{t}</div>
            <div className="muted small">{d}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

/* ---------------- Results ---------------- */

function Stat({ label, value, tone }) {
  return (
    <div className={`stat ${tone ? `stat-${tone}` : ""}`}>
      <div className="stat-value">{value}</div>
      <div className="stat-label">{label}</div>
    </div>
  );
}

function Section({ title, children, aside }) {
  return (
    <div className="section">
      <div className="section-head">
        <h4>{title}</h4>
        {aside}
      </div>
      {children}
    </div>
  );
}

function ClauseDetail({ clause, finding }) {
  const [showEn, setShowEn] = useState(false);
  const status = statusOf(finding);
  const retrieval = finding.evidence?.retrieval?.results || [];
  const cmp = finding.text_comparison || {};

  return (
    <div className="detail">
      <div className="detail-head">
        <div>
          <div className="muted small">Clause {clause.index + 1}</div>
          <h2>{clause.heading || `Clause ${clause.index + 1}`}</h2>
        </div>
        <Badge tone={status}>{STATUS_LABEL[status]}</Badge>
      </div>

      <div className="kv-row">
        <div className="kv">
          <span>Type</span>
          <b>{finding.clause_type}</b>
        </div>
        <div className="kv">
          <span>Confidence</span>
          <b>{pct(finding.confidence)}</b>
          <div className="bar">
            <i style={{ width: pct(finding.confidence) }} />
          </div>
        </div>
        <div className="kv">
          <span>Retrieval relevance</span>
          <b>{pct(finding.best_relevance)}</b>
          <div className="bar">
            <i style={{ width: pct(finding.best_relevance) }} />
          </div>
        </div>
        <div className="kv">
          <span>Review decision</span>
          <b className="mono">{finding.review_decision}</b>
        </div>
      </div>
      {finding.review_reason && <p className="muted small">{finding.review_reason}</p>}

      <Section
        title="Clause text"
        aside={
          <div className="seg">
            <button className={!showEn ? "on" : ""} onClick={() => setShowEn(false)}>
              Original
            </button>
            <button className={showEn ? "on" : ""} onClick={() => setShowEn(true)}>
              English
            </button>
          </div>
        }
      >
        <blockquote className="clause-text">{showEn ? clause.text_en : clause.text_original}</blockquote>
        {cmp.available ? (
          cmp.same_text ? (
            <p className="muted small">✓ Normalized original and English texts are equivalent.</p>
          ) : (
            <p className="muted small">
              ⚠ {cmp.note || "The texts differ."} ({cmp.original_length ?? 0} vs {cmp.english_length ?? 0} chars)
            </p>
          )
        ) : (
          <p className="muted small">{cmp.note || "Comparison unavailable."}</p>
        )}
      </Section>

      <Section title="Matched risk factors">
        {finding.risk_factors?.length ? (
          finding.risk_factors.map((f, i) => (
            <div key={i} className="factor">
              <div className="factor-title">{f.factor}</div>
              <div className="small">
                Matched: <mark>{f.matched_text}</mark>
              </div>
              {(f.sources || []).map((s, j) => (
                <a key={j} className="source" href={s.url} target="_blank" rel="noreferrer">
                  <b>{s.title}</b>
                  <span className="muted"> — {s.basis}</span>
                </a>
              ))}
            </div>
          ))
        ) : (
          <p className="muted">No configured risk factor matched this clause.</p>
        )}
      </Section>

      <Section title="AI explanation" aside={<span className="muted small">Restricted local LLM</span>}>
        <p className="prewrap">{finding.explanation}</p>
        <p className="muted small">
          Ollama only explains supplied evidence; it does not invent risk factors or make independent legal claims.
        </p>
      </Section>

      <div className="reco">
        <div className="reco-label">Recommendation</div>
        <div>{finding.recommendation}</div>
      </div>

      {finding.issues?.length > 0 && (
        <Section title="Evidence observations">
          {finding.issues.map((issue, i) => (
            <Note key={i} tone="unknown">
              <b>
                {titleCase(issue.category)} · {titleCase(issue.severity)}
              </b>
              <div>{issue.description}</div>
            </Note>
          ))}
        </Section>
      )}

      <Section title={`Supporting evidence (${retrieval.length})`}>
        {retrieval.length ? (
          retrieval.map((hit, i) => (
            <div key={i} className="evidence">
              <div className="evidence-head">
                <b>{hit.title}</b>
                <span className="mono small">{hit.relevance.toFixed(3)}</span>
              </div>
              <div className="small">{hit.snippet}</div>
              {hit.graph_path?.length > 0 && (
                <div className="path">
                  {hit.graph_path.map((p, j) => (
                    <span key={j}>{p}</span>
                  ))}
                </div>
              )}
            </div>
          ))
        ) : (
          <p className="muted">No supporting legal knowledge was retrieved.</p>
        )}
      </Section>
    </div>
  );
}

function ClausesView({ doc, findings }) {
  const items = doc.clauses.filter((c) => findings[c.clause_id]);
  const [selected, setSelected] = useState(items[0]?.clause_id);
  const [filter, setFilter] = useState("all");
  const shown = items.filter((c) => filter === "all" || statusOf(findings[c.clause_id]) === filter);
  const current = items.find((c) => c.clause_id === selected) || items[0];

  if (!items.length) return <div className="empty">No clause findings.</div>;

  return (
    <div className="master-detail">
      <div className="clause-list">
        <div className="filters">
          {[
            ["all", "All"],
            ["risk", "Attention"],
            ["unknown", "Unclear"],
            ["ok", "Clear"],
          ].map(([k, l]) => (
            <button key={k} className={filter === k ? "on" : ""} onClick={() => setFilter(k)}>
              {l}
            </button>
          ))}
        </div>
        {shown.map((c) => {
          const f = findings[c.clause_id];
          const flagged = ["flag", "reanalyze"].includes(f.review_decision);
          return (
            <button
              key={c.clause_id}
              className={`clause-item ${current?.clause_id === c.clause_id ? "active" : ""}`}
              onClick={() => setSelected(c.clause_id)}
            >
              <span className={`dot dot-${statusOf(f)}`} />
              <span className="clause-item-text">
                <span className="clause-item-title">{c.heading || `Clause ${c.index + 1}`}</span>
                <span className="muted small">{f.clause_type}</span>
              </span>
              {flagged && <span className="flag">🚩</span>}
            </button>
          );
        })}
        {!shown.length && <div className="empty small">Nothing in this filter.</div>}
      </div>
      {current && <ClauseDetail key={current.clause_id} clause={current} finding={findings[current.clause_id]} />}
    </div>
  );
}

function VersionView({ version }) {
  const { diff, comparison, recommendation } = version;
  const count = (t) => diff.entries.filter((e) => e.change_type === t).length;
  const verdict = {
    old: ["Previous version recommended", "unknown"],
    new: ["New version recommended", "ok"],
    equivalent: ["Versions roughly equivalent", "neutral"],
  }[recommendation.safer_version] || ["Recommendation", "neutral"];

  return (
    <div className="stack">
      <Note tone={verdict[1]}>
        <div className="verdict">{verdict[0]}</div>
        <div>{recommendation.reasoning}</div>
        <div className="muted small" style={{ marginTop: 6 }}>
          Matched a stored contract at {pct(version.match_score, 0)} similarity · reasoning source:{" "}
          {recommendation.reasoning_source}
        </div>
      </Note>
      <div className="stats">
        <Stat label="Added" value={count("added")} tone="ok" />
        <Stat label="Removed" value={count("removed")} tone="risk" />
        <Stat label="Modified" value={count("modified")} tone="unknown" />
        <Stat label="Unchanged" value={count("unchanged")} />
      </div>
      <div className="stats">
        <Stat label="Increased risk" value={comparison.increased_count} tone="risk" />
        <Stat label="Reduced risk" value={comparison.reduced_count} tone="ok" />
        <Stat label="Unchanged risk" value={comparison.unchanged_count} />
      </div>
      <p className="muted small">
        This compares whether risk language changed, not whether removing a clause was itself favorable or
        unfavorable.
      </p>
      <div className="diff-list">
        {diff.entries.map((e, i) => (
          <div key={i} className={`diff diff-${e.change_type}`}>
            <div className="diff-head">
              <Badge tone={{ added: "ok", removed: "risk", modified: "unknown", unchanged: "neutral" }[e.change_type]}>
                {titleCase(e.change_type)}
              </Badge>
              <b>{e.new_heading || e.old_heading || "(untitled clause)"}</b>
            </div>
            {(e.change_type === "modified" || e.change_type === "removed") && (
              <div className="diff-old small">{e.old_text_snippet}</div>
            )}
            {(e.change_type === "modified" || e.change_type === "added") && (
              <div className="diff-new small">{e.new_text_snippet}</div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

function DataView({ doc, report }) {
  const [which, setWhich] = useState("report");
  const data = which === "report" ? report : doc;
  return (
    <div className="stack">
      <div className="data-bar">
        <div className="seg">
          <button className={which === "report" ? "on" : ""} onClick={() => setWhich("report")}>
            Review report
          </button>
          <button className={which === "doc" ? "on" : ""} onClick={() => setWhich("doc")}>
            Parsed contract
          </button>
        </div>
        <button
          className="btn btn-primary"
          onClick={() =>
            downloadJson(data, `${stem(doc.filename)}_${which === "report" ? "review_report" : "parsed"}.json`)
          }
        >
          ↓ Download JSON
        </button>
      </div>
      <pre className="json">{JSON.stringify(data, null, 2)}</pre>
    </div>
  );
}

function Results({ document: doc, report, version }) {
  const findings = Object.fromEntries(report.clause_findings.map((f) => [f.clause_id, f]));
  const all = report.clause_findings;
  const n = (s) => all.filter((f) => statusOf(f) === s).length;
  const [tab, setTab] = useState("clauses");

  return (
    <div className="results">
      <div className="summary-card">
        <div className="summary-top">
          <div>
            <div className="muted small">📄 {doc.filename}</div>
            <h1 className="summary-title">{titleCase(report.overall_assessment)}</h1>
          </div>
          <div className="meta-chips">
            <span>{doc.page_count} pages</span>
            <span>{doc.clauses.length} clauses</span>
            <span>{doc.source_language}</span>
          </div>
        </div>
        <p className="summary-text">{report.summary}</p>
        <div className="stats">
          <Stat label="Attention required" value={n("risk")} tone="risk" />
          <Stat label="Insufficient evidence" value={n("unknown")} tone="unknown" />
          <Stat label="No risk factor found" value={n("ok")} tone="ok" />
        </div>
      </div>

      <nav className="tabs">
        <button className={tab === "clauses" ? "on" : ""} onClick={() => setTab("clauses")}>
          Clauses
        </button>
        {version && (
          <button className={tab === "version" ? "on" : ""} onClick={() => setTab("version")}>
            Version comparison <span className="pill">{pct(version.match_score, 0)}</span>
          </button>
        )}
        <button className={tab === "data" ? "on" : ""} onClick={() => setTab("data")}>
          Raw data
        </button>
      </nav>

      {tab === "clauses" && <ClausesView doc={doc} findings={findings} />}
      {tab === "version" && version && <VersionView version={version} />}
      {tab === "data" && <DataView doc={doc} report={report} />}
    </div>
  );
}

/* ---------------- App ---------------- */

export default function App() {
  const [history, setHistory] = useState([]);
  const [drawer, setDrawer] = useState(false);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [fileName, setFileName] = useState("");

  const loadHistory = () =>
    fetch("/api/history")
      .then((r) => (r.ok ? r.json() : []))
      .then(setHistory)
      .catch(() => {});

  useEffect(() => {
    loadHistory();
  }, []);

  const analyze = async (file) => {
    setLoading(true);
    setError(null);
    setResult(null);
    setFileName(file.name);
    const form = new FormData();
    form.append("file", file);
    try {
      const res = await fetch("/api/analyze", { method: "POST", body: form });
      const text = await res.text();
      let data = null;
      try {
        data = text ? JSON.parse(text) : null;
      } catch {
        /* non-JSON response */
      }
      if (!res.ok || !data) {
        throw new Error(
          data?.detail ||
            (text
              ? `Request failed (${res.status}): ${text.slice(0, 300)}`
              : `No response from backend (${res.status}). Is the API server running? Start it with: uvicorn server:app --reload --port 8000`)
        );
      }
      setResult(data);
      loadHistory();
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      <Header
        onHistory={() => setDrawer(true)}
        historyCount={history.length}
        hasResult={!!result}
        onReset={() => {
          setResult(null);
          setError(null);
        }}
      />
      <HistoryDrawer open={drawer} onClose={() => setDrawer(false)} history={history} />
      <main className="container">
        {result ? (
          <Results {...result} />
        ) : (
          <Landing onFile={analyze} loading={loading} error={error} fileName={fileName} />
        )}
      </main>
    </>
  );
}
