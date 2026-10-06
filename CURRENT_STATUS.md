# Legal Contract Review — Current Project Status

_Last updated: 2026-08-28_

## 1. Project Flow

```text
Contract (PDF/DOCX)
        │
        ▼
┌──────────────────────┐
│ 1. Document Parsing   │
│ PDF/DOCX → Clauses    │
└──────────┬────────────┘
           ▼
┌──────────────────────────────┐
│         SUPERVISOR AGENT       │   ← LangGraph, conditional routing
│  (not a fixed sequence)        │
└──────────┬────────────────────┘
           ▼
┌──────────────────────┐
│ 2. Classification      │
│ LegalBERT + LEDGAR     │
│ all 100 categories     │
└──────────┬────────────┘
           ▼
┌──────────────────────┐
│ 3. Retrieval           │
│ FAISS + GraphRAG       │
│ Legal KB (33,328 recs) │
└──────────┬────────────┘
           ▼
┌──────────────────────┐
│ 4. Risk Analysis       │
│ Evidence-based         │
│ multi-factor review    │
└──────────┬────────────┘
           ▼
     ┌─────┴─────┐
     ▼           ▼
┌─────────┐ ┌──────────────┐
│Adversarial│ │ (skip if no  │
│ Review    │ │ risk found)  │
│(if flagged│ │              │
└─────┬─────┘ └──────┬───────┘
      └───────┬───────┘
              ▼
┌──────────────────────┐
│ 5. Uncertainty Agent   │
│ Contextual bandit (RL) │
│ accept / flag / reanalyze │
└──────────┬────────────┘
           ▼
┌──────────────────────┐
│ 6. Report / UI         │
│ + Database Storage      │
└──────────┬────────────┘
           ▼
   (if content-matched to a
    stored prior analysis)
           ▼
┌──────────────────────┐
│ Version Workflow        │
│ Diff → Risk Comparison  │
│ → Version Recommendation│
└──────────────────────┘
```

## 2. Current Progress

| Stage                    | Status  | Main Technology                                    |
| ------------------------ | ------- | --------------------------------------------------- |
| Document Parsing         | ✅ Done | pdfplumber / python-docx + Ollama (segmentation)     |
| Supervisor Orchestration | ✅ Done | LangGraph — conditional routing, not a fixed sequence |
| Clause Classification    | ✅ Done | Fine-tuned LegalBERT, 20 LEDGAR categories           |
| Knowledge Retrieval      | ✅ Done | FAISS + GraphRAG, 33,328-record KB                   |
| Risk Analysis            | ✅ Done | Multi-factor analysis + restricted Ollama            |
| Adversarial Review       | ✅ Done | Restricted Ollama, fails closed on outage            |
| Uncertainty Agent        | ✅ Done | Contextual bandit (RL), learns online from feedback  |
| Database Storage         | ✅ Done | SQLite, every analysis persisted                     |
| Version Workflow         | ✅ Done | Content-similarity version matching + diff + recommendation |
| Report / UI              | ✅ Done | Streamlit — clause detail, version comparison, history |
| Real end-to-end run      | ✅ Verified | Ran successfully on `data/test_contract.docx` with the real trained model and real KB |

**Everything in the original proposed architecture is now implemented and verified**, not just stubbed. The only remaining item is a structural refactor (see §6) — not new capability.

## 3. What Each Stage Does

### 1. Document Parsing ✅
PDF / DOCX → extract text → detect language → LLM-assisted clause segmentation (verbatim, no rewriting) → translate non-English clauses. Produces a `Document` with page numbers and character spans for every clause.

### Supervisor Agent (Agentic Orchestration) ✅
Implemented in `supervisor.py` as a LangGraph `StateGraph`. Per clause: classify → retrieve → analyze, then **conditionally** routes:
- `risk_assessment == attention_required` → Adversarial Review, then Uncertainty
- weak retrieval relevance (< 0.45), first attempt only → retry retrieval with a broadened query (no graph narrowing, larger top_k), bounded to one retry
- otherwise → straight to Uncertainty (which always runs — it must produce the required `review_decision`)

This is what makes it agentic rather than a fixed pipeline: the routing decision depends on what the analysis itself finds.

### 2. Clause Classification ✅
Model: LegalBERT (`nlpaueb/legal-bert-base-uncased`), fine-tuned on **all 100 original LEDGAR categories** — the unmapped, original label set, not a reduced subset.

**Latest training run** (`legalbert_finetuned/metrics.json`): **84.6% eval accuracy, 0.773 macro-F1**, 3 epochs, 50,752 training examples (`train_classifier.py`'s `load_ledgar()` reproduces this exact example count). Trained on a Colab T4 GPU (~30 min) — the full 100-category/3-epoch config isn't practical on this CPU-only dev machine (~30+ hrs estimated); `train_classifier.py` is unchanged in its data/config and can be run on either, it'll just be much slower on CPU. The training notebook itself hasn't been added to the repo yet — worth doing for reproducibility.

Output per clause: predicted type + confidence + top-3 alternatives + attention-based salient tokens.

**Verified real end-to-end run** on `data/test_contract.docx` after swapping in this model: pipeline completes correctly, `risk_analysis.py`'s configured trigger families (keyed to clause text content, not the classifier label) still fire correctly regardless of the exact predicted category name.

### 3. Knowledge Retrieval ✅
Knowledge base: LEDGAR (27,328) + CUAD (3,000) + MAUD (3,000) = **33,328 records**, built by `build_knowledge_base.py`. FAISS `IndexFlatIP` over LegalBERT embeddings (`build_faiss_index.py`) + a lightweight GraphRAG category graph for candidate narrowing. Verified retrieving real, on-topic precedent clauses at 0.90+ relevance on a real contract.

### 4. Risk Analysis ✅
No arbitrary 0–100 score. Matches configured, source-cited trigger phrases (indemnification, limitation of liability, termination, confidentiality, payment terms) against the clause text, plus structural observations (e.g. missing scope-limit language). A restricted Ollama call explains the evidence only — cannot invent facts, risks, or citations.

### Adversarial Review ✅
`adversarial_review.py`. Runs only when Risk Analysis flags `attention_required`. Challenges the finding using only the already-supplied evidence — may point out a genuine counter-reading present in the clause text, but cannot introduce a new risk factor. **Fails closed**: if Ollama is unreachable or slow, the original flag is preserved (`still_concerning=True`), never silently cleared.

### 5. Uncertainty Agent ✅ (contextual bandit)
`uncertainty_bandit.py`. A real, lightweight RL formulation — per-action linear scorer with epsilon-greedy exploration over `accept` / `flag` / `reanalyze`, using confidence, retrieval relevance, risk assessment, and adversarial-review agreement as features. Initialized to exactly reproduce the old hand-written heuristic on day one (verified: 90/90 test cases match), then learns online via `.update()` as a reviewer confirms/corrects decisions. Weights persist to `data/bandit_weights.json` (gitignored, self-bootstrapping).

### 6. Report Generation + Database Storage ✅
`stubs.build_report` assembles the qualitative document-level report. `storage.py` persists every analysis to SQLite (`data/contract_review.db`, gitignored) so later uploads can be compared against it.

### Version Workflow ✅
Opt-in — uploads are **not** automatically compared against stored contracts. The user picks a previously analyzed contract ("Compare with a previous version") before uploading; only then is the new upload compared against it (`pipeline.run_with_versioning(..., compare_with_doc_id=...)`, API form field `compare_with`):
- `version_diff.py` — clause-level added/removed/modified/unchanged, via global-greedy content-similarity matching (not naive per-row greedy)
- `risk_comparison.py` — classifies each matched clause's risk change (increased/reduced/unchanged)
- `version_recommendation.py` — deterministic safer-version decision (never LLM-gated) + restricted-Ollama reasoning text with a templated fallback

Rendered in the Streamlit UI as a "🔄 Version Comparison" section.

## 4. Real End-to-End Verification

Ran the actual training + full pipeline (not synthetic/monkeypatched tests), against real data throughout:
1. Built the real 33,328-record knowledge base and FAISS index.
2. Fine-tuned LegalBERT on all 100 LEDGAR categories (84.6% accuracy, 0.773 macro-F1).
3. Ran `python pipeline.py data/test_contract.docx` against the real model, real KB, and real Supervisor graph — completed successfully. The Supervisor correctly routed the flagged indemnification clause through Adversarial Review while skipping it entirely for the clean termination clause; the bandit chose `flag` and `accept` respectively, both correct.
4. Confirmed Streamlit boots and serves cleanly with the real pipeline wired in.

Along the way, found and fixed two real bugs: `pipeline.py`'s CLI print crashed on Unicode characters in real LLM-generated text (Windows console defaults to `cp1252`) — fixed by forcing UTF-8 stdout. And a minor explainability gap in the uncertainty bandit: when its epsilon-greedy exploration (10% of calls) picks an action the evidence didn't actually favor, the returned `reason` now says so explicitly instead of reusing the evidence-based justification text.

## 5. Architecture Reference

- `interfaces.py` — frozen Pydantic data contracts (all Phase 1/2 additions are additive-only).
- `parsing.py`, `classifier.py`, `legal_kb.py`, `risk_analysis.py` — stages 1–4.
- `supervisor.py` — LangGraph orchestration.
- `adversarial_review.py`, `uncertainty_bandit.py` — the two RL/agentic additions.
- `storage.py`, `version_diff.py`, `risk_comparison.py`, `version_recommendation.py` — Database Storage + Version Workflow.
- `text_similarity.py` — shared content-similarity primitive used by storage + version diffing.
- `pipeline.py` — `run()` (unchanged, backward compatible) and `run_with_versioning()` (the full path, used by `app.py`).
- `app.py` — Streamlit UI.

## 6. What Is Left

🔄 **Structural only — not new capability:**
- Split Risk Indicator Extraction, Compliance, and Recommendation Generation out of `risk_analysis.py` into their own modules, to match the proposal's architecture diagram literally (they're currently functionally covered but bundled together).
- Update the report's UML/architecture diagrams to reflect the now-implemented system instead of the originally-proposed one.

🔄 **Operational:**
- Nothing has been committed to git yet — everything above is local, uncommitted work.
- Only tested on one real contract so far (`data/test_contract.docx`); broader testing across varied real contracts would strengthen the report's evaluation section.
- The RL bandit's online learning has not yet been exercised with real human feedback (no Streamlit control wired up for it yet — currently only provable via the standalone `uncertainty_bandit.py` demo).

## 7. How to Run the Current System

See `README.md` / chat for exact commands. Summary: `pip install -r requirements.txt`, ensure Ollama is running with `qwen2.5:3b`, ensure `legalbert_finetuned/` and `data/knowledge_base/` exist (already built on this machine), then `python pipeline.py <file>` or `streamlit run app.py`.
