"""Supervisor Agent — LangGraph orchestration layer.

Replaces the old fixed sequence (classify -> retrieve -> analyze ->
uncertainty) with conditional routing:

  classify -> retrieve -> analyze
  analyze  -> [weak evidence, 1 retry]   -> retrieve (broadened) -> analyze
  analyze  -> [attention_required?]      -> adversarial review -> uncertainty
  analyze  -> [otherwise]                -> uncertainty
  uncertainty -> [decision == reanalyze, first time] -> reanalyze
  reanalyze   -> [risk surfaced?]        -> adversarial review -> uncertainty
  reanalyze   -> [otherwise]             -> uncertainty
  uncertainty -> finalize -> ClauseFinding

"reanalyze" is an action, not just a label: the clause's risk is re-assessed
under the classifier's second-choice category with a category-free search of
the whole knowledge base, and the more cautious of the two assessments is
kept. Bounded to one re-analysis per clause.
"""
from __future__ import annotations

from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END

from interfaces import Clause, Classification, RetrievalResult, UncertaintyDecision, ClauseFinding
from classifier import classify_clause
from legal_kb import retrieve
from risk_analysis import analyze_clause
from adversarial_review import review_finding
from uncertainty_bandit import uncertainty_agent, LOW_RELEVANCE_THRESHOLD

RELEVANCE_RETRY_THRESHOLD = LOW_RELEVANCE_THRESHOLD
MAX_RETRIEVAL_ATTEMPTS = 1  # bounded retry: 1 retry beyond the initial attempt
MAX_REANALYSES = 1

RISK_RANK = {"no_configured_risk_found": 0, "insufficient_evidence": 1, "attention_required": 2}


class ClauseState(TypedDict, total=False):
    clause: Clause
    cls: Classification
    retr: RetrievalResult
    retrieval_attempts: int
    reanalysis_attempts: int
    partial: dict
    adversarial: dict | None
    decision: UncertaintyDecision
    finding: ClauseFinding


def _classify_node(state: ClauseState) -> dict:
    return {"cls": classify_clause(state["clause"])}


def _retrieve_node(state: ClauseState) -> dict:
    attempts = state.get("retrieval_attempts", 0)
    clause = state["clause"]
    cls = state["cls"]
    if attempts == 0:
        retr = retrieve(clause.text_en, cls.predicted_type, top_k=5)
    else:
        # Broaden the search: drop graph-neighborhood narrowing, widen top_k.
        # A literal retry with identical args would be a no-op since
        # retrieval is deterministic.
        retr = retrieve(clause.text_en, None, top_k=10)
    return {"retr": retr, "retrieval_attempts": attempts + 1}


def _analyze_node(state: ClauseState) -> dict:
    partial = analyze_clause(state["clause"], state["cls"], state["retr"])
    return {"partial": partial}


def _adversarial_node(state: ClauseState) -> dict:
    adversarial = review_finding(state["clause"], state["cls"], state["retr"], state["partial"])
    return {"adversarial": adversarial}


def _uncertainty_node(state: ClauseState) -> dict:
    partial = state["partial"]
    adversarial = state.get("adversarial")
    agreement = 1.0
    if adversarial is not None:
        agreement = 0.0 if adversarial.get("still_concerning") else 1.0

    features = {
        "clause_id": state["clause"].clause_id,
        "confidence": state["cls"].confidence,
        "relevance": partial["best_relevance"],
        "risk_assessment": partial["risk_assessment"],
        "agreement": agreement,
    }
    return {"decision": uncertainty_agent(features)}


def _reanalyze_node(state: ClauseState) -> dict:
    clause = state["clause"]
    cls = state["cls"]
    original = state["partial"]

    # Broadened evidence: a category-free search over the whole knowledge base.
    broadened = retrieve(clause.text_en, None, top_k=10)

    # Low confidence usually means the classifier is torn between categories,
    # and the wrong category sends the risk check to the wrong framework — so
    # re-assess under the runner-up category when there is one.
    alternatives = [t for t in cls.top_k if t.type != cls.predicted_type]
    if alternatives:
        alt = alternatives[0]
        alt_cls = cls.model_copy(update={"predicted_type": alt.type, "confidence": alt.score})
        candidate = analyze_clause(clause, alt_cls, broadened)
        category_checked, category_score = alt.type, alt.score
    else:
        candidate = analyze_clause(clause, cls, broadened)
        category_checked, category_score = cls.predicted_type, cls.confidence

    # Fail-safe: adopt the second look only if it surfaces *more* risk.
    adopted = RISK_RANK[candidate["risk_assessment"]] > RISK_RANK[original["risk_assessment"]]
    chosen = dict(candidate if adopted else original)
    chosen["clause_type"] = cls.predicted_type
    chosen["confidence"] = round(float(cls.confidence), 4)
    chosen["evidence"] = {
        **chosen.get("evidence", {}),
        "reanalysis": {
            "performed": True,
            "trigger_decision": state["decision"].reason,
            "category_checked": category_checked,
            "category_checked_score": round(float(category_score), 4),
            "evidence_search": "category-free search over the full knowledge base (top 10)",
            "original_assessment": original["risk_assessment"],
            "reanalysis_assessment": candidate["risk_assessment"],
            "reanalysis_best_relevance": candidate["best_relevance"],
            "adopted": adopted,
            "outcome": (
                f"Re-assessment under '{category_checked}' surfaced more risk "
                f"({candidate['risk_assessment']}); it was adopted."
                if adopted else
                f"Re-assessment under '{category_checked}' did not surface more risk "
                f"({candidate['risk_assessment']}); the original assessment was kept."
            ),
        },
    }
    return {
        "partial": chosen,
        "retr": broadened if adopted else state["retr"],
        "reanalysis_attempts": state.get("reanalysis_attempts", 0) + 1,
    }


def _finalize_node(state: ClauseState) -> dict:
    partial = dict(state["partial"])
    decision = state["decision"]
    reason = decision.reason
    if state.get("reanalysis_attempts", 0) > 0:
        reason += (
            " Automated re-analysis already ran once (see evidence.reanalysis); "
            + ("the clause still needs human review." if decision.action != "accept"
               else "it resolved the uncertainty.")
        )
    retrieval_attempts = state.get("retrieval_attempts", 1)
    partial["evidence"] = {
        **partial.get("evidence", {}),
        # The path the Supervisor actually took for this clause, so every
        # agentic branch is visible in the output, not just the final verdict.
        "agent_route": {
            "retrieval_attempts": retrieval_attempts,
            "broadened_retrieval_retry": retrieval_attempts > 1,
            "adversarial_review": state.get("adversarial") is not None,
            "reanalysis": state.get("reanalysis_attempts", 0) > 0,
            "final_decision": decision.action,
        },
    }
    finding = ClauseFinding(
        **partial,
        review_decision=decision.action,
        review_reason=reason,
        adversarial_review=state.get("adversarial"),
    )
    return {"finding": finding}


def _route_after_analyze(state: ClauseState) -> str:
    partial = state["partial"]
    attempts = state.get("retrieval_attempts", 0)

    if partial["best_relevance"] < RELEVANCE_RETRY_THRESHOLD and attempts <= MAX_RETRIEVAL_ATTEMPTS:
        return "retry"
    if partial["risk_assessment"] == "attention_required":
        return "adversarial"
    return "uncertainty"


def _route_after_uncertainty(state: ClauseState) -> str:
    if state["decision"].action == "reanalyze" and state.get("reanalysis_attempts", 0) < MAX_REANALYSES:
        return "reanalyze"
    return "finalize"


def _route_after_reanalyze(state: ClauseState) -> str:
    if state["partial"]["risk_assessment"] == "attention_required" and state.get("adversarial") is None:
        return "adversarial"
    return "uncertainty"


def _build_graph() -> StateGraph:
    g = StateGraph(ClauseState)
    g.add_node("classify", _classify_node)
    g.add_node("retrieve", _retrieve_node)
    g.add_node("analyze", _analyze_node)
    g.add_node("adversarial", _adversarial_node)
    g.add_node("uncertainty", _uncertainty_node)
    g.add_node("reanalyze", _reanalyze_node)
    g.add_node("finalize", _finalize_node)

    g.add_edge(START, "classify")
    g.add_edge("classify", "retrieve")
    g.add_edge("retrieve", "analyze")
    g.add_conditional_edges(
        "analyze",
        _route_after_analyze,
        {"retry": "retrieve", "adversarial": "adversarial", "uncertainty": "uncertainty"},
    )
    g.add_edge("adversarial", "uncertainty")
    g.add_conditional_edges(
        "uncertainty",
        _route_after_uncertainty,
        {"reanalyze": "reanalyze", "finalize": "finalize"},
    )
    g.add_conditional_edges(
        "reanalyze",
        _route_after_reanalyze,
        {"adversarial": "adversarial", "uncertainty": "uncertainty"},
    )
    g.add_edge("finalize", END)
    return g


_compiled = _build_graph().compile()


def run_clause(clause: Clause) -> ClauseFinding:
    result = _compiled.invoke({"clause": clause, "retrieval_attempts": 0, "reanalysis_attempts": 0})
    return result["finding"]


if __name__ == "__main__":
    # Routing self-test. Classification, retrieval, risk analysis, adversarial
    # review are replaced with deterministic fakes (risk analysis is LLM-driven,
    # so a real call would make routing assertions nondeterministic), and the
    # RL agent runs for real with exploration off and a throwaway weights
    # file. No trained model, FAISS index, or Ollama needed.
    import os
    import tempfile
    from unittest.mock import patch
    from interfaces import TypeScore, Explanation, RetrievalHit
    from uncertainty_bandit import UncertaintyBanditAgent

    test_agent = UncertaintyBanditAgent(weights_file=os.path.join(tempfile.mkdtemp(), "weights.json"))
    test_agent.epsilon = 0.0

    def fake_classify_factory(predicted_type, confidence, alternative=None):
        def _fake(clause):
            top_k = [TypeScore(type=predicted_type, score=confidence)]
            if alternative:
                top_k.append(TypeScore(type=alternative[0], score=alternative[1]))
            return Classification(
                clause_id=clause.clause_id, predicted_type=predicted_type, confidence=confidence,
                top_k=top_k,
                explanation=Explanation(method="attention", salient_tokens=[], rationale="fake"),
                low_conf_baseline=confidence < 0.60,
            )
        return _fake

    def fake_retrieve_factory(narrowed_relevance, broadened_relevance=None, log=None):
        def _fake(query, clause_type, top_k=5):
            if log is not None:
                log.append({"clause_type": clause_type, "top_k": top_k})
            rel = narrowed_relevance if clause_type is not None or broadened_relevance is None else broadened_relevance
            return RetrievalResult(query=query, results=[
                RetrievalHit(source_id="src_fake", title="Fake source", snippet="fake", relevance=rel,
                             graph_path=[clause_type or "unknown"])
            ])
        return _fake

    def fake_analyze_factory(risk_by_type, log=None):
        def _fake(clause, cls, retr):
            if log is not None:
                log.append(cls.predicted_type)
            return {
                "clause_id": clause.clause_id, "clause_type": cls.predicted_type,
                "risk_assessment": risk_by_type.get(cls.predicted_type, "no_configured_risk_found"),
                "risk_factors": [], "issues": [], "recommendation": "fake",
                "confidence": cls.confidence,
                "best_relevance": max((h.relevance for h in retr.results), default=0.0),
                "text_comparison": {}, "evidence": {}, "explanation": "fake",
            }
        return _fake

    def fake_review(clause, cls, retr, partial):
        return {"still_concerning": True, "note": "fake", "missed_angle": None, "source": "fake"}

    RISKS = {"Indemnifications": "attention_required", "Representations": "attention_required"}
    clause = Clause(clause_id="c_001", index=0, heading=None, text_original="", text_en="fake clause",
                    page=1, char_span=(0, 0))

    def run(classify, retrieve_fn, analyze_fn):
        # Patch via __name__ so the patch lands on the module the compiled
        # graph's nodes read from ("__main__" when run as a script).
        with patch(f"{__name__}.classify_clause", classify), \
             patch(f"{__name__}.retrieve", retrieve_fn), \
             patch(f"{__name__}.analyze_clause", analyze_fn), \
             patch(f"{__name__}.review_finding", fake_review), \
             patch(f"{__name__}.uncertainty_agent", lambda f: test_agent.decide(f["clause_id"], f)):
            return run_clause(clause)

    print("--- Case 1: clean, confident, strong evidence -> accept, no extra steps ---")
    f = run(fake_classify_factory("Governing Laws", 0.95), fake_retrieve_factory(0.92), fake_analyze_factory(RISKS))
    print(f"  decision={f.review_decision} adversarial={f.adversarial_review is not None} reanalysis={'reanalysis' in f.evidence}")
    assert f.review_decision == "accept" and f.adversarial_review is None and "reanalysis" not in f.evidence
    print("  OK\n")

    print("--- Case 2: risky, confident -> adversarial review -> flag ---")
    f = run(fake_classify_factory("Indemnifications", 0.92), fake_retrieve_factory(0.92), fake_analyze_factory(RISKS))
    print(f"  decision={f.review_decision} adversarial={f.adversarial_review is not None}")
    assert f.review_decision == "flag" and f.adversarial_review is not None and "reanalysis" not in f.evidence
    print("  OK\n")

    print("--- Case 3: low confidence, runner-up category is risky -> re-analysis surfaces it ---")
    analyze_log = []
    f = run(fake_classify_factory("Authority", 0.34, alternative=("Representations", 0.20)),
            fake_retrieve_factory(0.92), fake_analyze_factory(RISKS, log=analyze_log))
    r = f.evidence["reanalysis"]
    print(f"  analyze() calls: {analyze_log}")
    print(f"  reanalysis: checked={r['category_checked']} adopted={r['adopted']} -> {r['reanalysis_assessment']}")
    print(f"  final: risk={f.risk_assessment} adversarial={f.adversarial_review is not None} decision={f.review_decision}")
    print(f"  reason: {f.review_reason}")
    assert analyze_log == ["Authority", "Representations"], "expected exactly one re-analysis, under the runner-up category"
    assert r["adopted"] and f.risk_assessment == "attention_required"
    assert f.adversarial_review is not None, "a risk surfaced by re-analysis must still get adversarial review"
    assert f.clause_type == "Authority" and f.review_decision == "reanalyze"
    print("  OK — re-analysis actually ran, surfaced the risk, and stopped after one pass.\n")

    print("--- Case 4: low confidence, runner-up category is clean -> original kept ---")
    analyze_log = []
    f = run(fake_classify_factory("Authority", 0.34, alternative=("Notices", 0.20)),
            fake_retrieve_factory(0.92), fake_analyze_factory(RISKS, log=analyze_log))
    r = f.evidence["reanalysis"]
    print(f"  analyze() calls: {analyze_log}  adopted={r['adopted']}  final risk={f.risk_assessment}")
    assert analyze_log == ["Authority", "Notices"] and not r["adopted"]
    assert f.risk_assessment == "no_configured_risk_found" and f.review_decision == "reanalyze"
    print("  OK\n")

    print("--- Case 5: weak evidence, broadened retry finds strong evidence -> accept ---")
    retrieve_log = []
    f = run(fake_classify_factory("Governing Laws", 0.95),
            fake_retrieve_factory(0.70, broadened_relevance=0.90, log=retrieve_log), fake_analyze_factory(RISKS))
    print(f"  retrieve() calls: {retrieve_log}  final relevance={f.best_relevance} decision={f.review_decision}")
    assert retrieve_log == [{"clause_type": "Governing Laws", "top_k": 5}, {"clause_type": None, "top_k": 10}]
    assert f.best_relevance == 0.90 and f.review_decision == "accept"
    assert f.evidence["agent_route"]["broadened_retrieval_retry"] is True
    print(f"  agent_route: {f.evidence['agent_route']}")
    print("  OK — the retry fired, actually rescued the clause, and is visible in the output.\n")

    print("--- Case 6: weak evidence everywhere -> retry once, re-analyze once, then stop ---")
    retrieve_log = []
    f = run(fake_classify_factory("Governing Laws", 0.95),
            fake_retrieve_factory(0.70, log=retrieve_log), fake_analyze_factory(RISKS))
    print(f"  retrieve() calls: {len(retrieve_log)}  decision={f.review_decision}")
    assert len(retrieve_log) == 3, f"expected initial + 1 retry + 1 re-analysis search, got {len(retrieve_log)}"
    assert f.review_decision == "reanalyze" and f.evidence["reanalysis"]["performed"]
    print("  OK — bounded: no infinite loops.\n")

    print("ALL CHECKS PASSED.")
