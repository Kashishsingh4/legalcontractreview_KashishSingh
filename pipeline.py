"""End-to-end legal contract review pipeline."""
from __future__ import annotations

import json
import sys

from interfaces import ClauseFinding, ReviewReport, Document, VersionAnalysisResult
import stubs
import supervisor
import storage
import version_diff
import risk_comparison
import version_recommendation


def run(file_path: str = "data/sample_document.json") -> tuple[Document, ReviewReport]:
    # Stage 1 — parse.
    if file_path.lower().endswith(".json"):
        with open(file_path, "r", encoding="utf-8") as f:
            doc: Document = Document.model_validate(json.load(f))
    else:
        doc = stubs.parse_document(file_path)

    # Stages 2-5 — Supervisor Agent: classification, retrieval, risk analysis,
    # conditional adversarial review, and uncertainty verification, routed
    # per clause instead of a fixed sequence.
    findings: list[ClauseFinding] = [supervisor.run_clause(clause) for clause in doc.clauses]

    # Stage 6 — report.
    return doc, stubs.build_report(doc, findings)


def run_with_versioning(
    file_path: str = "data/sample_document.json",
    db_path: str = storage.DB_PATH,
    compare_with_doc_id: str | None = None,
) -> tuple[Document, ReviewReport, VersionAnalysisResult | None]:
    """Same as run(), plus: Database Storage + Version Workflow. Version
    comparison is opt-in: only when the caller names a previously stored
    analysis (`compare_with_doc_id`) are the clauses diffed, risk compared,
    and the safer version recommended. Uploads are never auto-matched
    against stored contracts. Every analysis is persisted so it can be
    chosen for comparison later.

    `db_path` is injectable (mirrors uncertainty_bandit.py's `weights_file`
    parameter) so tests can point storage at a throwaway file instead of the
    real database."""
    doc, report = run(file_path)

    version: VersionAnalysisResult | None = None
    if compare_with_doc_id:
        prior = storage.load_analysis(compare_with_doc_id, db_path=db_path)
        if prior is None:
            raise ValueError(f"Previous analysis '{compare_with_doc_id}' was not found.")
        old_doc, old_report = prior
        diff = version_diff.diff_documents(old_doc, doc)
        comparison = risk_comparison.compare_versions(diff, old_report, report)
        recommendation = version_recommendation.recommend_version(comparison)
        version = VersionAnalysisResult(
            matched_doc_id=compare_with_doc_id, match_score=None,
            diff=diff, comparison=comparison, recommendation=recommendation,
        )

    # Save AFTER loading the chosen prior, so a new upload can't be compared with itself.
    storage.save_analysis(doc, report, db_path=db_path)

    return doc, report, version


if __name__ == "__main__":
    # Windows consoles default stdout to cp1252, which can't encode
    # characters that show up in real LLM-generated text (e.g. U+2011
    # non-breaking hyphen) or clause text from real contracts — force UTF-8
    # so the CLI doesn't crash on otherwise-successful runs.
    sys.stdout.reconfigure(encoding="utf-8")

    path = sys.argv[1] if len(sys.argv) > 1 else "data/sample_document.json"
    _doc, report = run(path)
    print(report.model_dump_json(indent=2))
    print("\nOK — pipeline completed without numeric risk scoring.")



# """The walking skeleton: runs all stages in order using whatever implementation
# (stub or real) is currently wired in stubs.py.

# Run:  python pipeline.py
# """
# from __future__ import annotations
# import json
# import sys
# from interfaces import ClauseFinding, ReviewReport, Document
# import stubs


# def run(file_path: str = "data/sample_document.json") -> tuple[Document, ReviewReport]:
#     # Stage 1 — Drashti: parse into clauses.
#     # A pre-made .json (e.g. the sample doc) is already a serialized Document
#     # and is loaded directly; anything else goes through the real parser.
#     if file_path.lower().endswith(".json"):
#         with open(file_path, "r", encoding="utf-8") as f:
#             doc: Document = Document.model_validate(json.load(f))
#     else:
#         doc = stubs.parse_document(file_path)

#     findings: list[ClauseFinding] = []
#     for clause in doc.clauses:
#         # Stage 2 — Veda: classify
#         cls = stubs.classify_clause(clause)

#         # Stage 3 — Veda: retrieve legal knowledge
#         retr = stubs.retrieve(clause.text_en, cls.predicted_type)

#         # Stage 4 — Kashish: analysis agents
#         partial = stubs.analyze_clause(clause, cls, retr)

#         # Stage 5 — Kashish: RL uncertainty agent decides the review action
#         decision = stubs.uncertainty_agent({
#             "clause_id": clause.clause_id,
#             "confidence": cls.confidence,
#             "relevance": max((h.relevance for h in retr.results), default=0.0),
#             "agreement": 1.0,  # stub: agreement across analysis agents
#             "risk_level": partial["risk_level"],
#         })

#         findings.append(ClauseFinding(
#             **partial,
#             review_decision=decision.action,
#             review_reason=decision.reason,
#         ))

#     # Stage 6 — Kashish: assemble report
#     return doc, stubs.build_report(doc, findings)


# if __name__ == "__main__":
#     path = sys.argv[1] if len(sys.argv) > 1 else "data/sample_document.json"
#     _doc, report = run(path)
#     print(report.model_dump_json(indent=2))
#     print("\nOK — walking skeleton ran end-to-end and produced a valid ReviewReport.")
