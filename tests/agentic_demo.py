"""Real (unmocked) demonstration of the Supervisor Agent's agentic branches.

Runs three clauses through the full system — real classifier, real FAISS
knowledge base, real Ollama — and prints which branches fired for each:

  A  Long clause whose indemnification language sits past the classifier's
     256-token cutoff -> low confidence -> re-analysis under the runner-up
     category surfaces the risk -> adversarial review.
  B1 Parsing junk (a scanned-page footer) -> weak evidence -> broadened
     retrieval retry -> re-analysis.
  B2 A stray email pasted into a contract -> weak evidence -> retry.

Requires Ollama running with qwen2.5:3b, plus legalbert_finetuned/ and
data/knowledge_base/. Takes several minutes (real LLM calls).

The RL agent explores a random alternative 10% of the time (never down to
"accept"), so a re-run can occasionally take a slightly different path.

Run from the project folder:

    python tests/agentic_demo.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

from interfaces import Clause
from supervisor import run_clause

LONG_CLAUSE = (
    "This Section shall apply to all matters arising under or in connection with this Agreement, "
    "including without limitation any dealings, communications, transactions, or other interactions between the parties "
    "that may occur from time to time during the term hereof, whether such matters are addressed elsewhere in this Agreement "
    "or not, and the parties acknowledge and agree that the provisions set forth in this Section are intended to operate "
    "in addition to, and not in lieu of, any other rights or remedies available at law or in equity, and further agree that "
    "no delay or failure to exercise any right hereunder shall operate as a waiver of such right or any other right, "
    "and that this Section shall survive termination or expiration of this Agreement for any reason whatsoever, "
    "and shall remain binding upon the parties and their respective successors and permitted assigns. "
    "Furthermore, each party represents and warrants that it has full power and authority to enter into this Agreement, "
    "that this Agreement has been duly authorized by all necessary corporate or organizational action, and that this "
    "Agreement constitutes a legal, valid and binding obligation of such party, enforceable against it in accordance with "
    "its terms, subject to applicable bankruptcy, insolvency and other similar laws affecting creditors rights generally, "
    "and further subject to general principles of equity, regardless of whether enforcement is sought in a proceeding at law or in equity. "
    "Notwithstanding the foregoing, the Vendor shall indemnify, defend and hold harmless the Client from and against "
    "any and all claims, damages, losses and expenses, without limitation, arising from Vendor's breach of this Agreement."
)

CASES = {
    "A_low_confidence_long_clause": LONG_CLAUSE,
    "B1_parsing_junk_footer": "Page 4 of 12 ||| scanned image ~~ low quality ~~ please refer to the original hard copy",
    "B2_parsing_junk_cover_note": "Hi Sam, attaching the latest draft as discussed on our call. Lunch on Thursday still works for me!",
}

for name, text in CASES.items():
    clause = Clause(clause_id=name, index=0, heading=None, text_original=text, text_en=text,
                    page=1, char_span=(0, len(text)))
    f = run_clause(clause)
    result = {
        "case": name,
        "predicted_type": f.clause_type,
        "confidence": f.confidence,
        "top_k": f.evidence.get("classification", {}).get("top_k"),
        "best_relevance": f.best_relevance,
        "risk_assessment": f.risk_assessment,
        "review_decision": f.review_decision,
        "review_reason": f.review_reason,
        "agent_route": f.evidence.get("agent_route"),
        "reanalysis": f.evidence.get("reanalysis"),
        "adversarial_review": f.adversarial_review,
    }
    print(f"\n{'=' * 70}\n{name}\n{'=' * 70}")
    print(json.dumps(result, indent=2, ensure_ascii=False))
