"""Evidence-first risk analysis for the legal contract review pipeline.

The module deliberately does NOT calculate a numerical risk score.

Flow:
    LegalBERT LEDGAR category
        -> is the category mapped to a Practical Law risk-allocation area?
            NO  -> no_configured_risk_found (Risk LLM is NOT called)
            YES -> clause + classification + source guidance + existing
                   GraphRAG evidence are given to a restricted Risk LLM,
                   which returns one of:
                       attention_required / no_configured_risk_found /
                       insufficient_evidence

Governing framework: Practical Law, "Risk Allocation in Commercial
Contracts" (Practice Note 4-519-5496), supported by "Using Contractual Risk
Allocation Provisions to Minimize Risk and Maximize Reward". The framework
identifies risk-allocation mechanisms and the dimensions worth examining. It
does NOT supply universal binary rules, so no keyword, phrase, or missing
phrase decides the assessment here. Phrase matches are passed to the LLM as
observations only.

Ollama is a restricted evidence interpreter. It is not allowed to invent risk
areas, legal rules, citations, or facts, and it cannot change the
classification.
"""
from __future__ import annotations

import json
import re
from typing import Any

import requests

from interfaces import Clause, Classification, RetrievalResult, Issue


OLLAMA_URL = "http://localhost:11434/api/generate"
LLM_MODEL = "qwen2.5:3b"
# Local CPU inference takes ~60s per clause; leave generous headroom so
# mapped clauses get a contextual LLM assessment instead of the fallback.
LLM_TIMEOUT_SECONDS = 240

ASSESSMENTS = ("attention_required", "no_configured_risk_found", "insufficient_evidence")


PRACTICAL_LAW_PRIMARY = {
    "title": "Risk Allocation in Commercial Contracts (Practical Law Practice Note 4-519-5496)",
    "url": "https://content.next.westlaw.com/practical-law/document/I1c6311acef2811e28578f7ccc38dcbee/Risk-Allocation-in-Commercial-Contracts",
    "basis": "Identifies contractual mechanisms through which parties allocate and manage risk, including indemnification, limitations on liability, termination rights, force majeure, contractual remedies, UCC product warranties, insurance coverage, payment terms, guaranties, representations and warranties, covenants, and conditions precedent.",
}

PRACTICAL_LAW_SUPPORTING = {
    "title": "Using Contractual Risk Allocation Provisions to Minimize Risk and Maximize Reward (Practical Law)",
    "url": "https://ca.practicallaw.thomsonreuters.com/w-008-1168?contextData=%28sc.Default%29&transitionType=Default",
    "basis": "Explains how indemnification, limitation-of-liability, payment-timing, and representation-and-warranty provisions allocate loss and exposure between the parties, and the dimensions along which they can be drafted.",
}


# Classifier outputs (original LEDGAR labels) that have a configured,
# source-backed risk-allocation area. This is NOT a statement that these are
# the only legally important provisions. Practical Law areas with no matching
# LEDGAR label (limitation of liability, force majeure, guaranties, covenants,
# conditions precedent, product warranties) are intentionally left unmapped.
RISK_AREA_BY_LEDGAR: dict[str, str] = {
    "Indemnifications": "Indemnification",
    "Indemnity": "Indemnification",

    "Terminations": "Termination Rights",

    "Remedies": "Contractual Remedies",
    "Specific Performance": "Contractual Remedies",

    "Insurances": "Insurance Coverage",

    "Payments": "Payment Terms",
    "Fees": "Payment Terms",
    "Costs": "Payment Terms",
    "Expenses": "Payment Terms",

    "Representations": "Representations and Warranties",
    "Warranties": "Representations and Warranties",
}


# Dimensions are questions for contextual analysis, NOT mandatory legal
# requirements. The absence of a dimension does not by itself establish risk.
RISK_AREA_GUIDANCE: dict[str, dict[str, Any]] = {
    "Indemnification": {
        "sources": [PRACTICAL_LAW_PRIMARY, PRACTICAL_LAW_SUPPORTING],
        "mechanism": "Indemnification provisions allocate loss between the parties.",
        "dimensions": [
            "who is indemnified",
            "covered claims",
            "nexus or triggering event",
            "damages",
            "recoverable damages",
            "exceptions",
            "duration",
            "caps or limits",
            "indemnification procedures",
        ],
    },
    "Termination Rights": {
        "sources": [PRACTICAL_LAW_PRIMARY],
        "mechanism": "Termination rights let parties exit or end performance and allocate the consequences of doing so.",
        "dimensions": [
            "termination rights and conditions",
            "termination for cause",
            "termination for convenience",
            "notice/cure considerations where relevant",
            "effects of termination where supplied",
        ],
    },
    "Contractual Remedies": {
        "sources": [PRACTICAL_LAW_PRIMARY],
        "mechanism": "Contractual remedy provisions define the relief available to a party when the other party fails to perform.",
        "dimensions": [
            "equitable remedies",
            "cumulative remedies",
            "exclusive remedies",
            "liquidated damages",
        ],
    },
    "Insurance Coverage": {
        "sources": [PRACTICAL_LAW_PRIMARY],
        "mechanism": "Insurance obligations can shift the financial consequences of identified risks to insurers.",
        "dimensions": [
            "insurance obligations",
            "coverage allocation",
            "risk allocation through insurance",
        ],
    },
    "Payment Terms": {
        "sources": [PRACTICAL_LAW_PRIMARY, PRACTICAL_LAW_SUPPORTING],
        "mechanism": "Payment timing allocates nonpayment, default, and financing risk between the parties.",
        "dimensions": [
            "payment timing",
            "deferred payment",
            "advance payment",
            "nonpayment/default exposure",
            "effect of timing on the parties' risk",
        ],
    },
    "Representations and Warranties": {
        "sources": [PRACTICAL_LAW_PRIMARY, PRACTICAL_LAW_SUPPORTING],
        "mechanism": "Representations and warranties allocate exposure; broader statements can allocate more risk to the party making them.",
        "dimensions": [
            "scope",
            "accuracy/assurance",
            "allocation of exposure",
            "survival",
            "qualifications or limitations",
            "caps/remedies where supplied",
        ],
    },
}


# Phrase extraction is observation only. A match never decides an assessment,
# carries no weight, and is interpreted by the LLM in the context of the clause.
OBSERVATION_PHRASES: dict[str, list[str]] = {
    "Indemnification": [
        "any and all", "all claims", "all losses", "without limitation", "unlimited",
        "hold harmless", "defend", "regardless of", "negligence", "to the extent",
        "limited to", "survive",
    ],
    "Termination Rights": [
        "immediately", "without notice", "sole discretion", "for convenience", "for any reason",
        "material breach", "cure", "notice", "automatically renew", "automatic renewal",
    ],
    "Contractual Remedies": [
        "specific performance", "injunctive", "equitable relief", "cumulative",
        "sole and exclusive remedy", "exclusive remedy", "liquidated damages",
    ],
    "Insurance Coverage": [
        "additional insured", "waiver of subrogation", "certificate of insurance",
        "coverage", "deductible", "policy limits",
    ],
    "Payment Terms": [
        "in advance", "upon receipt", "within", "days", "immediately due", "accelerat",
        "non-refundable", "late fee", "interest", "penalty",
    ],
    "Representations and Warranties": [
        "as is", "disclaim", "to the knowledge", "material respects", "survive",
        "represents and warrants", "sole remedy", "exclusive remedy",
    ],
}


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def _text_comparison(original: str, english: str) -> dict[str, Any]:
    original = original or ""
    english = english or ""
    if not original or not english:
        return {
            "available": False,
            "same_text": original == english,
            "original_length": len(original),
            "english_length": len(english),
            "note": "Original/English comparison unavailable for one or both fields.",
        }
    same = _norm(original) == _norm(english)
    return {
        "available": True,
        "same_text": same,
        "original_length": len(original),
        "english_length": len(english),
        "length_difference": abs(len(original) - len(english)),
        "note": (
            "The normalized texts are equivalent."
            if same
            else "The original and English texts differ; review the translation alongside the original before relying on translated wording."
        ),
    }


def _best_relevance(retr: RetrievalResult) -> float:
    return max((hit.relevance for hit in retr.results), default=0.0)


def _retrieval_evidence(retr: RetrievalResult, limit: int = 5) -> list[dict[str, Any]]:
    return [
        {
            "source_id": hit.source_id,
            "title": hit.title,
            "snippet": hit.snippet,
            "relevance": hit.relevance,
            "graph_path": hit.graph_path or [],
        }
        for hit in retr.results[:limit]
    ]


def _observed_phrases(text: str, risk_area: str) -> list[str]:
    normalized = _norm(text)
    return [p for p in OBSERVATION_PHRASES.get(risk_area, []) if p in normalized]


def _restricted_risk_llm(payload: dict[str, Any]) -> dict[str, Any] | None:
    prompt = """You are a restricted contract risk-analysis module.

You are an interpreter of the supplied framework and evidence, not the source of legal knowledge.

The supplied Practical Law material identifies contractual risk-allocation mechanisms and factors to examine. It does not provide a universal binary rule that the presence or absence of one factor automatically establishes legal risk. Analyze the actual clause in context. A phrase, keyword, or missing phrase is only an observation. Do not convert observations into legal conclusions without supporting evidence.

TASK:
Using ONLY the clause, its classification, the mapped risk area, the source guidance dimensions, and the retrieved evidence below, decide whether the supplied material identifies a specific contractual concern in this clause.

STRICT RULES:
1. Do not browse, search, or rely on outside knowledge. Do not invent facts, legal rules, authorities, case law, citations, or clause text.
2. Do not infer jurisdiction, party intent, or missing contract terms as facts.
3. Do not call anything illegal, invalid, unenforceable, or definitely legally risky.
4. Do not change the classification or its confidence. High classifier confidence is not legal correctness; low confidence is not risk.
5. Do not create risk areas beyond the supplied mapped_risk_area. Do not produce any numerical score.
6. The analysis is clause-level. Write "No apparent cap is stated in the analyzed clause", never "The contract has no cap". Another clause may contain it.
7. If a safeguard or limitation appears in the supplied retrieved evidence, do not report it as absent. Do not assume retrieved items belong to the same agreement unless the supplied evidence says so.
8. observed_phrases are only observations, not findings.
9. Absence alone is not a concern. A dimension that is simply not stated in the clause (no cap, no notice, no procedure, no dispute mechanism) must be reported as "not_apparent_in_clause" and must NOT by itself justify "attention_required". Ordinary, mutual, or customary wording (for example "indemnify, defend and hold harmless", "within thirty days") is not a concern by itself.
10. "attention_required" requires a concern grounded in language actually present in the clause, read in context. At least one observation must quote that clause language verbatim as its evidence.
11. If the retrieval results are empty, or the clause text is too fragmentary to evaluate the dimensions, return "insufficient_evidence".

ASSESSMENT (choose exactly one):
- "attention_required": the clause, read in context with the supplied guidance and evidence, presents a specific contractual concern that warrants review. Explain the concern concretely.
- "no_configured_risk_found": the clause was examined against the supplied dimensions and no specific concern is identified. This does NOT mean the clause is legally safe.
- "insufficient_evidence": the clause or supplied evidence is too incomplete or weak to assess the relevant dimensions reliably.

Return JSON only, in exactly this shape:
{
  "assessment": "attention_required | no_configured_risk_found | insufficient_evidence",
  "risk_finding": "one sentence stating the specific concern, or why none / why evidence is insufficient",
  "observations": [
    {"factor": "<one of the supplied dimensions>", "status": "present | not_apparent_in_clause | unclear", "evidence": "<short quote from the clause or a supplied source_id>"}
  ],
  "reasoning": "concise explanation, under 120 words, based only on the supplied material",
  "supporting_evidence": ["clause_text or a supplied source_id"]
}

SUPPLIED MATERIAL:
"""
    prompt += json.dumps(payload, ensure_ascii=False, indent=2)

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": LLM_MODEL,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "options": {"temperature": 0},
            },
            timeout=LLM_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        data = json.loads(response.json().get("response", ""))
    except Exception:
        return None
    return _validate_llm_output(data, payload)


def _validate_llm_output(data: Any, payload: dict[str, Any]) -> dict[str, Any] | None:
    """Accept only the allowed schema; anything else counts as unavailable."""
    if not isinstance(data, dict):
        return None
    assessment = str(data.get("assessment", "")).strip()
    if assessment not in ASSESSMENTS:
        return None

    risk_finding = str(data.get("risk_finding") or "").strip()
    reasoning = str(data.get("reasoning") or "").strip()

    allowed_factors = set(payload["source_guidance"]["dimensions"])
    observations = []
    for item in data.get("observations") or []:
        if not isinstance(item, dict):
            continue
        factor = str(item.get("factor") or "").strip()
        if factor not in allowed_factors:
            continue
        status = str(item.get("status") or "").strip()
        if status not in ("present", "not_apparent_in_clause", "unclear"):
            status = "unclear"
        observations.append({
            "factor": factor,
            "status": status,
            "evidence": str(item.get("evidence") or "").strip(),
        })

    allowed_refs = {"clause_text"} | {r["source_id"] for r in payload["retrieval"]["results"]}
    supporting = [
        str(ref) for ref in (data.get("supporting_evidence") or [])
        if str(ref) in allowed_refs
    ]

    # A concern must be stated, explained, and grounded in language present in
    # the clause; an absence-only "concern" is not a valid finding.
    if assessment == "attention_required":
        clause_norm = _norm(payload["clause_text"])
        grounded = any(
            obs["evidence"] and _norm(obs["evidence"].strip("\"'")) in clause_norm
            for obs in observations
        )
        if not (risk_finding and reasoning and grounded):
            return None

    return {
        "assessment": assessment,
        "risk_finding": risk_finding,
        "observations": observations,
        "reasoning": reasoning,
        "supporting_evidence": supporting,
    }


def analyze_clause(
    clause: Clause,
    cls: Classification,
    retr: RetrievalResult,
) -> dict[str, Any]:
    """Produce evidence-first qualitative analysis with no numeric risk score."""
    clause_type = cls.predicted_type.strip()
    risk_area = RISK_AREA_BY_LEDGAR.get(clause_type)
    best_relevance = _best_relevance(retr)
    comparison = _text_comparison(clause.text_original, clause.text_en)
    retrieval = {
        "best_relevance": best_relevance,
        "results": _retrieval_evidence(retr),
    }
    classification = {
        "predicted_type": cls.predicted_type,
        "confidence": cls.confidence,
        "top_k": [x.model_dump() for x in cls.top_k],
    }

    risk_factors: list[dict[str, Any]] = []
    issues: list[Issue] = []
    recommendations: list[str] = []

    if risk_area is None:
        # Unmapped category: the Risk LLM is not called.
        assessment = "no_configured_risk_found"
        assessment_basis = (
            f"No configured source-backed risk framework applies to the classifier category "
            f"'{clause_type}'. This does not mean the clause is legally risk-free."
        )
        recommendations.append(
            "No configured source-backed risk framework is attached to this clause category; "
            "review the clause in the context of the complete agreement."
        )
        evidence_payload = {
            "clause_text": clause.text_en,
            "original_text": clause.text_original,
            "classification": classification,
            "mapped_risk_area": None,
            "retrieval": retrieval,
            "text_comparison": comparison,
            "risk_llm_used": False,
            "assessment": assessment,
            "assessment_basis": assessment_basis,
        }
        explanation = (
            f"{assessment_basis} Classification confidence is {cls.confidence:.2f}; "
            f"best retrieval relevance is {best_relevance:.2f}."
        )
    else:
        guidance = RISK_AREA_GUIDANCE[risk_area]
        observed = _observed_phrases(clause.text_en, risk_area)
        llm_payload = {
            "clause_text": clause.text_en,
            "original_text": clause.text_original,
            "classification": classification,
            "mapped_risk_area": risk_area,
            "source_guidance": {
                "sources": [s["title"] for s in guidance["sources"]],
                "mechanism": guidance["mechanism"],
                "dimensions": guidance["dimensions"],
                "note": "Dimensions are questions for contextual examination, not mandatory requirements.",
            },
            "retrieval": retrieval,
            "observed_phrases": observed,
            "text_comparison": comparison,
        }
        llm_result = _restricted_risk_llm(llm_payload)

        if llm_result is None:
            # Fail-safe: no fabricated conclusion, phrases do not become findings.
            assessment = "insufficient_evidence"
            assessment_basis = (
                f"Contextual Risk LLM assessment was unavailable for the mapped risk area "
                f"'{risk_area}', so no contextual conclusion was reached. Observed phrases are "
                f"recorded as evidence only."
            )
            recommendations.append(
                f"Contextual assessment was unavailable; manually review this clause against the "
                f"{risk_area} dimensions and the surrounding contract provisions."
            )
            explanation = (
                f"{assessment_basis} Classification confidence is {cls.confidence:.2f}; "
                f"best retrieval relevance is {best_relevance:.2f}."
            )
        else:
            assessment = llm_result["assessment"]
            assessment_basis = llm_result["risk_finding"]
            explanation = llm_result["reasoning"] or llm_result["risk_finding"]

            if assessment == "attention_required":
                for obs in llm_result["observations"]:
                    risk_factors.append({
                        "factor": f"{risk_area}: {obs['factor']}",
                        "status": obs["status"],
                        "observation": llm_result["risk_finding"],
                        "evidence": obs["evidence"],
                        "matched_text": obs["evidence"],
                        "risk_area": risk_area,
                        "sources": guidance["sources"],
                    })
                if not risk_factors:
                    risk_factors.append({
                        "factor": risk_area,
                        "status": "present",
                        "observation": llm_result["risk_finding"],
                        "evidence": clause.text_en[:300],
                        "matched_text": clause.text_en[:300],
                        "risk_area": risk_area,
                        "sources": guidance["sources"],
                    })
                issues.append(Issue(
                    category="risk",
                    severity="medium",
                    description=llm_result["risk_finding"],
                    evidence=llm_result["supporting_evidence"] + [s["title"] for s in guidance["sources"]],
                ))
                recommendations.append(
                    f"Have a human/legal reviewer examine the identified {risk_area} concern: "
                    f"{llm_result['risk_finding']}"
                )
            elif assessment == "insufficient_evidence":
                recommendations.append(
                    f"The available evidence was insufficient for a reliable {risk_area} assessment; "
                    f"review the relevant surrounding contract provisions."
                )
            else:
                recommendations.append(
                    f"No configured source-backed {risk_area} concern was identified in the analyzed clause; "
                    f"this is not a statement that the clause is legally risk-free."
                )

        evidence_payload = {
            **llm_payload,
            "source_guidance": {**llm_payload["source_guidance"], "sources": guidance["sources"]},
            "risk_llm_used": llm_result is not None,
            "risk_llm_result": llm_result,
            "assessment": assessment,
            "assessment_basis": assessment_basis,
        }

    if comparison["available"] and not comparison["same_text"]:
        recommendations.append("Review the original-language text alongside the English text because the two versions differ.")

    return {
        "clause_id": clause.clause_id,
        "clause_type": cls.predicted_type,
        "risk_assessment": assessment,
        "risk_factors": risk_factors,
        "issues": issues,
        "recommendation": " ".join(recommendations),
        "confidence": round(float(cls.confidence), 4),
        "best_relevance": round(best_relevance, 4),
        "text_comparison": comparison,
        "evidence": evidence_payload,
        "explanation": explanation,
    }
