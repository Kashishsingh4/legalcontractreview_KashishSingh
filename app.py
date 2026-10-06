"""Streamlit UI for the AI Legal Contract Review System."""
from __future__ import annotations

import json
import os

import streamlit as st

from pipeline import run_with_versioning
import storage

UPLOAD_DIR = "data/uploads"

st.set_page_config(
    page_title="AI Legal Contract Review",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.block-container {padding-top: 2rem; padding-bottom: 3rem;}
.hero {padding: 1.4rem 1.6rem; border-radius: 16px; border: 1px solid rgba(128,128,128,.25); margin-bottom: 1.2rem;}
.hero h1 {margin-bottom: .25rem;}
.hero p {margin: 0; opacity: .78;}
.card {padding: 1rem 1.1rem; border-radius: 14px; border: 1px solid rgba(128,128,128,.25); margin-bottom: .8rem;}
.small {font-size: .86rem; opacity: .72;}
.factor {padding: .7rem .85rem; border-left: 4px solid currentColor; border-radius: 8px; background: rgba(128,128,128,.08); margin: .45rem 0;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
<h1>⚖️ AI Legal Contract Review</h1>
<p>Evidence-first contract analysis using document parsing, LegalBERT classification, GraphRAG retrieval and restricted local-LLM explanation.</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("Analysis Pipeline")
    st.markdown("1. 📄 Parse contract")
    st.markdown("2. 🏷️ Classify clauses")
    st.markdown("3. 🔎 Retrieve legal evidence")
    st.markdown("4. 🧾 Analyze evidence")
    st.markdown("5. 👩‍⚖️ Review decision")
    st.markdown("6. 📊 Generate report")
    st.divider()
    st.caption("No numerical 0–100 risk score is assigned.")

    st.divider()
    st.header("📚 Analysis History")
    past_analyses = storage.list_analyses()
    if past_analyses:
        st.caption(f"{len(past_analyses)} contract(s) previously analyzed and stored.")
        for row in past_analyses[:10]:
            status = row["overall_assessment"].replace("_", " ").title() if row["overall_assessment"] else "—"
            st.markdown(f"**{row['filename']}**  \n<span class='small'>{row['clause_count']} clauses · {status}</span>", unsafe_allow_html=True)
    else:
        st.caption("No contracts analyzed yet.")
    st.caption("Version comparison is optional — choose a previous contract below the uploader to compare against.")

uploaded = st.file_uploader(
    "Upload Contract",
    type=["pdf", "docx"],
    help="Upload the contract you want to analyze.",
)

compare_options = {"": "No comparison — analyze this contract only"}
compare_options.update({row["doc_id"]: f"{row['filename']} · {row['clause_count']} clauses" for row in past_analyses})
compare_with = st.selectbox(
    "Compare with a previous version (optional)",
    options=list(compare_options),
    format_func=compare_options.get,
)

if uploaded is None:
    st.info("Upload a PDF or DOCX contract to begin analysis.")
    st.stop()

os.makedirs(UPLOAD_DIR, exist_ok=True)
save_path = os.path.join(UPLOAD_DIR, uploaded.name)
with open(save_path, "wb") as f:
    f.write(uploaded.getbuffer())

try:
    with st.spinner("Parsing, classifying, retrieving evidence and analyzing clauses..."):
        doc, report, version = run_with_versioning(save_path, compare_with_doc_id=compare_with or None)
except Exception as e:
    st.error(str(e))
    st.stop()

# ============================================================
# DOCUMENT SUMMARY
# ============================================================
st.header("📄 Document Overview")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Pages", doc.page_count)
c2.metric("Clauses", len(doc.clauses))
c3.metric("Language", doc.source_language)
c4.metric("Review Status", report.overall_assessment.replace("_", " ").title())

st.info(report.summary)

st.divider()

# ============================================================
# VERSION COMPARISON — only rendered when the user chose a previous
# version to compare against.
# ============================================================
if version is not None:
    st.header("🔄 Version Comparison")
    st.caption(
        f"Compared against the previous version you selected: "
        f"{compare_options.get(version.matched_doc_id, version.matched_doc_id)}."
    )

    diff = version.diff
    comparison = version.comparison
    recommendation = version.recommendation

    d1, d2, d3, d4 = st.columns(4)
    added = sum(e.change_type == "added" for e in diff.entries)
    removed = sum(e.change_type == "removed" for e in diff.entries)
    modified = sum(e.change_type == "modified" for e in diff.entries)
    unchanged = sum(e.change_type == "unchanged" for e in diff.entries)
    d1.metric("Added", added)
    d2.metric("Removed", removed)
    d3.metric("Modified", modified)
    d4.metric("Unchanged", unchanged)

    r1, r2, r3 = st.columns(3)
    r1.metric("Increased Risk", comparison.increased_count)
    r2.metric("Reduced Risk", comparison.reduced_count)
    r3.metric("Unchanged Risk", comparison.unchanged_count)

    verdict_label = {"old": "Previous version recommended", "new": "New version recommended", "equivalent": "Versions roughly equivalent"}
    if recommendation.safer_version == "new":
        st.success(f"**{verdict_label[recommendation.safer_version]}** — {recommendation.reasoning}")
    elif recommendation.safer_version == "old":
        st.warning(f"**{verdict_label[recommendation.safer_version]}** — {recommendation.reasoning}")
    else:
        st.info(f"**{verdict_label[recommendation.safer_version]}** — {recommendation.reasoning}")
    st.caption(
        "This compares whether risk language changed, not whether removing a clause was itself favorable "
        f"or unfavorable — see the clause-level detail below. (Reasoning source: {recommendation.reasoning_source})"
    )

    with st.expander("Clause-level changes"):
        for entry in diff.entries:
            icon = {"added": "➕", "removed": "➖", "modified": "✏️", "unchanged": "="}[entry.change_type]
            heading = entry.new_heading or entry.old_heading or "(untitled clause)"
            with st.container(border=True):
                st.write(f"{icon} **{entry.change_type.title()}** — {heading}")
                if entry.change_type == "modified":
                    st.caption(f"Old: {entry.old_text_snippet}")
                    st.caption(f"New: {entry.new_text_snippet}")
                elif entry.change_type == "added":
                    st.caption(f"New: {entry.new_text_snippet}")
                elif entry.change_type == "removed":
                    st.caption(f"Old: {entry.old_text_snippet}")

    st.divider()

# ============================================================
# CLAUSE ANALYSIS
# ============================================================
st.header("📑 Clause-by-Clause Analysis")
findings = {f.clause_id: f for f in report.clause_findings}

for clause in doc.clauses:
    finding = findings.get(clause.clause_id)
    if finding is None:
        continue

    flag_icon = "🚩" if finding.review_decision in {"flag", "reanalyze"} else "✓"
    heading = clause.heading or f"Clause {clause.index + 1}"

    with st.expander(f"{flag_icon} {heading}", expanded=False):
        st.subheader("Clause Text")
        st.markdown("**Original**")
        st.text_area(
            "Original clause",
            clause.text_original,
            height=150,
            key=f"orig_{clause.clause_id}",
            label_visibility="collapsed",
        )
        st.markdown("**English / analyzed text**")
        st.text_area(
            "English clause",
            clause.text_en,
            height=150,
            key=f"en_{clause.clause_id}",
            label_visibility="collapsed",
        )

        st.divider()
        a, b, c = st.columns(3)
        a.metric("Predicted Clause Type", finding.clause_type)
        b.metric("Classification Confidence", f"{finding.confidence:.1%}")
        c.metric("Best Retrieval Relevance", f"{finding.best_relevance:.1%}")

        st.subheader("🧠 Evidence-Based Assessment")
        if finding.risk_assessment == "attention_required":
            st.warning("Attention required — a configured, source-backed risk factor was matched directly in the clause.")
        elif finding.risk_assessment == "insufficient_evidence":
            st.info("Insufficient evidence — no configured risk factor was matched and supporting retrieval evidence is weak or absent.")
        else:
            st.success("No configured source-backed risk factor found. This is not a statement that the clause is legally risk-free.")

        st.markdown(f"**Review decision:** `{finding.review_decision}`")
        st.caption(finding.review_reason)

        st.subheader("🔎 Matched Risk Factors")
        if finding.risk_factors:
            for factor in finding.risk_factors:
                st.markdown(
                    f"<div class='factor'><b>{factor['factor']}</b><br>"
                    f"Matched phrase: <code>{factor['matched_text']}</code></div>",
                    unsafe_allow_html=True,
                )
                for source in factor.get("sources", []):
                    st.markdown(
                        f"- **{source['title']}** — {source['basis']}  \n"
                        f"  {source['url']}"
                    )
        else:
            st.write("No configured risk factor matched this clause.")

        st.subheader("📚 Supporting Retrieval Evidence")
        retrieval_items = finding.evidence.get("retrieval", {}).get("results", [])
        if retrieval_items:
            for hit in retrieval_items:
                with st.container(border=True):
                    st.write(f"**{hit['title']}**")
                    st.write(hit["snippet"])
                    st.caption(f"Relevance: {hit['relevance']:.3f}")
                    if hit.get("graph_path"):
                        st.caption(" ➜ ".join(hit["graph_path"]))
        else:
            st.info("No supporting legal knowledge was retrieved.")

        st.subheader("🔁 Original vs English Text")
        comparison = finding.text_comparison
        if comparison.get("available"):
            if comparison.get("same_text"):
                st.success("The normalized original and English texts are equivalent.")
            else:
                st.warning(comparison.get("note", "The texts differ."))
                x, y = st.columns(2)
                x.metric("Original characters", comparison.get("original_length", 0))
                y.metric("English characters", comparison.get("english_length", 0))
        else:
            st.info(comparison.get("note", "Comparison unavailable."))

        st.subheader("🤖 Restricted Ollama Explanation")
        st.write(finding.explanation)
        st.caption("Ollama is used only to explain supplied evidence; it is not used to invent risk factors or make independent legal claims.")

        st.subheader("💡 Recommendation")
        st.info(finding.recommendation)

        if finding.issues:
            st.subheader("⚠️ Evidence Observations")
            for issue in finding.issues:
                st.warning(f"**{issue.category.title()} — {issue.severity.title()}**  \n{issue.description}")

# ============================================================
# PARSED JSON — PRESERVED
# ============================================================
st.divider()
st.header("📂 Parsed Contract JSON")
st.caption("The parser output is preserved so the intermediate representation can be inspected independently of the final analysis.")
parsed_json = doc.model_dump()
with st.expander("View parsed JSON"):
    st.json(parsed_json)
st.download_button(
    "⬇️ Download Parsed JSON",
    data=json.dumps(parsed_json, indent=2, ensure_ascii=False),
    file_name=f"{os.path.splitext(doc.filename)[0]}_parsed.json",
    mime="application/json",
)

# ============================================================
# FINAL REPORT JSON — PRESERVED
# ============================================================
st.divider()
st.header("📊 Final Review Report")
report_json = report.model_dump()
with st.expander("View final report JSON"):
    st.json(report_json)
st.download_button(
    "⬇️ Download Review Report JSON",
    data=json.dumps(report_json, indent=2, ensure_ascii=False),
    file_name=f"{os.path.splitext(doc.filename)[0]}_review_report.json",
    mime="application/json",
)



# """File-upload test UI for the legal contract review pipeline (stage 7 scaffold).

# Run:  streamlit run app.py
# """
# import os
# import streamlit as st
# from pipeline import run

# UPLOAD_DIR = "data/uploads"

# st.title("Legal Contract Review")

# uploaded = st.file_uploader("Upload a contract", type=["pdf", "docx"])

# if uploaded is None:
#     st.info("Upload a .pdf or .docx contract above to analyze it.")
# else:
#     os.makedirs(UPLOAD_DIR, exist_ok=True)
#     save_path = os.path.join(UPLOAD_DIR, uploaded.name)
#     with open(save_path, "wb") as f:
#         f.write(uploaded.getbuffer())

#     try:
#         with st.spinner("Analyzing contract..."):
#             doc, report = run(save_path)
#     except Exception as e:
#         st.error(str(e))
#     else:
#         clauses_by_id = {c.clause_id: c for c in doc.clauses}

#         st.write(f"**Detected source language:** `{doc.source_language}`")
#         st.metric("Overall risk", report.overall_risk.upper(), f"score {report.overall_score}")
#         st.write(report.summary)

#         for finding in report.clause_findings:
#             clause = clauses_by_id.get(finding.clause_id)
#             flagged = finding.clause_id in report.flagged_for_review
#             heading = clause.heading if clause and clause.heading else finding.clause_id
#             title = f"{'🚩 FLAGGED — ' if flagged else ''}{heading} ({finding.risk_level} risk)"

#             with st.expander(title):
#                 if clause:
#                     st.write("**Text (original):**")
#                     st.write(clause.text_original)
#                     st.write("**Text (English):**")
#                     st.write(clause.text_en)

#                 st.write("**Risk level:**", finding.risk_level)
#                 st.write("**Review decision:**", finding.review_decision, "—", finding.review_reason)

#                 if finding.issues:
#                     st.write("**Issues:**")
#                     for issue in finding.issues:
#                         st.write(f"- [{issue.severity}] {issue.category}: {issue.description}")
#                 else:
#                     st.write("**Issues:** none")

#         with st.expander("Raw JSON"):
#             st.json(report.model_dump())
