"""Minded Hugging Face Space: Autonomous Analytical Intelligence OS.

Enterprise-Grade Autonomous Data Analyst.
Hosted as a thin adapter over the canonical Minded analytical kernel
(apps.api.src.ai.runtime.InvestigationRuntime -> InvestigationController).
100% deterministic, local-first, zero-cloud API dependency.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Dict, List

import gradio as gr
import pandas as pd
import plotly.express as px

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from apps.api.src.ai.runtime import InvestigationRuntime
from packages.analytics_core.src.ingestion.robust_loader import RobustFileLoader
from hf_space.resource_policy import (
    MAX_DATASETS,
    MAX_QUESTION_CHARS,
    validate_dataframe,
    validate_file,
)

SAMPLE_DIR = Path(__file__).resolve().parent / "sample_data"


class _DeterministicProvider:
    """Zero-AI deterministic compatibility provider."""
    is_ai_enabled = False
    model = "none"

    def generate(self, *args, **kwargs):
        raise RuntimeError("External AI is disabled in the hosted deterministic Space.")


def _load_files(files) -> Dict[str, pd.DataFrame]:
    if not files:
        raise gr.Error("Upload at least one CSV or Parquet dataset.")
    if len(files) > MAX_DATASETS:
        raise gr.Error(f"Upload at most {MAX_DATASETS} datasets per investigation.")

    datasets: Dict[str, pd.DataFrame] = {}
    loader = RobustFileLoader()
    for item in files:
        path = item.name if hasattr(item, "name") else str(item)
        try:
            validate_file(path)
            suffix = Path(path).suffix.lower()
            if suffix == ".parquet":
                df = pd.read_parquet(path)
            else:
                df, _report = loader.load(file_path=path, filename=Path(path).name)
            validate_dataframe(df)
        except Exception as exc:
            raise gr.Error(f"Could not load {Path(path).name}: {exc}") from exc
        name = Path(path).stem or f"dataset_{len(datasets) + 1}"
        if name in datasets:
            name = f"{name}_{len(datasets) + 1}"
        datasets[name] = df
    return datasets


def _json_default(value: Any):
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def _evidence_frame(response) -> pd.DataFrame:
    rows = []
    for evidence in response.evidence:
        raw = evidence.raw_metrics or {}
        structured = raw.get("structured_result", raw) if isinstance(raw, dict) else {}
        calc_trace = getattr(evidence, "calculation_trace", None)
        trace_str = (
            f"formula: {calc_trace.get('formula')}, inputs: {calc_trace.get('inputs')}"
            if isinstance(calc_trace, dict)
            else ""
        )
        rows.append({
            "evidence_id": evidence.id,
            "validation": getattr(evidence.validation_status, "value", evidence.validation_status),
            "effect_size": f"{evidence.effect_size:.4f}" if evidence.effect_size is not None else "N/A",
            "p_value": f"{evidence.p_value:.4e}" if evidence.p_value is not None else "N/A",
            "statement": evidence.statement,
            "rows_analyzed": getattr(evidence, "row_count_analyzed", 0),
            "calculation_trace": trace_str,
            "result_summary": json.dumps(structured, default=_json_default)[:500],
        })
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["evidence_id", "validation", "effect_size", "p_value", "statement", "rows_analyzed", "calculation_trace", "result_summary"])


def _hypothesis_frame(response) -> pd.DataFrame:
    rows = [
        {
            "hypothesis": h.statement,
            "posterior": round(float(h.posterior_probability), 4) if h.posterior_probability is not None else None,
            "status": h.status,
            "target_metric": h.target_metric,
            "target_dimension": h.target_dimension,
            "target_value": h.target_value,
        }
        for h in response.hypotheses
    ]
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["hypothesis", "posterior", "status", "target_metric", "target_dimension", "target_value"])


def _recommendations_frame(response) -> pd.DataFrame:
    rows = []
    for r in getattr(response, "decision_recommendations", []) or []:
        eu = getattr(r, "expected_utility", None)
        rows.append({
            "action": r.action_title,
            "description": r.action_description,
            "target_metric": r.target_metric,
            "expected_gain": getattr(eu, "expected_gain_metric", "N/A") if eu else "N/A",
            "downside_risk": getattr(eu, "downside_risk_metric", "N/A") if eu else "N/A",
            "win_probability": f"{getattr(eu, 'probability_of_success', 0):.0%}" if eu else "N/A",
            "net_utility": f"{getattr(eu, 'net_expected_utility', 0):.2f}" if eu else "N/A",
            "preconditions": ", ".join(r.required_preconditions) if r.required_preconditions else "None",
        })
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["action", "description", "target_metric", "expected_gain", "downside_risk", "win_probability", "net_utility", "preconditions"])


def _suggested_questions_markdown(response) -> str:
    questions = getattr(response, "suggested_questions", []) or []
    if not questions:
        return "*No additional automated questions discovered.*"
    lines = ["### 💡 Autonomous Follow-Up Questions Discovered:"]
    for q in questions[:6]:
        text = q.get("question") if isinstance(q, dict) else getattr(q, "question", str(q))
        lines.append(f"- **{text}**")
    return "\n".join(lines)


def _make_plot(response):
    df = _hypothesis_frame(response)
    if df.empty or "posterior" not in df:
        return None
    df = df.dropna(subset=["posterior"])
    if df.empty:
        return None
    fig = px.bar(
        df,
        x="hypothesis",
        y="posterior",
        title="Bayesian Hypothesis Posteriors (Canonical Evidence Evaluation)",
        color="posterior",
        color_continuous_scale="Viridis",
    )
    fig.update_yaxes(range=[0, 1])
    fig.update_layout(showlegend=False, xaxis_tickangle=-25)
    return fig


def run_minded_investigation(files, question, progress=gr.Progress()):
    question = (question or "").strip()
    if not question:
        raise gr.Error("Enter an investigative question.")
    if len(question) > MAX_QUESTION_CHARS:
        raise gr.Error(f"Question is limited to {MAX_QUESTION_CHARS:,} characters.")

    progress(0.05, desc="Loading & sanitizing enterprise datasets...")
    datasets = _load_files(files)
    progress(0.25, desc="Autonomous semantic relational resolution & hypothesis generation...")

    runtime = InvestigationRuntime(ai_provider=_DeterministicProvider())
    response = runtime.execute_investigation(
        question=question,
        project_id="hf-hosted-ephemeral",
        datasets=datasets,
        business_metrics=None,
        max_steps=0,
    )
    progress(0.85, desc="Calculating statistical evidence & decision playbooks...")

    manifest = response.manifest.model_dump(mode="json") if response.manifest else {}
    claim_gate = getattr(response, "claim_gate", None)
    claim_gate_payload = claim_gate.model_dump(mode="json") if hasattr(claim_gate, "model_dump") else claim_gate

    report = f"""## ⚡ Minded Autonomous Analytical Verdict

| Attribute | Evaluated Status |
| :--- | :--- |
| **Verdict** | `{response.verdict.value}` |
| **Execution Status** | `{response.status.value}` |
| **Epistemic Confidence** | `{getattr(response.confidence, 'value', response.confidence)}` |

### 🎯 Direct Analyst Answer
{response.direct_answer or 'No admissible answer was produced.'}

### 🔍 Main Analytical Finding
{response.main_finding or 'No main finding was produced.'}
"""

    progress(1.0, desc="Investigation complete.")
    return (
        report,
        _make_plot(response),
        _recommendations_frame(response),
        _hypothesis_frame(response),
        _evidence_frame(response),
        _suggested_questions_markdown(response),
        json.dumps(claim_gate_payload, indent=2, default=_json_default),
        json.dumps(manifest, indent=2, default=_json_default),
    )


# Scenario helper functions for 1-click loading
def load_scenario_b2b():
    paths = [
        str(SAMPLE_DIR / "b2b_saas" / "customers.csv"),
        str(SAMPLE_DIR / "b2b_saas" / "orders.csv"),
        str(SAMPLE_DIR / "b2b_saas" / "products.csv"),
    ]
    question = "Which customer segment generated the highest revenue from the premium product category?"
    return paths, question


def load_scenario_financials():
    paths = [
        str(SAMPLE_DIR / "financials" / "quarterly_financials_messy.csv"),
    ]
    question = "What is the total revenue by division and which division had negative net profit?"
    return paths, question


def load_scenario_cohorts():
    paths = [
        str(SAMPLE_DIR / "cohorts" / "user_retention_cohorts.csv"),
    ]
    question = "Which tier has the lowest retention rate, and what is the average monthly spend for that tier?"
    return paths, question


custom_theme = gr.themes.Default(primary_hue="amber", secondary_hue="emerald", neutral_hue="slate")

with gr.Blocks(title="Minded — Autonomous Analytical Intelligence OS") as demo:
    gr.Markdown("""
    # ⚡ MINDED: Autonomous Analytical Intelligence OS
    ### Real Human Data Analyst Level AI — Local-First, Zero-Cloud, Relational & Deterministic

    Upload your own CSV / Parquet data across multiple tables, or click a demo scenario below to test multi-table relational joins, messy enterprise types, and compound questions.
    """)

    with gr.Row():
        btn_b2b = gr.Button("🏢 Scenario 1: Multi-Table Relational (Customers + Orders + Products)", variant="secondary")
        btn_fin = gr.Button("📊 Scenario 2: Messy Financials & Dirty Types ($1,200.50, (450.00), #N/A)", variant="secondary")
        btn_coh = gr.Button("📈 Scenario 3: SaaS Retention & Churn Cohorts", variant="secondary")

    with gr.Row():
        with gr.Column(scale=1):
            files = gr.File(label="Upload CSV / Parquet datasets (multi-table supported)", file_types=[".csv", ".parquet"], file_count="multiple")
            question = gr.Textbox(label="Investigative Question", placeholder="e.g. Which customer segment generated the highest revenue from the premium product category?", lines=3)
            run = gr.Button("⚡ RUN MINDED INVESTIGATION", variant="primary", size="lg")
        with gr.Column(scale=2):
            report = gr.Markdown(label="Scientific Verdict & Direct Answer")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### 📋 Prescriptive Strategic Playbook (Decision Recommendations)")
            recommendations = gr.Dataframe(label="Strategic Playbook & Expected Utility", interactive=False)

    with gr.Row():
        posterior_plot = gr.Plot(label="Canonical Hypothesis Posteriors")

    with gr.Row():
        with gr.Tab("Hypothesis Ledger"):
            hypotheses = gr.Dataframe(label="Evaluated Hypotheses", interactive=False)
        with gr.Tab("Evidence Ledger & Calculation Traces"):
            evidence = gr.Dataframe(label="Statistical Evidence Ledger", interactive=False)

    with gr.Row():
        suggested_q = gr.Markdown(label="Discovered Follow-Up Inquiries")

    with gr.Accordion("🔒 Claim Gate & Epistemic Audit Trail (Deterministic Manifest)", open=False):
        with gr.Row():
            with gr.Column():
                gr.Markdown("#### Claim Gate")
                claim_gate_box = gr.Code(label="Claim Gate Payload", language="json")
            with gr.Column():
                gr.Markdown("#### Analysis Manifest")
                manifest_box = gr.Code(label="Analysis Manifest", language="json")

    # Wire up scenario buttons
    btn_b2b.click(load_scenario_b2b, outputs=[files, question])
    btn_fin.click(load_scenario_financials, outputs=[files, question])
    btn_coh.click(load_scenario_cohorts, outputs=[files, question])

    # Wire up run button
    run.click(
        run_minded_investigation,
        inputs=[files, question],
        outputs=[report, posterior_plot, recommendations, hypotheses, evidence, suggested_q, claim_gate_box, manifest_box],
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, theme=custom_theme)
