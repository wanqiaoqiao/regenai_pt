from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd

from .schemas import (
    CMCNote,
    LiteratureSynthesisReport,
    MechanismEvidence,
    ProtocolEvidence,
    ReagentEvidence,
)

LLMClient = Callable[[str], str]


def _safe_json_loads(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start : end + 1])
        raise


def synthesize_literature(
    extracted_papers: list[dict[str, Any]],
    llm_client: LLMClient,
) -> LiteratureSynthesisReport:
    compact = {
        "papers": extracted_papers,
    }
    prompt = (
        "Synthesize the extracted evidence into JSON with keys: "
        "synthesis_summary, protocol_evidence, reagent_evidence, mechanism_evidence, cmc_notes.\n"
        + json.dumps(compact)
    )
    raw = llm_client(prompt)
    payload = _safe_json_loads(raw)

    report = LiteratureSynthesisReport(synthesis_summary=str(payload.get("synthesis_summary", "")))

    for x in payload.get("protocol_evidence", []):
        report.protocol_evidence.append(
            ProtocolEvidence(
                paper_id=str(x.get("paper_id", "unknown")),
                protocol_step=str(x.get("protocol_step", "")),
                treatment=str(x.get("treatment", "")),
                rationale=str(x.get("rationale", "")),
                confidence=float(x.get("confidence", 0.5)),
            )
        )
    for x in payload.get("reagent_evidence", []):
        report.reagent_evidence.append(
            ReagentEvidence(
                paper_id=str(x.get("paper_id", "unknown")),
                reagent_name=str(x.get("reagent_name", "")),
                dose=str(x.get("dose", "")),
                duration=str(x.get("duration", "")),
                purpose=str(x.get("purpose", "")),
                confidence=float(x.get("confidence", 0.5)),
            )
        )
    for x in payload.get("mechanism_evidence", []):
        report.mechanism_evidence.append(
            MechanismEvidence(
                paper_id=str(x.get("paper_id", "unknown")),
                pathway=str(x.get("pathway", "")),
                mechanism=str(x.get("mechanism", "")),
                supports_treatment=str(x.get("supports_treatment", "")),
                confidence=float(x.get("confidence", 0.5)),
            )
        )
    for x in payload.get("cmc_notes", []):
        report.cmc_notes.append(
            CMCNote(
                paper_id=str(x.get("paper_id", "unknown")),
                topic=str(x.get("topic", "")),
                note=str(x.get("note", "")),
                risk_level=str(x.get("risk_level", "medium")),
            )
        )

    return report


def write_literature_outputs(
    report: LiteratureSynthesisReport,
    output_dir: str | Path,
) -> dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    evidence_json = out / "literature_evidence.json"
    synthesis_md = out / "literature_synthesis.md"
    rationale_csv = out / "treatment_rationale_table.csv"

    evidence_json.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True))

    md = "\n".join(
        [
            "# Literature Synthesis",
            "",
            report.synthesis_summary or "No summary provided.",
            "",
            "## Counts",
            f"- Protocol evidence: {len(report.protocol_evidence)}",
            f"- Reagent evidence: {len(report.reagent_evidence)}",
            f"- Mechanism evidence: {len(report.mechanism_evidence)}",
            f"- CMC notes: {len(report.cmc_notes)}",
            "",
        ]
    )
    synthesis_md.write_text(md)

    rows = [
        {
            "paper_id": x.paper_id,
            "treatment": x.treatment,
            "rationale": x.rationale,
            "confidence": x.confidence,
        }
        for x in report.protocol_evidence
    ]
    pd.DataFrame(rows).to_csv(rationale_csv, index=False)

    return {
        "literature_evidence.json": str(evidence_json),
        "literature_synthesis.md": str(synthesis_md),
        "treatment_rationale_table.csv": str(rationale_csv),
    }
