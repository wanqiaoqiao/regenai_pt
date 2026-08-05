from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict
from typing import Any

from .schemas import CMCNote, MechanismEvidence, ProtocolEvidence, ReagentEvidence

LLMClient = Callable[[str], str]


class ExtractionError(ValueError):
    pass


def _safe_json_loads(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError as salvage_exc:
                raise ExtractionError("Invalid JSON from LLM after salvage attempt") from salvage_exc
        raise ExtractionError("Invalid JSON from LLM") from exc


def _coerce_protocol(items: list[dict[str, Any]], paper_id: str) -> list[ProtocolEvidence]:
    out = []
    for x in items:
        out.append(
            ProtocolEvidence(
                paper_id=paper_id,
                protocol_step=str(x.get("protocol_step", "")),
                treatment=str(x.get("treatment", "")),
                rationale=str(x.get("rationale", "")),
                confidence=float(x.get("confidence", 0.5)),
            )
        )
    return out


def _coerce_reagent(items: list[dict[str, Any]], paper_id: str) -> list[ReagentEvidence]:
    out = []
    for x in items:
        out.append(
            ReagentEvidence(
                paper_id=paper_id,
                reagent_name=str(x.get("reagent_name", "")),
                dose=str(x.get("dose", "")),
                duration=str(x.get("duration", "")),
                purpose=str(x.get("purpose", "")),
                confidence=float(x.get("confidence", 0.5)),
            )
        )
    return out


def _coerce_mechanism(items: list[dict[str, Any]], paper_id: str) -> list[MechanismEvidence]:
    out = []
    for x in items:
        out.append(
            MechanismEvidence(
                paper_id=paper_id,
                pathway=str(x.get("pathway", "")),
                mechanism=str(x.get("mechanism", "")),
                supports_treatment=str(x.get("supports_treatment", "")),
                confidence=float(x.get("confidence", 0.5)),
            )
        )
    return out


def _coerce_cmc(items: list[dict[str, Any]], paper_id: str) -> list[CMCNote]:
    out = []
    for x in items:
        out.append(
            CMCNote(
                paper_id=paper_id,
                topic=str(x.get("topic", "")),
                note=str(x.get("note", "")),
                risk_level=str(x.get("risk_level", "medium")),
            )
        )
    return out


def extract_paper_evidence(
    paper_id: str,
    paper_text: str,
    llm_client: LLMClient,
    max_chars: int = 8000,
) -> dict[str, Any]:
    clipped = paper_text[:max_chars]
    prompt = (
        "Extract structured treatment rationale and CMC notes as JSON with keys "
        "protocol_evidence,reagent_evidence,mechanism_evidence,cmc_notes. "
        "Each key should map to a list of objects."
        "\nPAPER_ID: "
        + paper_id
        + "\nTEXT:\n"
        + clipped
    )
    raw = llm_client(prompt)
    payload = _safe_json_loads(raw)

    protocol = _coerce_protocol(payload.get("protocol_evidence", []), paper_id)
    reagent = _coerce_reagent(payload.get("reagent_evidence", []), paper_id)
    mechanism = _coerce_mechanism(payload.get("mechanism_evidence", []), paper_id)
    cmc = _coerce_cmc(payload.get("cmc_notes", []), paper_id)

    return {
        "paper_id": paper_id,
        "protocol_evidence": [asdict(x) for x in protocol],
        "reagent_evidence": [asdict(x) for x in reagent],
        "mechanism_evidence": [asdict(x) for x in mechanism],
        "cmc_notes": [asdict(x) for x in cmc],
    }


def map_extract_papers(
    papers: list[dict[str, str]],
    llm_client: LLMClient,
    max_chars: int = 8000,
) -> list[dict[str, Any]]:
    outputs: list[dict[str, Any]] = []
    for paper in papers:
        pid = paper.get("paper_id", "unknown")
        text = paper.get("text", "")
        outputs.append(extract_paper_evidence(pid, text, llm_client=llm_client, max_chars=max_chars))
    return outputs
