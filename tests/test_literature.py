from __future__ import annotations

import json
from pathlib import Path

import pytest

from ipsc_digital_twin.literature import (
    ExtractionError,
    map_extract_papers,
    synthesize_literature,
    write_literature_outputs,
)


class MockLLM:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if "PAPER_ID:" in prompt:
            return json.dumps(
                {
                    "protocol_evidence": [
                        {
                            "protocol_step": "Induce endoderm",
                            "treatment": "Activin A",
                            "rationale": "Supports definitive endoderm",
                            "confidence": 0.9,
                        }
                    ],
                    "reagent_evidence": [
                        {
                            "reagent_name": "Activin A",
                            "dose": "100 ng/mL",
                            "duration": "48h",
                            "purpose": "Endoderm induction",
                            "confidence": 0.8,
                        }
                    ],
                    "mechanism_evidence": [
                        {
                            "pathway": "TGF_beta",
                            "mechanism": "SMAD2/3 activation",
                            "supports_treatment": "Activin A",
                            "confidence": 0.85,
                        }
                    ],
                    "cmc_notes": [
                        {
                            "topic": "Reagent consistency",
                            "note": "Batch potency should be monitored",
                            "risk_level": "medium",
                        }
                    ],
                }
            )

        return json.dumps(
            {
                "synthesis_summary": "Activin A evidence is consistent across papers.",
                "protocol_evidence": [
                    {
                        "paper_id": "p1",
                        "protocol_step": "Induce endoderm",
                        "treatment": "Activin A",
                        "rationale": "Endoderm support",
                        "confidence": 0.9,
                    }
                ],
                "reagent_evidence": [],
                "mechanism_evidence": [],
                "cmc_notes": [],
            }
        )


class BadJSONLLM:
    def __call__(self, prompt: str) -> str:
        return "not-json"


def test_map_reduce_literature_flow_and_outputs(tmp_path: Path) -> None:
    papers = [
        {"paper_id": "p1", "text": "Protocol uses Activin A for endoderm induction."},
        {"paper_id": "p2", "text": "CMC considerations mention batch-to-batch variability."},
    ]
    llm = MockLLM()

    extracted = map_extract_papers(papers, llm_client=llm, max_chars=120)
    assert len(extracted) == 2
    assert extracted[0]["paper_id"] == "p1"
    assert "protocol_evidence" in extracted[0]

    report = synthesize_literature(extracted, llm_client=llm)
    assert "Activin A" in report.synthesis_summary or len(report.protocol_evidence) > 0

    paths = write_literature_outputs(report, tmp_path)
    assert Path(paths["literature_evidence.json"]).exists()
    assert Path(paths["literature_synthesis.md"]).exists()
    assert Path(paths["treatment_rationale_table.csv"]).exists()


def test_invalid_json_raises_extraction_error() -> None:
    papers = [{"paper_id": "p_bad", "text": "some text"}]
    with pytest.raises(ExtractionError):
        map_extract_papers(papers, llm_client=BadJSONLLM(), max_chars=100)


def test_max_chars_clipping_applied() -> None:
    llm = MockLLM()
    long_text = "A" * 500
    papers = [{"paper_id": "p_long", "text": long_text}]
    _ = map_extract_papers(papers, llm_client=llm, max_chars=50)
    prompt = llm.prompts[0]
    assert len(prompt) < 500
