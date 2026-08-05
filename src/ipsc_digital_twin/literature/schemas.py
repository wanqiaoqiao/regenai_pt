from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class ProtocolEvidence:
    paper_id: str
    protocol_step: str
    treatment: str
    rationale: str
    confidence: float = 0.5


@dataclass(slots=True)
class ReagentEvidence:
    paper_id: str
    reagent_name: str
    dose: str
    duration: str
    purpose: str
    confidence: float = 0.5


@dataclass(slots=True)
class MechanismEvidence:
    paper_id: str
    pathway: str
    mechanism: str
    supports_treatment: str
    confidence: float = 0.5


@dataclass(slots=True)
class CMCNote:
    paper_id: str
    topic: str
    note: str
    risk_level: str = "medium"


@dataclass(slots=True)
class LiteratureSynthesisReport:
    protocol_evidence: list[ProtocolEvidence] = field(default_factory=list)
    reagent_evidence: list[ReagentEvidence] = field(default_factory=list)
    mechanism_evidence: list[MechanismEvidence] = field(default_factory=list)
    cmc_notes: list[CMCNote] = field(default_factory=list)
    synthesis_summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol_evidence": [asdict(x) for x in self.protocol_evidence],
            "reagent_evidence": [asdict(x) for x in self.reagent_evidence],
            "mechanism_evidence": [asdict(x) for x in self.mechanism_evidence],
            "cmc_notes": [asdict(x) for x in self.cmc_notes],
            "synthesis_summary": self.synthesis_summary,
        }
