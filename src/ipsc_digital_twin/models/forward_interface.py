from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from ..forward_model import CurrentStateProfile, TargetStateProfile
from ..forward_model import TransitionPrediction as PredictedState


@runtime_checkable
class ForwardTransitionModelInterface(Protocol):
    def fit(self, adata: Any, config: Any | None = None) -> ForwardTransitionModelInterface:
        ...

    def predict_future_state(
        self,
        current_state: CurrentStateProfile | dict[str, Any],
        round1_treatment: str,
        round2_treatment: str | None = None,
        iPSC_line: str | None = None,
    ) -> PredictedState:
        ...

    def save(self, path: str | Path) -> None:
        ...

    @classmethod
    def load(cls, path: str | Path) -> ForwardTransitionModelInterface:
        ...


__all__ = [
    "CurrentStateProfile",
    "ForwardTransitionModelInterface",
    "PredictedState",
    "TargetStateProfile",
]
