"""Base action class and ActionResult data model."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from neuralguard.config.settings import NeuralGuardConfig
    from neuralguard.models.schemas import (
        EvaluateRequest,
        LayerArbitrationResult,
        ScanOutputRequest,
    )


@dataclass
class ActionResult:
    """Result of executing a response action."""

    status_code: int
    body: dict[str, Any]
    headers: dict[str, str] = field(default_factory=dict)
    audit_event: dict[str, Any] = field(default_factory=dict)


class BaseAction(ABC):
    """Abstract base for response action handlers."""

    def __init__(self, config: NeuralGuardConfig) -> None:
        self.config = config

    @abstractmethod
    def execute(
        self,
        arbitration: LayerArbitrationResult,
        request: EvaluateRequest | ScanOutputRequest,
    ) -> ActionResult:
        """Execute the action."""
        ...
