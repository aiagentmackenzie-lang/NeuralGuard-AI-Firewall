"""Operator audit-chain verification (F14).

``verify_audit_files`` loads JSONL audit files, groups events per worker
chain (a naive single-chain verify over an interleaved multi-worker file
FAILS BY DESIGN — hash chains are per-process, P2-10 tracks cross-worker
ordering + signing), and verifies each chain with
``neuralguard.logging.chain.verify_chain``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from pathlib import Path

from neuralguard.logging.chain import verify_chain
from neuralguard.models.schemas import AuditEvent

logger = structlog.get_logger(__name__)


@dataclass
class ChainReport:
    worker_id: str
    event_count: int
    valid: bool


@dataclass
class AuditVerifyReport:
    files_read: int
    events_parsed: int
    parse_errors: int
    chains: list[ChainReport]

    @property
    def all_valid(self) -> bool:
        return self.parse_errors == 0 and all(c.valid for c in self.chains)

    def to_dict(self) -> dict[str, object]:
        return {
            "files_read": self.files_read,
            "events_parsed": self.events_parsed,
            "parse_errors": self.parse_errors,
            "all_valid": self.all_valid,
            "chains": [
                {
                    "worker_id": c.worker_id,
                    "events": c.event_count,
                    "valid": c.valid,
                }
                for c in self.chains
            ],
        }


def _audit_files(target: Path) -> list[Path]:
    """Expand a file or directory into the audit JSONL files to verify."""
    if target.is_dir():
        return sorted(target.rglob("*.jsonl"))
    if target.suffix != ".jsonl":
        raise ValueError(f"not a .jsonl audit file: {target}")
    return [target]


def verify_audit_files(target: Path) -> AuditVerifyReport:
    """Load + group + verify every per-worker chain under ``target``.

    Files are read in sorted order (daily rotation names sort
    chronologically); events group by ``worker_id`` across files so a chain
    spanning a rotation boundary is still verified end-to-end.
    """
    files = _audit_files(target)
    chains: dict[str, list[AuditEvent]] = {}
    events_parsed = 0
    parse_errors = 0

    for path in files:
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    event = AuditEvent.model_validate_json(line)
                except Exception:
                    parse_errors += 1
                    logger.warning("audit_verify_parse_error", file=str(path), line=line_no)
                    continue
                events_parsed += 1
                worker = event.worker_id or "<unknown-worker>"
                chains.setdefault(worker, []).append(event)

    report = AuditVerifyReport(
        files_read=len(files),
        events_parsed=events_parsed,
        parse_errors=parse_errors,
        chains=[],
    )
    for worker_id in sorted(chains):
        events = chains[worker_id]
        report.chains.append(
            ChainReport(
                worker_id=worker_id,
                event_count=len(events),
                valid=verify_chain(events),
            )
        )
    return report
