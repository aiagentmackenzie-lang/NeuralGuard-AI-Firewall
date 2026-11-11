"""NG-10: argument and resource constraints for MCP tool calls.

The Intent Gate (NG-8) decides *whether* a tool may be called, on headers, before
the body is read. It cannot decide *how* — ``read_file`` is either allowed or not,
regardless of the ``path`` argument. This module closes that gap: per-tenant,
per-tool constraints over the parsed JSON-RPC arguments, evaluated before the
call is forwarded upstream.

Design rules (all fail-closed):

- **Unknown or malformed input denies.** ``params.arguments`` that is not a JSON
  object, a non-string value where a scalar constraint expects one, or an
  evaluation error all DENY rather than skip the check.
- **Absent arguments are vacuous, except ``required``.** A ``path_prefix`` rule
  with no ``path`` argument does not fire; ``required`` exists to demand it.
  Without this split, every optional argument would need a paired rule and one
  omission would silently widen the policy.
- **Most-restrictive wins.** All rules are evaluated; DENY beats ESCALATE beats
  ALLOW, matching the Intent Gate's resolution order. A rule order cannot make a
  policy more permissive than its strictest matching rule.
- **Argument values never reach the audit trail.** Findings carry the argument
  *path*, the rule that fired, and the failure kind — never the value, which is
  routinely a path, a URL, or a secret.

Scope, stated plainly: these are **boundary checks on declared arguments**, not a
sandbox. They stop a confused or coerced agent from asking for the wrong thing;
they do not stop an upstream tool that ignores its own arguments. Path checks are
string-level (``..`` and prefix), not filesystem-level (no symlink resolution),
and must not be described as confinement.
"""

from __future__ import annotations

import fnmatch
import posixpath
import re as _stdlib_re
from typing import Any, cast
from urllib.parse import urlparse

import regex as _regex
from pydantic import BaseModel

from neuralguard.mcp.constraints import ArgumentConstraint, ConstraintAction, ConstraintKind
from neuralguard.mcp.intent_gate import McpToolPolicy, ToolAction

# ReDoS safety: the same 50 ms per-pattern budget the pattern layer uses. A
# pattern that cannot finish inside it is treated as a violation, not a pass.
_REGEX_TIMEOUT_MS = 50


class ArgumentDecision(BaseModel):
    """The outcome of evaluating a tool's argument constraints."""

    action: ToolAction = ToolAction.ALLOW
    rule_id: str = "MCP-ARG-000"
    reason: str = "no constraint matched"
    argument: str | None = None
    kind: str | None = None


# Constraint kind -> rule id, so findings are greppable in the audit trail.
_RULE_IDS: dict[ConstraintKind, str] = {
    ConstraintKind.REQUIRED: "MCP-ARG-001",
    ConstraintKind.KEY_ALLOWLIST: "MCP-ARG-002",
    ConstraintKind.ENUM: "MCP-ARG-003",
    ConstraintKind.MAX_LENGTH: "MCP-ARG-004",
    ConstraintKind.REGEX_DENY: "MCP-ARG-005",
    ConstraintKind.PATH_PREFIX: "MCP-ARG-006",
    ConstraintKind.PATH_GLOB: "MCP-ARG-007",
    ConstraintKind.URL_HOST: "MCP-ARG-008",
}
_TYPE_RULE_ID = "MCP-ARG-009"  # non-scalar where a scalar is required
_INPUT_RULE_ID = "MCP-ARG-010"  # arguments not an object / evaluation failed

_STRICTNESS: dict[ToolAction, int] = {
    ToolAction.ALLOW: 0,
    ToolAction.ESCALATE: 1,
    ToolAction.DENY: 2,
}

_ALLOWED_URL_SCHEMES = frozenset({"http", "https"})


class _ConstraintError(Exception):
    """Internal: one constraint failed. Carries the rule and no argument value."""

    def __init__(self, *, constraint: ArgumentConstraint, detail: str) -> None:
        super().__init__(detail)
        self.constraint = constraint
        self.detail = detail

    @property
    def kind(self) -> ConstraintKind:
        return self.constraint.kind

    @property
    def argument(self) -> str:
        return self.constraint.argument


def _resolve(arguments: dict[str, Any], path: str) -> tuple[bool, Any]:
    """Resolve a dotted path. Returns ``(present, value)``.

    Presence is distinct from a ``None`` value: a key explicitly set to null is
    present and fails scalar checks, while an absent key is vacuous.
    """
    if path == "":
        return True, arguments
    current: Any = arguments
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return False, None
        current = current[part]
    return True, current


def _require_string(value: Any, *, constraint: ArgumentConstraint) -> str:
    """Reject non-strings: a list or object where a path/URL is expected is a bypass."""
    if isinstance(value, bool) or not isinstance(value, str):
        raise _ConstraintError(
            constraint=constraint,
            detail=f"expected a string, got {type(value).__name__}",
        )
    return cast("str", value)


def _check_scalar(constraint: ArgumentConstraint, value: Any) -> None:
    """Apply one scalar (single-argument) constraint, raising on violation."""
    kind = constraint.kind

    if kind is ConstraintKind.MAX_LENGTH:
        text = _require_string(value, constraint=constraint)
        try:
            limit = int(constraint.values[0])
        except (IndexError, ValueError):
            raise _ConstraintError(
                constraint=constraint, detail="max_length needs one integer in values"
            ) from None
        if len(text) > limit:
            raise _ConstraintError(constraint=constraint, detail=f"length {len(text)} > {limit}")
        return

    text = _require_string(value, constraint=constraint)

    if kind is ConstraintKind.ENUM:
        if text not in constraint.values:
            raise _ConstraintError(constraint=constraint, detail="value not in enum")
    elif kind is ConstraintKind.REGEX_DENY:
        for pattern in constraint.values:
            try:
                if _regex.search(pattern, text, timeout=_REGEX_TIMEOUT_MS):
                    raise _ConstraintError(constraint=constraint, detail="matched deny pattern")
            except TimeoutError:
                # A pattern that cannot be evaluated in budget is a violation.
                raise _ConstraintError(constraint=constraint, detail="pattern timed out") from None
            except _stdlib_re.error:
                raise _ConstraintError(
                    constraint=constraint, detail="invalid deny pattern"
                ) from None
    elif kind is ConstraintKind.PATH_PREFIX:
        _check_path_prefix(text, constraint)
    elif kind is ConstraintKind.PATH_GLOB:
        normalized = _normalize_path(text, constraint=constraint)
        if not any(fnmatch.fnmatch(normalized, glob) for glob in constraint.values):
            raise _ConstraintError(constraint=constraint, detail="path outside allowed globs")
    elif kind is ConstraintKind.URL_HOST:
        _check_url_host(text, constraint)


def _normalize_path(text: str, *, constraint: ArgumentConstraint) -> str:
    """Normalize an argument path and refuse traversal outright.

    ``posixpath.normpath`` alone is not enough: ``a/../../b`` normalizes to
    ``../b``, which is still an escape. Any surviving ``..`` segment or a NUL
    byte is a violation regardless of the configured prefixes.
    """
    if "\x00" in text or "\\" in text:
        raise _ConstraintError(constraint=constraint, detail="path contains NUL or backslash")
    normalized = posixpath.normpath(text)
    if normalized == ".." or normalized.startswith("../") or "/../" in normalized:
        raise _ConstraintError(constraint=constraint, detail="path traversal")
    return normalized


def _check_path_prefix(text: str, constraint: ArgumentConstraint) -> None:
    normalized = _normalize_path(text, constraint=constraint)
    for prefix in constraint.values:
        clean = posixpath.normpath(prefix).rstrip("/") or "/"
        if clean == "/":
            if normalized.startswith("/"):
                return
            continue
        if normalized == clean or normalized.startswith(clean + "/"):
            return
    raise _ConstraintError(constraint=constraint, detail="path outside allowed prefixes")


def _check_url_host(text: str, constraint: ArgumentConstraint) -> None:
    parsed = urlparse(text)
    if parsed.scheme.lower() not in _ALLOWED_URL_SCHEMES:
        raise _ConstraintError(constraint=constraint, detail="scheme not http(s)")
    host = (parsed.hostname or "").lower().rstrip(".")
    if not host:
        raise _ConstraintError(constraint=constraint, detail="no host in URL")
    for allowed in constraint.values:
        candidate = allowed.lower().rstrip(".")
        if host == candidate or host.endswith("." + candidate):
            return
    raise _ConstraintError(constraint=constraint, detail="host not allowlisted")


def _violation_to_action(violation: _ConstraintError) -> ArgumentDecision:
    constraint = violation.constraint
    action = ToolAction.DENY if constraint.action is ConstraintAction.DENY else ToolAction.ESCALATE
    rule_id = (
        _TYPE_RULE_ID if "expected a string" in violation.detail else _RULE_IDS[violation.kind]
    )
    reason = constraint.reason or violation.detail
    return ArgumentDecision(
        action=action,
        rule_id=rule_id,
        reason=f"{violation.argument or 'arguments'}: {reason}",
        argument=violation.argument or None,
        kind=violation.kind.value,
    )


def evaluate_arguments(
    policy: McpToolPolicy,
    tool: str,
    arguments: Any,
) -> ArgumentDecision:
    """Evaluate a tenant's argument constraints for one ``tools/call``.

    Every constraint is evaluated; the strictest matching violation wins
    (DENY > ESCALATE > ALLOW) so declaration order cannot widen the policy.
    Returns ALLOW when the tool has no constraints, when every constraint
    passes, or when a non-required constraint's argument is absent. Never raises.
    """
    constraints = policy.tool_constraints.get(tool, [])
    if not constraints:
        return ArgumentDecision()

    if not isinstance(arguments, dict):
        return ArgumentDecision(
            action=ToolAction.DENY,
            rule_id=_INPUT_RULE_ID,
            reason=f"arguments must be a JSON object, got {type(arguments).__name__}",
            kind="malformed",
        )

    worst = ArgumentDecision()
    for constraint in constraints:
        try:
            if constraint.kind is ConstraintKind.KEY_ALLOWLIST:
                unexpected = sorted(set(arguments) - set(constraint.values))
                if unexpected:
                    # Names only — never values.
                    raise _ConstraintError(
                        constraint=constraint,
                        detail="unexpected argument key(s): " + ", ".join(unexpected[:5]),
                    )
                continue

            present, value = _resolve(arguments, constraint.argument)
            if constraint.kind is ConstraintKind.REQUIRED:
                empty = value is None or (isinstance(value, str) and not value.strip())
                if not present or empty:
                    raise _ConstraintError(
                        constraint=constraint, detail="required argument missing"
                    )
                continue

            if not present:
                continue  # vacuous: an absent optional argument cannot violate

            _check_scalar(constraint, value)
        except _ConstraintError as violation:
            candidate = _violation_to_action(violation)
            if _STRICTNESS[candidate.action] > _STRICTNESS[worst.action]:
                worst = candidate
        except Exception:  # pragma: no cover - defensive: never fail open
            return ArgumentDecision(
                action=ToolAction.DENY,
                rule_id=_INPUT_RULE_ID,
                reason="argument constraint evaluation failed",
                kind="error",
            )

    return worst
