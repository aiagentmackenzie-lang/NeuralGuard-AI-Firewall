"""NG-10: data model for per-tool MCP argument constraints.

Kept in its own module so the policy (``intent_gate``) and the evaluator
(``argument_gate``) can both depend on it without a cycle.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class ConstraintKind(StrEnum):
    """What a single argument constraint asserts."""

    REQUIRED = "required"
    KEY_ALLOWLIST = "key_allowlist"
    ENUM = "enum"
    MAX_LENGTH = "max_length"
    REGEX_DENY = "regex_deny"
    PATH_PREFIX = "path_prefix"
    PATH_GLOB = "path_glob"
    URL_HOST = "url_host"


class ConstraintAction(StrEnum):
    """What a violation of this constraint does. Never silently allows."""

    DENY = "deny"
    ESCALATE = "escalate"


class ArgumentConstraint(BaseModel):
    """One constraint over one tool argument.

    ``argument`` is a dotted path into the arguments object (``"path"``,
    ``"options.timeout"``). An empty string addresses the arguments object
    itself — used by ``key_allowlist``, which inspects top-level keys.
    """

    model_config = {"extra": "forbid"}

    argument: str = Field(
        default="",
        description="Dotted argument path; '' addresses the whole arguments object.",
    )
    kind: ConstraintKind
    values: list[str] = Field(
        default_factory=list,
        description=(
            "Allow/deny values, path prefixes, globs, URL hosts, or the length "
            "limit as one integer string (max_length)."
        ),
    )
    action: ConstraintAction = Field(
        default=ConstraintAction.DENY,
        description="Violation outcome. ESCALATE is refused fail-closed at the route.",
    )
    reason: str = Field(default="", description="Operator-facing explanation for findings.")
