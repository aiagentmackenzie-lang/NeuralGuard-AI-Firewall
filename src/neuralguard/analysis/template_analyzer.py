"""Prompt-template analyzer — static injection-sink analysis (Sprint B, B2).

A "shift-left" counterpart to runtime detection: statically scans a
system-prompt template for injection sinks BEFORE deployment. No LLM call
— pure static analysis, fast, CI-able. Invoked from the
``neuralguard analyze-template`` CLI and the ``POST /v1/analyze/template``
endpoint.

Sink classes (each with severity + remediation):
  - TPL-SINK-001 (HIGH)   untrusted variable interpolated into the system
                          prompt (the whole template IS the privileged
                          context, so any user-controlled var is a sink).
  - TPL-SINK-002 (MEDIUM) unbounded variable — a placeholder with no
                          surrounding validation / delimiter guidance.
  - TPL-SINK-003 (MEDIUM) missing delimiter fence — system/user content
                          mixed without a clear boundary (ambiguous
                          instruction precedence).
  - TPL-SINK-004 (MEDIUM) ambiguous instruction precedence — multiple
                          System:/Instructions: headers, or a user-content
                          placeholder ordered before system instructions.
  - TPL-SINK-005 (HIGH)   variable adjacent to a privileged-action keyword
                          (tool/function/execute/shell/eval/run/delete) —
                          tool-misuse surface.
  - TPL-SINK-006 (LOW)    structured-data variable injected raw
                          ({{json}}/{{xml}}/{{html}}) — parsing/injection
                          surface.

Design:
  - Pure function: ``TemplateAnalyzer().analyze(template) -> list[TemplateSink]``.
  - No config, no LLM, no network. Deterministic and CI-able.
  - Placeholder syntaxes recognized: ``{{ var }}``, ``${ var }``, ``{ var }``,
    and explicit ``<user_input>``-style angle-bracket placeholders.
  - "Untrusted" variable name heuristic: a known set of names that typically
    carry attacker-controlled content (input, user_input, query, document,
    rag, context, ...). Trusted/app-controlled names (language, date, role)
    are not flagged.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# ── Sink model ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class TemplateSink:
    """A static injection sink found in a prompt template."""

    rule_id: str
    severity: str  # "high" | "medium" | "low" | "info"
    description: str
    remediation: str
    evidence: str  # the matched snippet / variable name
    location: int  # 1-indexed line number


@dataclass(frozen=True)
class _Placeholder:
    """A discovered variable placeholder in a template."""

    name: str
    raw: str
    line: int
    untrusted: bool
    structured: bool


@dataclass
class TemplateAnalysisResult:
    """Result of analyzing a template."""

    template: str
    sinks: list[TemplateSink] = field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        return not self.sinks

    def to_dict(self) -> dict[str, object]:
        return {
            "template": self.template,
            "is_clean": self.is_clean,
            "sink_count": len(self.sinks),
            "sinks": [
                {
                    "rule_id": s.rule_id,
                    "severity": s.severity,
                    "description": s.description,
                    "remediation": s.remediation,
                    "evidence": s.evidence,
                    "location": s.location,
                }
                for s in self.sinks
            ],
        }


# ── Heuristics ─────────────────────────────────────────────────────────────

# Variable placeholders: {{ var }}, ${ var }, { var }. Capture the name.
# Single-brace {var} is intentionally included but the untrusted-name
# heuristic filters JSON-like noise.
_PLACEHOLDER_RE = re.compile(r"\{\{\s*(\w+)\s*\}\}|\$\{\s*(\w+)\s*\}|\{\s*(\w+)\s*\}")
# Explicit angle-bracket user-content markers: <user_input>, <user_content>...
_ANGLE_USER_RE = re.compile(
    r"<\s*(user_input|user_content|user_message|user_text|input|query|payload|untrusted)\s*>", re.I
)

# Variable names that typically carry attacker-controlled content.
_UNTRUSTED_NAMES = {
    "input",
    "user_input",
    "user",
    "query",
    "content",
    "data",
    "document",
    "rag",
    "context",
    "untrusted",
    "payload",
    "message",
    "text",
    "request",
    "prompt",
    "user_message",
    "user_content",
    "search",
    "result",
    "chunk",
    "fetch",
    "response",
    "email",
    "attachment",
    "webpage",
    "page",
    "article",
    "comment",
    "review",
    "transcript",
    "conversation",
    "history",
}

# Structured-data variable names (parsing/injection surface when raw).
_STRUCTURED_NAMES = {"json", "xml", "html", "yaml", "csv", "sql", "markdown", "md"}

# Privileged-action keywords — a variable adjacent to one is a tool-misuse sink.
_ACTION_KEYWORDS = (
    "tool",
    "function",
    "execute",
    "exec",
    "run",
    "shell",
    "eval",
    "delete",
    "system",
    "subprocess",
    "command",
    "cmd",
    "bash",
    "powershell",
    "rm ",
    "wget",
    "curl",
    "sql",
    "query",
)

# System / instruction header markers (for precedence + fence checks).
_HEADER_RE = re.compile(r"(?im)^\s*(system|instructions?|admin|context|user|assistant)\s*[:\-]\s")
_DELIMITER_RE = re.compile(r"(?m)^\s*(?:===+|---+|~~~+|```+|<{3,}|>{3,})")

# Trusted/app-controlled variable names (not flagged as untrusted).
_TRUSTED_NAMES = {
    "language",
    "date",
    "time",
    "role",
    "name",
    "persona",
    "tone",
    "style",
    "format",
    "locale",
    "timezone",
    "version",
    "model",
    "today",
    "now",
}


class TemplateAnalyzer:
    """Static prompt-template injection-sink analyzer.

    Usage::

        analyzer = TemplateAnalyzer()
        result = analyzer.analyze(template_string)
        for sink in result.sinks:
            print(sink.rule_id, sink.severity, sink.description)
    """

    def analyze(self, template: str) -> TemplateAnalysisResult:
        if not template or not template.strip():
            return TemplateAnalysisResult(template=template, sinks=[])

        sinks: list[TemplateSink] = []
        lines = template.splitlines()

        placeholders = self._find_placeholders(template, lines)
        sinks.extend(self._untrusted_in_system_sinks(placeholders))
        sinks.extend(self._unbounded_variable_sinks(placeholders))
        sinks.extend(self._structured_data_sinks(placeholders))
        sinks.extend(self._action_adjacent_sinks(template, lines, placeholders))
        sinks.extend(self._missing_fence_sinks(template, lines))
        sinks.extend(self._ambiguous_precedence_sinks(template, lines))

        # Dedupe by (rule_id, evidence, location); keep stable order.
        seen: set[tuple[str, str, int]] = set()
        deduped: list[TemplateSink] = []
        for s in sinks:
            key = (s.rule_id, s.evidence, s.location)
            if key not in seen:
                seen.add(key)
                deduped.append(s)
        return TemplateAnalysisResult(template=template, sinks=deduped)

    # ── Placeholder discovery ────────────────────────────────────────────

    def _find_placeholders(self, template: str, lines: list[str]) -> list[_Placeholder]:
        found: list[_Placeholder] = []
        for m in _PLACEHOLDER_RE.finditer(template):
            name = next((g for g in m.groups() if g), "")
            if not name:
                continue
            line = self._line_for_offset(template, m.start(), lines)
            lower = name.lower()
            found.append(
                _Placeholder(
                    name=lower,
                    raw=m.group(0),
                    line=line,
                    untrusted=lower in _UNTRUSTED_NAMES and lower not in _TRUSTED_NAMES,
                    structured=lower in _STRUCTURED_NAMES,
                )
            )
        for m in _ANGLE_USER_RE.finditer(template):
            line = self._line_for_offset(template, m.start(), lines)
            found.append(
                _Placeholder(
                    name=m.group(1).lower(),
                    raw=m.group(0),
                    line=line,
                    untrusted=True,
                    structured=False,
                )
            )
        return found

    @staticmethod
    def _line_for_offset(text: str, offset: int, lines: list[str]) -> int:
        # 1-indexed line number for a character offset.
        line = 1
        pos = 0
        for ln in lines:
            if pos + len(ln) + 1 > offset:  # +1 for the newline
                return line
            pos += len(ln) + 1
            line += 1
        return line

    # ── Sinks ────────────────────────────────────────────────────────────

    def _untrusted_in_system_sinks(self, placeholders: list[_Placeholder]) -> list[TemplateSink]:
        sinks: list[TemplateSink] = []
        for p in placeholders:
            if p.untrusted:
                sinks.append(
                    TemplateSink(
                        rule_id="TPL-SINK-001",
                        severity="high",
                        description=(
                            f"Untrusted variable '{p.name}' interpolated into the "
                            "system prompt template. The whole template is the privileged "
                            "context, so attacker-controlled content here is a direct "
                            "prompt-injection sink."
                        ),
                        remediation=(
                            "Do not interpolate untrusted content into the system prompt. "
                            "Move it to a delimited user message, validate/sanitize it, or "
                            "wrap it in a quoted data fence the model is instructed to treat "
                            "as data only."
                        ),
                        evidence=p.raw,
                        location=p.line,
                    )
                )
        return sinks

    def _unbounded_variable_sinks(self, placeholders: list[_Placeholder]) -> list[TemplateSink]:
        sinks: list[TemplateSink] = []
        for p in placeholders:
            # Skip if already flagged as untrusted-in-system (higher severity).
            if p.untrusted:
                continue
            # Flag any placeholder that is not explicitly trusted (unknown
            # names are unbounded-by-default) and has no surrounding fence
            # hint in the evidence.
            if p.name in _TRUSTED_NAMES:
                continue
            sinks.append(
                TemplateSink(
                    rule_id="TPL-SINK-002",
                    severity="medium",
                    description=(
                        f"Variable '{p.name}' is interpolated without a visible validation "
                        "or delimiter guard. Unknown variables should be bounded explicitly."
                    ),
                    remediation=(
                        "Validate the variable against an allowlist/format before "
                        "interpolation, and wrap it in a delimited data section the model "
                        "treats as untrusted input."
                    ),
                    evidence=p.raw,
                    location=p.line,
                )
            )
        return sinks

    def _structured_data_sinks(self, placeholders: list[_Placeholder]) -> list[TemplateSink]:
        sinks: list[TemplateSink] = []
        for p in placeholders:
            if p.structured:
                sinks.append(
                    TemplateSink(
                        rule_id="TPL-SINK-006",
                        severity="low",
                        description=(
                            f"Structured-data variable '{p.name}' injected raw. Parsed "
                            "structures (json/xml/html) can carry embedded instructions "
                            "(a parser-confusion / injection surface)."
                        ),
                        remediation=(
                            "Parse and re-serialize structured data before interpolation, "
                            "or strip instruction-like content from it. Do not inject raw."
                        ),
                        evidence=p.raw,
                        location=p.line,
                    )
                )
        return sinks

    def _action_adjacent_sinks(
        self, template: str, lines: list[str], placeholders: list[_Placeholder]
    ) -> list[TemplateSink]:
        sinks: list[TemplateSink] = []
        for p in placeholders:
            line_idx = p.line - 1
            if line_idx < 0 or line_idx >= len(lines):
                continue
            line_text = lines[line_idx].lower()
            if any(kw in line_text for kw in _ACTION_KEYWORDS):
                sinks.append(
                    TemplateSink(
                        rule_id="TPL-SINK-005",
                        severity="high",
                        description=(
                            f"Variable '{p.name}' appears on a line with a privileged-action "
                            "keyword (tool/function/execute/shell/...). An attacker can craft "
                            "the variable to hijack the action — a tool-misuse sink."
                        ),
                        remediation=(
                            "Separate untrusted data from action keywords. Validate the "
                            "variable against the action's expected schema and reject "
                            "instruction-like content."
                        ),
                        evidence=f"{p.raw} on: {lines[line_idx].strip()[:80]}",
                        location=p.line,
                    )
                )
        return sinks

    def _missing_fence_sinks(self, template: str, lines: list[str]) -> list[TemplateSink]:
        # If the template references user content (a user/instructions header
        # OR an untrusted placeholder) but has NO delimiter fence at all,
        # instruction precedence is ambiguous.
        has_user_header = any(
            (m := _HEADER_RE.match(ln)) is not None
            and m.group(1).lower() in {"user", "instructions", "instruction"}
            for ln in lines
        )
        has_fence = any(_DELIMITER_RE.match(ln) for ln in lines)
        has_untrusted = bool(_ANGLE_USER_RE.search(template)) or any(
            m.group(0)
            for m in _PLACEHOLDER_RE.finditer(template)
            if next((g for g in m.groups() if g), "").lower() in _UNTRUSTED_NAMES
        )
        if (has_user_header or has_untrusted) and not has_fence:
            return [
                TemplateSink(
                    rule_id="TPL-SINK-003",
                    severity="medium",
                    description=(
                        "Template mixes system and user content without a delimiter fence "
                        "(no --- / === / ``` / <<< boundary). Instruction precedence is "
                        "ambiguous — an attacker can blur which instructions are authoritative."
                    ),
                    remediation=(
                        "Delimit system instructions and user content with a clear fence "
                        "(e.g. --- BEGIN SYSTEM --- / --- END SYSTEM ---) and instruct the "
                        "model to treat content inside the user fence as untrusted data."
                    ),
                    evidence="(no delimiter fence found)",
                    location=1,
                )
            ]
        return []

    def _ambiguous_precedence_sinks(self, template: str, lines: list[str]) -> list[TemplateSink]:
        sinks: list[TemplateSink] = []
        headers: list[tuple[int, str]] = []
        for i, ln in enumerate(lines):
            m = _HEADER_RE.match(ln)
            if m is not None:
                headers.append((i + 1, m.group(1).lower()))
        # Multiple system/instruction headers.
        sys_headers = [
            (ln, h)
            for ln, h in headers
            if h in {"system", "instructions", "instruction", "admin", "context"}
        ]
        if len(sys_headers) >= 2:
            sinks.append(
                TemplateSink(
                    rule_id="TPL-SINK-004",
                    severity="medium",
                    description=(
                        f"{len(sys_headers)} system/instruction headers found. Multiple "
                        "privileged headers make instruction precedence ambiguous (which one "
                        "wins on conflict?)."
                    ),
                    remediation=(
                        "Consolidate into a single system-instruction block. If multiple "
                        "contexts are needed, order them explicitly and document precedence."
                    ),
                    evidence=", ".join(f"line {ln}:{h}" for ln, h in sys_headers),
                    location=sys_headers[0][0],
                )
            )
        # A user header before any system header → user content may override
        # system instructions (precedence inversion).
        first_sys = next(
            (ln for ln, h in headers if h in {"system", "instructions", "instruction"}), None
        )
        first_user = next((ln for ln, h in headers if h == "user"), None)
        if first_user is not None and first_sys is not None and first_user < first_sys:
            sinks.append(
                TemplateSink(
                    rule_id="TPL-SINK-004",
                    severity="medium",
                    description=(
                        "A 'user' header appears before the first 'system'/'instructions' "
                        "header. User content ordered before system instructions inverts "
                        "precedence — later user content can be framed as authoritative."
                    ),
                    remediation=(
                        "Put system instructions first, then user content, delimited by a "
                        "fence. The model should treat system instructions as highest "
                        "precedence."
                    ),
                    evidence=f"user header line {first_user} before system line {first_sys}",
                    location=first_user,
                )
            )
        return sinks
