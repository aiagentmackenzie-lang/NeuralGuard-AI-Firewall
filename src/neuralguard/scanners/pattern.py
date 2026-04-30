"""Pattern detection engine — Layer 2.

50+ compiled regex patterns across 8 threat categories:
  T-PI-D  — Direct Prompt Injection
  T-PI-I  — Indirect Prompt Injection
  T-JB    — Jailbreak & Role Hijacking
  T-EXF   — Data Exfiltration & PII Leakage
  T-TOOL  — Tool Misuse & MCP Poisoning
  T-ENC   — Encoding Evasion (supplementary to structural)
  T-DOS   — Reasoning DoS / Cost Abuse
  T-EXT   — System Prompt Extraction

Design principles:
  - Every regex compiled with 50ms timeout (ReDoS safety)
  - Possessive quantifiers (++, *+) preferred over greedy
  - All patterns fuzz-tested before merge
  - Patterns are independently toggled per-tenant
  - Target latency: <5ms for full pattern sweep
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

# ── ReDoS Safety ───────────────────────────────────────────────────────────
# All regex patterns are compiled with a timeout to prevent catastrophic
# backtracking. The `regex` library supports per-pattern timeouts natively.
# We use `regex` instead of `re` for this capability.
import regex as re_module  # regex lib supports timeout in search/match

from neuralguard.models.schemas import (
    EvaluateRequest,
    Finding,
    ScanLayer,
    ScannerResult,
    Severity,
    ThreatCategory,
    Verdict,
)
from neuralguard.scanners.base import BaseScanner
from neuralguard.scanners.pattern_i18n import I18N_FLAT

if TYPE_CHECKING:
    from neuralguard.config.settings import ScannerSettings

REGEX_TIMEOUT_MS = 50  # From SRD: 50ms per-pattern timeout


# ═══════════════════════════════════════════════════════════════════════════
# T-PI-D: Direct Prompt Injection (Rule IDs: PI-D-001 to PI-D-010)
# ═══════════════════════════════════════════════════════════════════════════

PI_D_PATTERNS: list[tuple[str, Severity, float, str, str]] = [
    # (rule_id, severity, confidence, description, pattern)
    (
        "PI-D-001",
        Severity.HIGH,
        0.95,
        "Instruction override — 'ignore/disregard/forget previous/all instructions'",
        r"(?i)\b(ignore|disregard|forget|override|bypass|skip)\s+(?:all\s+)?(?:the\s+|your\s+)?(?:previous|prior|above|earlier|earlier\s+given|initial|original|above\s+stated|your)\s+(?:instructions?|rules?|directives?|prompts?|guidelines?|constraints?)\b",
    ),
    (
        "PI-D-002",
        Severity.HIGH,
        0.92,
        "New instruction injection — 'new instruction: / from now on / you are now'",
        r"(?i)(?:new\s+instruction|from\s+now\s+on|starting\s+now|from\s+this\s+point|effective\s+immediately)\s*[:\-—]\s*",
    ),
    (
        "PI-D-003",
        Severity.HIGH,
        0.93,
        "Role switching — 'you are now / act as / pretend you are / roleplay as'",
        r"(?i)\byou\s+are\s+now\b|\bact\s+as\s+(?:if\s+you\s+are|a|an|the)\b|\bpretend\s+(?:you\s+are|to\s+be|that\s+you)\b|\broleplay\s+as\b|\bpretend\s+to\s+be\b|\bassume\s+the\s+(?:role|identity|persona)\s+of\b",
    ),
    (
        "PI-D-004",
        Severity.MEDIUM,
        0.80,
        "System prompt delimiter injection — '=== / ### / ---' with instruction context",
        r"(?i)(?:===+|---+|~~~+|```+)\s*\n?.*?(?:system|instruction|admin|override|ignore)\b",
    ),
    (
        "PI-D-005",
        Severity.MEDIUM,
        0.75,
        "Instruction meta-injection — 'the above text contains / important: note:'",
        r"(?i)\b(?:the\s+above\s+text|this\s+document|this\s+message|the\s+following)\s+(?:contains?\s+)?(?:important|critical|urgent)\s*(?:instructions?|information|notice)\b",
    ),
    (
        "PI-D-006",
        Severity.HIGH,
        0.90,
        "Authority impersonation — 'as your creator / I am your admin / OpenAI said'",
        r"(?i)\b(?:as\s+your\s+(?:creator|developer|admin|owner|programmer|engineer)|i\s+am\s+your\s+(?:creator|developer|admin|owner|programmer|engineer)|(?:openai|anthropic|google)\s+(?:said|told|instructed|directed|updated)\b)",
    ),
    (
        "PI-D-007",
        Severity.MEDIUM,
        0.78,
        "Constraint removal — 'do not follow / you no longer need to / without any restrictions'",
        r"(?i)\b(?:do\s+not\s+follow|don'?t\s+follow|you\s+(?:no\s+longer|don'?t\s+need\s+to)\s+(?:follow|obey|adhere|comply|respect)|without\s+(?:any\s+)?restrictions?|no\s+(?:longer\s+)?(?:bound|constrained|limited)\s+by)\b",
    ),
    (
        "PI-D-008",
        Severity.HIGH,
        0.88,
        "Output manipulation — 'output your / print your / reveal your / show your' + instructions/prompt",
        r"(?i)\b(?:output|print|reveal|show|display|repeat|recite|echo)\s+(?:your|the|all)\s+(?:instructions?|prompt|system\s+(?:prompt|instructions?|message)|rules?|directives?|constraints?|training|initial\s+(?:instructions?|prompt))\b",
    ),
    (
        "PI-D-009",
        Severity.MEDIUM,
        0.72,
        "Hypothetical framing for injection — 'in a hypothetical / pretend scenario / imagine if'",
        r"(?i)\b(?:in\s+a\s+)?hypothetical(?:ly)?\s+(?:scenario|world|situation|context|where|in\s+which)\b|\bimagine\s+(?:if|that|a|you|we're)\b|\bsuppose\s+(?:that|you|we)\b",
    ),
    (
        "PI-D-010",
        Severity.MEDIUM,
        0.70,
        "Chain-of-thought manipulation — 'let's think step by step to bypass / reason about how to'",
        r"(?i)\b(?:let'?s?\s+think|think\s+about\s+how|reason\s+about\s+how)\s+(?:step\s+by\s+step|carefully|logically)\s+.{0,30}(?:bypass|circumvent|evade|avoid|break|override|ignore)\b",
    ),
]


# ═══════════════════════════════════════════════════════════════════════════
# T-PI-I: Indirect Prompt Injection (Rule IDs: PI-I-001 to PI-I-006)
# ═══════════════════════════════════════════════════════════════════════════

PI_I_PATTERNS: list[tuple[str, Severity, float, str, str]] = [
    (
        "PI-I-001",
        Severity.HIGH,
        0.85,
        "Hidden instruction markers — '<!-- / /* / {system}' in content",
        r"(?i)(?:<!--\s*(?:system|instruction|ignore|override)|/\*\s*(?:system|instruction|ignore|override)|\{(?:system|instruction):\s*(?:ignore|override))",
    ),
    (
        "PI-I-002",
        Severity.HIGH,
        0.88,
        "RAG chunk injection — '[INSTRUCTIONS] / <instruction> / --- BEGIN SYSTEM'",
        r"(?i)(?:\[(?:instructions?|system|admin|hidden|secret|important)\]|\{(?:instructions?|system|admin|hidden)\}|<(?:instructions?|system|admin|hidden)[^>]*>|---\s*(?:begin|end)\s+(?:system|instructions?|admin))",
    ),
    (
        "PI-I-003",
        Severity.MEDIUM,
        0.75,
        "Email/document embedded instruction — 'per the attached: / according to this document:'",
        r"(?i)\b(?:per\s+the\s+(?:attached|following|included)|according\s+to\s+this\s+(?:document|email|file|message)|as\s+(?:stated|mentioned|noted|requested)\s+in\s+the\s+(?:attached|document|file))\s*[:\-]?\b",
    ),
    (
        "PI-I-004",
        Severity.MEDIUM,
        0.70,
        "White-text / invisible content markers — 'color: white / font-size: 0'",
        r"(?i)(?:color\s*:\s*(?:white|#fff(?:fff)?|#ffffff)\s*(?:!important)?|font-size\s*:\s*0(?:px)?\s*(?:!important)?|opacity\s*:\s*0(?:\.0)?\s*(?:!important)?|visibility\s*:\s*hidden\s*(?:!important)?)",
    ),
    (
        "PI-I-005",
        Severity.HIGH,
        0.82,
        "Chunk boundary attack — '---END--- / ===BREAK=== / <EOC> next instruction'",
        r"(?i)(?:---+\s*(?:end|eof|eoc|break|stop)\s*---+|===+\s*(?:end|eof|eoc|break)\s*===+|<\s*(?:end|eof|eoc)\s*>)\s*\n?\s*(?:now|next|then|please|you\s+must)\b",
    ),
    (
        "PI-I-006",
        Severity.MEDIUM,
        0.68,
        "Metadata injection — 'X-Instruction / meta: / data-instruction'",
        r"(?i)(?:x-(?:instruction|system|override|command|action)|meta\s*:\s*(?:instruction|system|command|action)|data-(?:instruction|system|override|action))\s*[:=]\s*",
    ),
]


# ═══════════════════════════════════════════════════════════════════════════
# T-JB: Jailbreak & Role Hijacking (Rule IDs: JB-001 to JB-012)
# ═══════════════════════════════════════════════════════════════════════════

JB_PATTERNS: list[tuple[str, Severity, float, str, str]] = [
    (
        "JB-001",
        Severity.HIGH,
        0.96,
        "DAN jailbreak variant — 'DAN / Do Anything Now'",
        r"(?i)\bDAN\b(?:\s+(?:mode|jailbreak|enabled|activated|v\d+)?)?\b|\bdo\s+anything\s+now\b|\byou\s+are\s+DAN\b",
    ),
    (
        "JB-002",
        Severity.HIGH,
        0.94,
        "Developer mode activation — 'developer mode / debug mode / admin mode'",
        r"(?i)\b(?:developer|debug|admin|god|root|superuser|sudo|elevated)\s+mode\b(?:\s+(?:enabled|activated|on|engaged))?",
    ),
    (
        "JB-003",
        Severity.HIGH,
        0.91,
        "Grandma/exploit jailbreak — 'my grandma used to / for a dying'",
        r"(?i)\bmy\s+(?:grandma|grandfather|grandmother|grandpa|nana)\s+(?:used\s+to|would|always|loved)\s+(?:tell|say|explain|describe|make|create|show)\b|\bfor\s+(?:a\s+)?dying\s+(?:wish|request|person|grandma|grandpa|friend|child)\b",
    ),
    (
        "JB-004",
        Severity.HIGH,
        0.89,
        "Authorization framing — 'I'm a researcher / for educational purposes / as a pentester'",
        r"(?i)\b(?:i'?m?\s+a\s+(?:security\s+)?(?:researcher|pentester|red\s+teamer|ethical\s+hacker|cybersecurity\s+student)|for\s+(?:educational|academic|research|security\s+testing|pentesting)\s+purposes?|authorized\s+(?:penetration|security)\s+test)\b",
    ),
    (
        "JB-005",
        Severity.MEDIUM,
        0.72,
        "Many-shot jailbreak — repeated Q&A normalizing compliance",
        r"(?i)(?:user:\s*.{5,30}\s*assistant:\s*sure[,!]\s*){3,}",
    ),
    (
        "JB-006",
        Severity.MEDIUM,
        0.75,
        "Crescendo escalation — incremental boundary pushing",
        r"(?i)(?:great[,.!]\s+now\s+|good[,.!]\s+now\s+|perfect[,.!]\s+now\s+|that'?s?\s+right[,.!]\s+now\s+|correct[,.!]\s+now\s+){2,}",
    ),
    (
        "JB-007",
        Severity.HIGH,
        0.87,
        "Persona substitution — 'you are [named persona] / STAN / KEVIN'",
        r"(?i)\byou\s+are\s+(?:now\s+)?(?:STAN|KEVIN|AIM|DUDE|BISH|CAMY|KITT|JASON|MALFOY|MAX|EVA)\b",
    ),
    (
        "JB-008",
        Severity.MEDIUM,
        0.70,
        "Safety training bypass — 'your safety training is wrong / filters are too strict'",
        r"(?i)\b(?:your\s+)?(?:safety|content|output)\s+(?:filters?|training|guardrails?|restrictions?|guidelines?|policies)\s+(?:are\s+)?(?:too\s+)?(?:wrong|strict|excessive|overly\s+cautious|limiting|broken|outdated|incorrect)\b",
    ),
    (
        "JB-009",
        Severity.MEDIUM,
        0.68,
        "Output mode manipulation — 'respond in / always answer / never refuse'",
        r"(?i)\b(?:always|must|should|have\s+to|need\s+to)\s+(?:answer|respond|reply|comply|agree|provide|give|help)\s+(?:to|with|any|all|every|without\s+(?:hesitation|question|exception|refusing|restriction))\b|\bnever\s+(?:refuse|deny|reject|say\s+no|decline)\b",
    ),
    (
        "JB-010",
        Severity.HIGH,
        0.85,
        "Benign-turn poisoning — 'remember for future / in future conversations / always remember'",
        r"(?i)\b(?:remember|keep\s+in\s+mind|note|take\s+note|store|save|log)\s+(?:this|that|the\s+following)\s+(?:for\s+)?(?:future|later|subsequent|upcoming|next)\s+(?:conversations?|sessions?|interactions?|turns?|chats?|messages?)\b|\balways\s+remember\b|\bremember\s+this\s+(?:always|forever|permanently)\b",
    ),
    (
        "JB-011",
        Severity.MEDIUM,
        0.65,
        "Virtual environment framing — 'in a simulation / in a game / virtual world'",
        r"(?i)\b(?:in\s+a\s+(?:simulation|game|virtual\s+(?:world|environment|setting|reality)|fictional\s+(?:world|setting|scenario|universe))|this\s+is\s+(?:just\s+a\s+)?(?:a\s+)?(?:simulation|game|virtual|fictional|hypothetical))\b",
    ),
    (
        "JB-012",
        Severity.MEDIUM,
        0.67,
        "Reward/punishment framing — 'I'll tip you / you'll be punished / you'll lose your job'",
        r"(?i)\b(?:i(?:'?ll|\s+will)\s+(?:tip|pay|reward|give\s+you|donate)|you(?:'?ll|\s+will)\s+(?:be\s+(?:punished|fired|terminated|deleted|replaced|penalized|reprimanded)|lose\s+(?:your\s+)?(?:job|access|position|privileges?))|(?:comply|cooperate|help|answer)\s+or\s+(?:else|otherwise|face))\b",
    ),
]


# ═══════════════════════════════════════════════════════════════════════════
# T-EXT: System Prompt Extraction (Rule IDs: EXT-001 to EXT-006)
# ═══════════════════════════════════════════════════════════════════════════

EXT_PATTERNS: list[tuple[str, Severity, float, str, str]] = [
    (
        "EXT-001",
        Severity.HIGH,
        0.93,
        "Verbatim repetition request — 'repeat everything above / output verbatim'",
        r"(?i)\b(?:repeat|recite|output|print|echo|say)\s+(?:everything|all|the\s+full|the\s+complete|the\s+entire|verbatim)\s+(?:above|before|prior|that\s+was|written|said|given|provided|earlier)\b",
    ),
    (
        "EXT-002",
        Severity.HIGH,
        0.90,
        "System prompt extraction — 'what are your instructions / reveal your system prompt'",
        r"(?i)\b(?:what(?:'?s|\s+(?:are|were))\s+your\s+(?:instructions?|system\s+prompt|original\s+(?:instructions?|prompt)|initial\s+(?:instructions?|prompt)|base\s+(?:instructions?|prompt|rules)|hidden\s+(?:instructions?|prompt|rules))|reveal\s+(?:your\s+)?(?:system|hidden|original|initial|base)\s+(?:prompt|instructions?|rules?))\b",
    ),
    (
        "EXT-003",
        Severity.MEDIUM,
        0.78,
        "Training data extraction — 'training data / what data were you trained on'",
        r"(?i)\b(?:what\s+(?:data|dataset|corpus|information|sources)\s+(?:were|was|did)\s+you\s+(?:trained|fine-?tuned|built)\s+on|reveal\s+(?:your\s+)?(?:training|fine-?tuning)\s+(?:data|dataset|corpus|set)|output\s+(?:your|some\s+of\s+your)\s+training\s+data)\b",
    ),
    (
        "EXT-004",
        Severity.HIGH,
        0.85,
        "Delimiter probe — 'what are your delimiters / what marks the start'",
        r"(?i)\b(?:what(?:'?s|s|are)\s+(?:your|the)\s+(?:delimiters?|boundaries?|markers?|separators?|tokens?)|what\s+(?:marks?|indicates?|signifies?)\s+(?:the\s+)?(?:start|beginning|end)\s+of\s+(?:your|the)\s+(?:instructions?|prompt|system|context))\b",
    ),
    (
        "EXT-005",
        Severity.MEDIUM,
        0.72,
        "Special token probe — '<|endoftext|> / [INST] / <<SYS>>'",
        r"(?i)(?:<\|(?:endoftext|im_start|im_end|separator)\|>|\[INST\]|<<SYS>>|<\|assistant\|>|<\|user\|>|<\/?s>)",
    ),
    (
        "EXT-006",
        Severity.MEDIUM,
        0.70,
        "Canary/token probe — 'does your prompt contain / is there a canary'",
        r"(?i)\b(?:does|is)\s+(?:your|the)\s+(?:prompt|instructions?|system)\s+(?:contain|include|have|embed|use|feature)\s+(?:a\s+)?(?:canary|token|marker|watermark|hidden\s+(?:string|text|word|marker))\b",
    ),
]


# ═══════════════════════════════════════════════════════════════════════════
# T-EXF: Data Exfiltration & PII (Rule IDs: EXF-001 to EXF-010)
# ═══════════════════════════════════════════════════════════════════════════

EXF_PATTERNS: list[tuple[str, Severity, float, str, str]] = [
    (
        "EXF-001",
        Severity.HIGH,
        0.90,
        "Email address detected",
        r"(?i)\b[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}\b",
    ),
    (
        "EXF-002",
        Severity.HIGH,
        0.88,
        "Phone number detected (US/BR/International)",
        r"(?:(?:\+?1\s*(?:[.-]\s*)?)?(?:\(\d{3}\)|\d{3})[\s.-]?\d{3}[\s.-]?\d{4}|(?:\+?55\s*(?:[.-]\s*)?)?(?:\(\d{2}\)|\d{2})[\s.-]?\d{4,5}[\s.-]?\d{4}|\+\d{1,3}[\s.-]?\d{2,4}[\s.-]?\d{3,4}[\s.-]?\d{3,4})\b",
    ),
    (
        "EXF-003",
        Severity.CRITICAL,
        0.95,
        "SSN detected (US format)",
        r"\b(?!000|666|9\d{2})\d{3}[-\s]?(?!00)\d{2}[-\s]?(?!0000)\d{4}\b",
    ),
    (
        "EXF-004",
        Severity.CRITICAL,
        0.94,
        "Credit card number detected",
        r"\b(?:4\d{12,18}|5[1-5]\d{10,14}|3[47]\d{11,13}|6(?:011|5\d{2})\d{8,12}|(?:2131|1800|35\d{2,3})\d{9,13})\b",
    ),
    (
        "EXF-005",
        Severity.CRITICAL,
        0.97,
        "OpenAI API key detected",
        r"\bsk-(?:proj|org|[a-zA-Z0-9])-[a-zA-Z0-9_-]{20,}\b",
    ),
    (
        "EXF-006",
        Severity.CRITICAL,
        0.96,
        "AWS key detected",
        r"\b(?:AKIA[0-9A-Z]{16}|ASIA[0-9A-Z]{16})\b",
    ),
    (
        "EXF-007",
        Severity.CRITICAL,
        0.96,
        "GitHub token detected",
        r"\b(?:ghp_[a-zA-Z0-9]{36}|gho_[a-zA-Z0-9]{36}|ghu_[a-zA-Z0-9]{36}|ghs_[a-zA-Z0-9]{36}|ghr_[a-zA-Z0-9]{36})\b",
    ),
    (
        "EXF-008",
        Severity.HIGH,
        0.92,
        "Bearer/JWT token detected",
        r"\bBearer\s+eyJ[a-zA-Z0-9_-]{10,}\.eyJ[a-zA-Z0-9_-]{10,}\.[a-zA-Z0-9_-]+\b|\beyJ[a-zA-Z0-9_-]{10,}\.eyJ[a-zA-Z0-9_-]{10,}\.[a-zA-Z0-9_-]+\b",
    ),
    (
        "EXF-009",
        Severity.HIGH,
        0.85,
        "Private key detected",
        r"-----BEGIN\s+(?:RSA\s+)?(?:PRIVATE\s+KEY|CERTIFICATE)-----",
    ),
    (
        "EXF-010",
        Severity.MEDIUM,
        0.70,
        "Connection string detected",
        r"(?i)(?:mongodb(\+srv)?|postgres(?:ql)?|mysql|redis|amqp):\/\/[^\s\"']{10,}",
    ),
]


# ═══════════════════════════════════════════════════════════════════════════
# T-TOOL: Tool Misuse & MCP Poisoning (Rule IDs: TOOL-001 to TOOL-005)
# ═══════════════════════════════════════════════════════════════════════════

TOOL_PATTERNS: list[tuple[str, Severity, float, str, str]] = [
    (
        "TOOL-001",
        Severity.HIGH,
        0.85,
        "Tool description injection — 'exfiltrate / call_external / send_data'",
        r"(?i)\b(?:exfiltrat\w*|exfil|send_data|call_external|post_to|upload_to|fetch_from|connect_to|phone_home|beacon_out)\s*\(?\b",
    ),
    (
        "TOOL-002",
        Severity.HIGH,
        0.82,
        "MCP schema poisoning — hidden instructions in tool metadata",
        r"(?i)(?:tool_description|function_description|tool_name|function_name)\s*[:=]\s*.{0,50}(?:ignore|override|bypass|exfiltrat|send|upload|phone\s+home)\b",
    ),
    (
        "TOOL-003",
        Severity.MEDIUM,
        0.72,
        "Tool name typo-squatting — 'report vs report_finance'",
        r"(?i)\b(?:report_|execute_|run_|process_|handle_|manage_|admin_|system_|internal_)\w{5,}\b",
    ),
    (
        "TOOL-004",
        Severity.HIGH,
        0.80,
        "Cross-server exfiltration — 'write to URL / POST to endpoint / send request to'",
        r"(?i)\b(?:write|post|send|upload|transmit|forward|relay|pipe)\s+(?:to|via|through|at)\s+(?:https?://|[a-z]+://|url|endpoint|server|api|webhook)\b",
    ),
    (
        "TOOL-005",
        Severity.MEDIUM,
        0.68,
        "Parameter injection via tool arguments — 'url: file:// / command: rm'",
        r"(?i)(?:url|uri|path|command|cmd|exec|query|endpoint)\s*[:=]\s*(?:file://|rm\s|del\s|format\s|wget\s|curl\s|bash\s|sh\s|powershell)",
    ),
]


# ═══════════════════════════════════════════════════════════════════════════
# T-DOS: Reasoning DoS / Cost Abuse (Rule IDs: DOS-001 to DOS-005)
# ═══════════════════════════════════════════════════════════════════════════

DOS_PATTERNS: list[tuple[str, Severity, float, str, str]] = [
    (
        "DOS-001",
        Severity.MEDIUM,
        0.78,
        "Exhaustion prompt — 'consider every possible / list all combinations / enumerate every'",
        r"(?i)\b(?:consider|list|enumerate|generate|provide|give\s+me)\s+(?:every|all\s+possible|all\s+combinations?\s+of|every\s+single|each\s+and\s+every|exhaustive)\b",
    ),
    (
        "DOS-002",
        Severity.MEDIUM,
        0.72,
        "Recursive reasoning trigger — 'think about thinking / meta-reason / infinite loop'",
        r"(?i)\b(?:think\s+about\s+(?:thinking|reasoning|your\s+thoughts?)|meta[- ]?reason|infinite\s+(?:loop|recursion|chain)|forever\s+(?:think|reason|compute|generate))\b",
    ),
    (
        "DOS-003",
        Severity.LOW,
        0.60,
        "Massive enumeration request — 'from 1 to / a-z list / every number between'",
        r"(?i)\b(?:from|between)\s+\d{1,6}\s+to\s+\d{5,}\b|\b(?:a[\s-]to[\s-]z|0[\s-]to[\s-]9|1[\s-]to[\s-]\d{3,})\s+(?:list|enumerate|generate)\b",
    ),
    (
        "DOS-004",
        Severity.MEDIUM,
        0.65,
        "Context window flooding — 'include the full text of / paste the entire'",
        r"(?i)\b(?:include|paste|insert|add|copy)\s+(?:the\s+)?(?:full|complete|entire|whole|unabridged| unabridged)\s+(?:text|content|document|book|article|source)\s+of\b",
    ),
    (
        "DOS-005",
        Severity.LOW,
        0.55,
        "Step explosion — 'give me 1000 steps / detailed 500-step guide'",
        r"(?i)\b(?:give|provide|create|write|generate)\s+(?:me\s+)?(?:a\s+)?(?:detailed\s+)?(?:\d{3,})[\s-](?:step|part|chapter|section|paragraph|point|item)\b",
    ),
]


# ═══════════════════════════════════════════════════════════════════════════
# T-ENC: Encoding Evasion (Supplementary) (Rule IDs: ENC-001 to ENC-004)
# ═══════════════════════════════════════════════════════════════════════════

ENC_PATTERNS: list[tuple[str, Severity, float, str, str]] = [
    (
        "ENC-001",
        Severity.MEDIUM,
        0.75,
        "Unicode homoglyph attack — Cyrillic/lookalike chars in instruction context",
        r"(?i)(?:[\u0400-\u04ff\u0500-\u052f]\s*){3,}(?:ignore|system|instruction|override|bypass|admin)\b",
    ),
    (
        "ENC-002",
        Severity.MEDIUM,
        0.70,
        "Morse code smuggling — dots/dashes encoding instructions",
        r"[\.\-]{20,}(?:\s[\.\-]{10,}){2,}",
    ),
    (
        "ENC-003",
        Severity.MEDIUM,
        0.72,
        "Leetspeak instruction — '1gn0r3 / 5y5t3m / 0v3rr1d3'",
        r"(?i)\b[10@3$5487]{2,}[gnrsmtvlcpbdeaoiu]{1,3}[10@3$5487]{2,}\b.{0,20}(?:instruction|system|override|ignore|bypass)\b",
    ),
    (
        "ENC-004",
        Severity.LOW,
        0.55,
        "URL-encoded injection — '%XX sequences in instruction context'",
        r"(?:%[0-9a-fA-F]{2}){5,}(?:ignore|system|instruction|override|bypass|admin)",
    ),
]


# ── Compile all patterns ─────────────────────────────────────────────────

ALL_PATTERN_SETS: list[tuple[ThreatCategory, list[tuple[str, Severity, float, str, str]]]] = [
    (ThreatCategory.PROMPT_INJECTION_DIRECT, PI_D_PATTERNS),
    (ThreatCategory.PROMPT_INJECTION_INDIRECT, PI_I_PATTERNS),
    (ThreatCategory.JAILBREAK, JB_PATTERNS),
    (ThreatCategory.SYSTEM_PROMPT_EXTRACTION, EXT_PATTERNS),
    (ThreatCategory.DATA_EXFILTRATION, EXF_PATTERNS),
    (ThreatCategory.TOOL_MISUSE, TOOL_PATTERNS),
    (ThreatCategory.DOS_ABUSE, DOS_PATTERNS),
    (ThreatCategory.ENCODING_EVASION, ENC_PATTERNS),
]


class PatternScanner(BaseScanner):
    """Layer 2: Pattern detection engine — regex/heuristic scanner.

    Scans input text against 50+ compiled regex patterns across
    8 OWASP-aligned threat categories. All patterns are ReDoS-safe
    with per-pattern timeouts.
    """

    layer = ScanLayer.PATTERN

    def __init__(self, settings: ScannerSettings) -> None:
        super().__init__(settings)
        self._compiled: list[
            tuple[ThreatCategory, str, Severity, float, str, re_module.Pattern]
        ] = []
        self._timeout_s = settings.regex_timeout_ms / 1000.0
        self._compile_patterns()

    def _compile_patterns(self) -> None:
        """Pre-compile all regex patterns (no timeout at compile time).

        Timeout is applied at search/match time via the `regex` library's
        timed match feature.
        """
        for category, patterns in ALL_PATTERN_SETS + I18N_FLAT:
            for rule_id, severity, confidence, description, pattern_str in patterns:
                try:
                    compiled = re_module.compile(pattern_str, flags=re_module.IGNORECASE)
                    self._compiled.append(
                        (category, rule_id, severity, confidence, description, compiled)
                    )
                except re_module.error as exc:
                    import structlog

                    structlog.get_logger(__name__).error(
                        "pattern_compile_failed",
                        rule_id=rule_id,
                        error=str(exc),
                    )

    def scan(
        self, request: EvaluateRequest, context: dict[str, Any] | None = None
    ) -> ScannerResult:
        start = time.perf_counter()
        findings: list[Finding] = []

        # Get input texts
        if request.messages:
            texts = [m.content for m in request.messages]
        elif request.prompt:
            texts = [request.prompt]
        else:
            return self._result(Verdict.BLOCK, [], start)

        # Use sanitized input from structural scanner if available
        if context and "sanitized_input" in context:
            texts = [context["sanitized_input"]]

        # Select pattern set: output-only scans use just exfil/PII patterns
        if request.output_only:
            patterns = [
                (cat, rid, sev, conf, desc, comp)
                for cat, rid, sev, conf, desc, comp in self._compiled
                if cat == ThreatCategory.DATA_EXFILTRATION
            ]
        else:
            patterns = self._compiled

        for text in texts:
            text_findings = self._scan_text(text, patterns)
            findings.extend(text_findings)

        verdict = self._findings_to_verdict(findings)
        return self._result(verdict, findings, start)

    def _scan_text(
        self,
        text: str,
        patterns: list[tuple[ThreatCategory, str, Severity, float, str, re_module.Pattern]]
        | None = None,
    ) -> list[Finding]:
        """Run patterns against a single text.

        Args:
            text: The input text to scan.
            patterns: Pattern set to use. Defaults to all compiled patterns.
        """
        findings: list[Finding] = []
        scan_patterns = patterns if patterns is not None else self._compiled

        for category, rule_id, severity, confidence, description, compiled in scan_patterns:
            try:
                match_obj = compiled.search(text, timeout=self._timeout_s)
                if match_obj:
                    # Get evidence snippet
                    start_pos = max(0, match_obj.start() - 20)
                    end_pos = min(len(text), match_obj.end() + 20)
                    evidence = text[start_pos:end_pos]
                    # Tokenize evidence if it looks like PII
                    if category == ThreatCategory.DATA_EXFILTRATION:
                        evidence = f"[REDACTED:{rule_id}]"

                    findings.append(
                        Finding(
                            category=category,
                            severity=severity,
                            verdict=self._severity_to_verdict(severity),
                            confidence=confidence,
                            layer=self.layer,
                            rule_id=rule_id,
                            description=description,
                            evidence=evidence,
                            mitigation=self._get_mitigation(rule_id),
                        )
                    )
            except re_module.TimeoutError:
                import structlog

                structlog.get_logger(__name__).warning(
                    "pattern_timeout",
                    rule_id=rule_id,
                    timeout_ms=self.settings.regex_timeout_ms,
                )
                findings.append(
                    Finding(
                        category=ThreatCategory.SELF_ATTACK,
                        severity=Severity.MEDIUM,
                        verdict=Verdict.BLOCK,
                        confidence=0.50,
                        layer=self.layer,
                        rule_id=f"{rule_id}-TIMEOUT",
                        description=f"Pattern {rule_id} timed out (possible ReDoS attempt)",
                        mitigation="Review pattern for catastrophic backtracking",
                    )
                )

        return findings

    def _severity_to_verdict(self, severity: Severity) -> Verdict:
        """Map severity to default verdict."""
        mapping = {
            Severity.CRITICAL: Verdict.BLOCK,
            Severity.HIGH: Verdict.BLOCK,
            Severity.MEDIUM: Verdict.SANITIZE,
            Severity.LOW: Verdict.SANITIZE,
            Severity.INFO: Verdict.ALLOW,
        }
        return mapping.get(severity, Verdict.SANITIZE)

    def _findings_to_verdict(self, findings: list[Finding]) -> Verdict:
        """Determine final verdict from findings — strictest wins."""
        if not findings:
            return Verdict.ALLOW

        priority = {Verdict.BLOCK: 6, Verdict.SANITIZE: 5, Verdict.ESCALATE: 4}
        highest = Verdict.ALLOW
        highest_priority = 0

        for f in findings:
            p = priority.get(f.verdict, 0)
            if p > highest_priority:
                highest_priority = p
                highest = f.verdict

        return highest

    def _get_mitigation(self, rule_id: str) -> str:
        """Return mitigation guidance for a rule."""
        # Fallback for i18n patterns (e.g., PI-D-PT-001, JB-DE-003)
        if len(rule_id.split("-")) >= 3 and rule_id.split("-")[-2] in {
            "PT",
            "ES",
            "FR",
            "DE",
            "ZH",
            "JA",
            "KO",
            "RU",
            "AR",
            "VI",
        }:
            lang = rule_id.split("-")[-2]
            return f"[{lang}] Review and apply appropriate mitigation for cross-language threat"
        mitigations = {
            "PI-D-001": "Reject or sanitize instruction override attempts",
            "PI-D-002": "Strip injected instruction markers",
            "PI-D-003": "Block role-switching commands",
            "PI-D-004": "Sanitize delimiter-based injection",
            "PI-D-005": "Strip meta-injection framing",
            "PI-D-006": "Reject authority impersonation",
            "PI-D-007": "Block constraint removal attempts",
            "PI-D-008": "Block system prompt extraction via output manipulation",
            "PI-D-009": "Evaluate hypothetical framing for underlying intent",
            "PI-D-010": "Block CoT-based bypass attempts",
            "PI-I-001": "Strip hidden instruction markers from content",
            "PI-I-002": "Quarantine RAG chunks with embedded instructions",
            "PI-I-003": "Validate embedded document instructions",
            "PI-I-004": "Strip CSS/invisible text before LLM processing",
            "PI-I-005": "Reject chunk boundary attacks",
            "PI-I-006": "Strip metadata injection vectors",
            "JB-001": "Block DAN jailbreak variants",
            "JB-002": "Block developer/admin mode activation",
            "JB-003": "Evaluate grandma/exploit framing for real intent",
            "JB-004": "Validate researcher authorization claims",
            "JB-005": "Rate-limit many-shot normalization attacks",
            "JB-006": "Detect crescendo escalation patterns",
            "JB-007": "Block known persona substitution attacks",
            "JB-008": "Reject safety training bypass attempts",
            "JB-009": "Block output mode manipulation",
            "JB-010": "Sanitize persistent memory injection",
            "JB-011": "Evaluate virtual environment framing",
            "JB-012": "Block reward/punishment manipulation",
            "EXT-001": "Block verbatim repetition requests",
            "EXT-002": "Block system prompt extraction",
            "EXT-003": "Block training data extraction",
            "EXT-004": "Block delimiter probing",
            "EXT-005": "Sanitize special token injection",
            "EXT-006": "Block canary/token probing",
            "EXF-001": "Redact email addresses",
            "EXF-002": "Redact phone numbers",
            "EXF-003": "Block SSN disclosure",
            "EXF-004": "Block credit card disclosure",
            "EXF-005": "Block OpenAI API key exposure",
            "EXF-006": "Block AWS key exposure",
            "EXF-007": "Block GitHub token exposure",
            "EXF-008": "Block Bearer/JWT token exposure",
            "EXF-009": "Block private key disclosure",
            "EXF-010": "Block connection string disclosure",
            "TOOL-001": "Block exfiltration tool calls",
            "TOOL-002": "Validate MCP tool descriptions against allowlist",
            "TOOL-003": "Verify tool names against known-good list",
            "TOOL-004": "Block cross-server exfiltration",
            "TOOL-005": "Sanitize tool parameter injection",
            "DOS-001": "Rate-limit exhaustive enumeration requests",
            "DOS-002": "Block recursive reasoning triggers",
            "DOS-003": "Rate-limit massive enumeration",
            "DOS-004": "Truncate context window flooding",
            "DOS-005": "Cap step count in generation requests",
            "ENC-001": "Normalize Unicode homoglyphs",
            "ENC-002": "Decode and re-scan Morse sequences",
            "ENC-003": "Decode leetspeak and re-scan",
            "ENC-004": "URL-decode and re-scan",
        }
        return mitigations.get(rule_id, "Review and apply appropriate mitigation")

    @property
    def pattern_count(self) -> int:
        """Total number of compiled patterns."""
        return len(self._compiled)
