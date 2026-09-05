"""Standalone appliance mode (F9): NeuralGuard as a transparent guardian
in front of an existing LLM endpoint.

``POST /v1/proxy/chat/completions`` accepts an OpenAI-format chat payload,
evaluates the user turns through the full pipeline, forwards ALLOWed requests
to the configured upstream (OpenAI-compatible — Ollama ``/v1``, vLLM, most
clouds), scans the completion with output-scan semantics (PII/exfil/canary),
and returns the verdict-shaped result to the caller. Non-allow inputs never
reach the upstream.
"""
