"""NeuralGuard CLI — command-line interface for server + template analysis."""

from __future__ import annotations

import argparse
import json
import sys

from neuralguard.main import main as serve_main


def _cmd_analyze_template(args: argparse.Namespace) -> int:
    """`neuralguard analyze-template <file|->` — static injection-sink analysis."""
    from neuralguard.analysis import TemplateAnalyzer

    if args.template == "-":
        template = sys.stdin.read()
    else:
        from pathlib import Path

        try:
            template = Path(args.template).read_text(encoding="utf-8")
        except OSError as exc:
            print(f"error: cannot read {args.template!r}: {exc}", file=sys.stderr)
            return 2

    if not template.strip():
        print("error: template is empty", file=sys.stderr)
        return 2

    analyzer = TemplateAnalyzer()
    result = analyzer.analyze(template)

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
        return 0 if result.is_clean else 1

    # Human-readable text output.
    if result.is_clean:
        print(f"clean: no injection sinks found ({len(template)} chars).")
        return 0

    print(f"found {len(result.sinks)} injection sink(s):\n")
    severity_order = {"high": 0, "medium": 1, "low": 2, "info": 3}
    for sink in sorted(result.sinks, key=lambda s: (severity_order.get(s.severity, 9), s.location)):
        print(f"  [{sink.severity.upper():<6}] {sink.rule_id}  (line {sink.location})")
        print(f"          {sink.description}")
        print(f"          evidence: {sink.evidence}")
        print(f"          fix: {sink.remediation}\n")
    # Exit non-zero so CI gates can fail on sinks (e.g. high-severity only).
    has_high = any(s.severity == "high" for s in result.sinks)
    return 1 if (args.fail_on_high and has_high) else 0


def _cmd_canary_mint(args: argparse.Namespace) -> int:
    """`neuralguard canary-mint <session_id>` — mint per-session canary token(s)."""
    from neuralguard.config.settings import load_config

    config = load_config()
    if not config.canary.enabled:
        print(
            "error: canary feature is disabled. Set NEURALGUARD_CANARY_ENABLED=true "
            "and NEURALGUARD_CANARY_SECRET.",
            file=sys.stderr,
        )
        return 2
    try:
        from neuralguard.canary import CanaryManager

        manager = CanaryManager(config.canary)
        tokens = manager.mint(args.session_id, args.count)
    except Exception as exc:  # misconfigured secret / validation
        print(f"error: cannot mint canary: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps({"session_id": args.session_id, "tokens": tokens}))
    else:
        for tok in tokens:
            print(tok)
    return 0


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        prog="neuralguard",
        description="NeuralGuard — LLM Guard / AI Application Firewall",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # serve
    serve_parser = subparsers.add_parser("serve", help="Start the NeuralGuard server")
    serve_parser.add_argument("--host", default=None, help="Bind address")
    serve_parser.add_argument("--port", type=int, default=None, help="Bind port")
    serve_parser.add_argument("--workers", type=int, default=None, help="Worker count")
    serve_parser.add_argument("--log-level", default=None, help="Log level")

    # version
    subparsers.add_parser("version", help="Print version")

    # analyze-template (B2)
    at = subparsers.add_parser(
        "analyze-template",
        help="Statically analyze a system-prompt template for injection sinks.",
    )
    at.add_argument(
        "template",
        help="Path to the template file, or '-' to read from stdin.",
    )
    at.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of human-readable text.",
    )
    at.add_argument(
        "--fail-on-high",
        action="store_true",
        help="Exit non-zero only if a HIGH-severity sink is found (for CI gates).",
    )
    at.set_defaults(func=_cmd_analyze_template)

    # canary-mint (B3)
    cm = subparsers.add_parser(
        "canary-mint",
        help="Mint per-session canary token(s) for system-prompt exfiltration detection.",
    )
    cm.add_argument("session_id", help="Session ID to bind the canary token(s) to.")
    cm.add_argument(
        "--count",
        type=int,
        default=None,
        help="Number of distinct canaries to mint (1-8). Defaults to NEURALGUARD_CANARY_TOKEN_COUNT.",
    )
    cm.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    cm.set_defaults(func=_cmd_canary_mint)

    args = parser.parse_args()

    if args.command == "version":
        from neuralguard import __version__

        print(f"NeuralGuard v{__version__}")
        sys.exit(0)

    if args.command == "analyze-template":
        sys.exit(args.func(args))

    if args.command == "canary-mint":
        sys.exit(args.func(args))

    if args.command == "serve" or args.command is None:
        # Override config with CLI args
        if args.host:
            import os

            os.environ["NEURALGUARD_HOST"] = args.host
        if args.port:
            import os

            os.environ["NEURALGUARD_PORT"] = str(args.port)
        if args.workers:
            import os

            os.environ["NEURALGUARD_WORKERS"] = str(args.workers)
        if args.log_level:
            import os

            os.environ["NEURALGUARD_LOG_LEVEL"] = args.log_level

        serve_main()

    parser.print_help()


if __name__ == "__main__":
    main()
