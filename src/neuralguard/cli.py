"""NeuralGuard CLI — command-line interface for server management."""

from __future__ import annotations

import argparse
import sys

from neuralguard.main import main as serve_main


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

    args = parser.parse_args()

    if args.command == "version":
        from neuralguard import __version__

        print(f"NeuralGuard v{__version__}")
        sys.exit(0)

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
