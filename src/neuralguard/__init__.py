"""NeuralGuard — LLM Guard / AI Application Firewall."""

try:
    from importlib.metadata import version as _version

    __version__ = _version("neuralguard")
except Exception:
    __version__ = "0.1.0"  # fallback for editable installs


def create_app(config=None):
    """Lazy import to avoid circular dependency."""
    from neuralguard.main import create_app as _create

    return _create(config)


def main():
    """Lazy import to avoid circular dependency."""
    from neuralguard.main import main as _main

    _main()
