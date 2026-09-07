"""NG-7/NG-8: MCP gateway — signed tool-inventory baselining + Intent Gate.

See ``manifest.py`` (catalog hashing + signed baselines), ``baseliner.py``
(stateful rug-pull detection), ``intent_gate.py`` (header-based per-tool
policy, pre-body-parse), ``transport.py`` (JSON-RPC passthrough), and the
``/v1/mcp`` routes.
"""
