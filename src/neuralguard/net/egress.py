"""Egress classification for outbound endpoints (judge F10.3, proxy F9).

Shared so the judge gate, the proxy, and /v1/info all agree on what counts
as leaving the trust boundary: loopback literals, RFC1918, link-local,
IPv6 ULA, dot-less container-internal names, and the Docker host reference
are LOCAL; everything else (public IPs, public hostnames) is EGRESS —
prompts leave the machine when sent there.
"""

from __future__ import annotations

import ipaddress
from urllib.parse import urlparse


def is_private_endpoint(url: str) -> bool:
    """True if the endpoint is loopback/private; False = egress."""
    host = (urlparse(url).hostname or "").lower()
    if not host:
        return False
    if host == "host.docker.internal":
        return True
    try:
        addr = ipaddress.ip_address(host)
    except ValueError:
        # Not an IP literal: bare names are container-internal; public
        # hostnames have dots and are egress.
        return "." not in host
    return addr.is_loopback or addr.is_private or addr.is_link_local or addr.is_unspecified
