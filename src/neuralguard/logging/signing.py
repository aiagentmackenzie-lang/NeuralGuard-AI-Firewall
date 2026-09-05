"""Ed25519 signing for audit events (P2-10).

The per-worker hash chain (``logging.chain``) detects modification of
written events. Its documented limit: an attacker who can delete a whole
chain file and write a NEW internally-consistent chain is undetected — the
chain verifies because nothing binds it to a holder of secret key material.

Signing closes exactly that gap: the worker signs each event's
``event_hash`` with an Ed25519 private key (hex-encoded 32-byte seed in
``NEURALGUARD_AUDIT_SIGNING_KEY``). The signature rides on the event
(``event_sig``); verification needs only the public key (``neuralguard
audit-verify --pubkey <hex>``). Forging a chain now requires the private
key, not just file access.

Honest scope (unchanged from the ledger):
- Cross-worker ORDERING is still not established — signing authenticates
  each per-worker chain; a WORM sink or DB sequence (future) is what
  proves global ordering.
- Key rotation: signatures verify against the public key of the key that
  made them. Rotate = new key epoch; verify per-epoch (documented in the
  secret-rotation runbook).
- The signing module imports `cryptography` lazily; the dependency is a
  main requirement, but chain.py stays pure-stdlib.
"""

from __future__ import annotations

import secrets
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

_SIGNING_ALGORITHM = "ed25519"


class SigningKeyError(ValueError):
    """The configured signing key material is invalid."""


def generate_signing_keypair() -> tuple[str, str]:
    """Generate (seed_hex, pubkey_hex) for a new Ed25519 signing key."""
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    private = Ed25519PrivateKey.generate()
    seed = private.private_bytes_raw()
    pubkey = private.public_key().public_bytes_raw()
    return seed.hex(), pubkey.hex()


def generate_seed() -> str:
    """Random raw seed hex (equivalent to `secrets.token_hex(32)`)."""
    return secrets.token_hex(32)


def _private_key_from_seed_hex(seed_hex: str) -> Ed25519PrivateKey:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey,
    )

    try:
        seed = bytes.fromhex(seed_hex)
    except ValueError as exc:
        raise SigningKeyError("signing key is not valid hex") from exc
    if len(seed) != 32:
        raise SigningKeyError(f"signing key must decode to exactly 32 bytes, got {len(seed)}")
    return Ed25519PrivateKey.from_private_bytes(seed)


def sign_event_hash(event_hash: str, seed_hex: str) -> str:
    """Sign one event's chain hash; returns the hex signature.

    Raises SigningKeyError on bad key material (a configuration error —
    fail loudly at boot, not silently per event).
    """
    private = _private_key_from_seed_hex(seed_hex)
    signature: bytes = private.sign(bytes.fromhex(event_hash))
    return signature.hex()


def verify_event_signature(event_hash: str, signature_hex: str, pubkey_hex: str) -> bool:
    """Verify one event signature; False on any mismatch or bad input."""
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PublicKey,
    )

    try:
        event_hash_bytes = bytes.fromhex(event_hash)
        signature = bytes.fromhex(signature_hex)
        pubkey_bytes = bytes.fromhex(pubkey_hex)
    except ValueError:
        return False
    if len(pubkey_bytes) != 32:
        return False
    try:
        Ed25519PublicKey.from_public_bytes(pubkey_bytes).verify(signature, event_hash_bytes)
        return True
    except (InvalidSignature, ValueError):
        return False


def public_key_from_seed(seed_hex: str) -> str:
    """Derive the hex public key from the hex seed (operator convenience)."""
    private = _private_key_from_seed_hex(seed_hex)
    pubkey: bytes = private.public_key().public_bytes_raw()
    return pubkey.hex()
