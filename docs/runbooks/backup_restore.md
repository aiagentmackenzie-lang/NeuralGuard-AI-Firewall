# Audit Backup & Restore Runbook (P1-6)

The audit log is the forensic record of every firewall decision. It is
append-only by convention and tamper-evident by hash chain (P1-4). Back it
up so an operator can reconstruct any decision after a host loss, and so
tampering is detectable against an off-host copy.

## Backends

- **JSONL** (default): `NEURALGUARD_AUDIT_JSONL_PATH` (default
  `./audit_logs`, `/data/audit` in the container). One file per UTC day,
  rotated at 100 MiB. Files: `audit-YYYY-MM-DD.jsonl`, `audit-YYYY-MM-DD-N.jsonl`.
- **Postgres**: table `audit_events` (when `audit.backend=postgres`).

## JSONL backup

### Daily snapshot to off-host storage

```bash
# From the host (or a sidecar with the audit volume mounted):
AUDIT_DIR="${NEURALGUARD_AUDIT_JSONL_PATH:-/data/audit}"
DEST="s3://your-bucket/neuralguard-audit/$(date -u +%Y/%m/)"
aws s3 sync --no-progress "$AUDIT_DIR" "$DEST" \
  --exclude "*" --include "audit-*.jsonl" \
  --exclude "$(date -u +%Y-%m-%d).jsonl"   # skip today's still-open file
```

Run nightly via cron / systemd timer / k8s CronJob. Skipping today's open
file avoids backing up a partial line mid-append; it is captured the next
night when closed.

### Integrity manifest (recommended)

Hash-chained events are self-verifying, but an off-host manifest makes
tampering of the *file set* (deletion/insertion) detectable too:

```bash
cd "$AUDIT_DIR"
sha256sum audit-*.jsonl > manifest.$(date -u +%Y%m%d).txt
aws s3 cp manifest.$(date -u +%Y%m%d).txt "$DEST"
```

Compare successive manifests nightly; any line that disappears or changes
its hash is an alert.

## Postgres backup

### `pg_dump` (logical)

```bash
PGPASSWORD="$POSTGRES_PASSWORD" pg_dump -h host -U neuralguard -d neuralguard \
  -t audit_events --data-only --column-inserts \
  | gzip > audit_events_$(date -u +%Y%m%d).sql.gz
```

Upload to off-host storage. Restore with `gunzip -c file.sql.gz | psql ...`.

### Continuous (recommended for production)

Use managed PITR (RDS/CloudSQL automated backups + point-in-time recovery)
or WAL-G / pgBackRest for self-hosted. The audit table is append-only, so
a base backup + WAL replay reconstructs the chain exactly.

## Restore

### JSONL restore (read-only)

Restored JSONL files are for **analysis**, not re-ingestion — the logger
never reads back. Copy them to an analyst workstation and verify:

```python
from pathlib import Path
from neuralguard.models.schemas import AuditEvent
from neuralguard.logging.chain import verify_chain

# Group by worker_id (each process is its own chain) and verify each.
events = [AuditEvent.model_validate_json(l) for l in Path("restored").glob("audit-*.jsonl")]
chains: dict[str, list[AuditEvent]] = {}
for e in events:
    chains.setdefault(e.worker_id, []).append(e)
for wid, chain in chains.items():
    chain.sort(key=lambda e: e.timestamp)
    ok = verify_chain(chain)
    print(f"chain {wid[:8]}: {len(chain)} events, verify={ok}")
```

A `verify=False` means a restored event was modified after writing, or the
chain is incomplete (a deleted event). Both are forensically meaningful.

### Postgres restore

```bash
# Recreate schema, then load:
PGPASSWORD="$POSTGRES_PASSWORD" psql -h host -U neuralguard -d neuralguard \
  -c "TRUNCATE audit_events;"   # only on a fresh recovery target
gunzip -c audit_events_YYYYMMDD.sql.gz \
  | PGPASSWORD="$POSTGRES_PASSWORD" psql -h host -U neuralguard -d neuralguard
```

Do NOT restore into a live production DB — restore into a separate
forensic target and verify there.

## Retention

`NEURALGUARD_AUDIT_RETENTION_DAYS` (default 30) controls on-disk JSONL
cleanup. **Backups are independent of retention** — set retention to keep
the hot window small (fast scans) and keep backups for as long as your
compliance regime requires (SOC2: 7 years; EU AI Act logging: durable).
Postgres retention is the operator's responsibility (partitioning +
archival, not enforced by NeuralGuard).

## Verification (after any restore)

```bash
# 1. The restored chain verifies (see snippet above).
# 2. Event count matches the backup manifest.
# 3. No event_hash collisions across the restored set:
sort audit-*.jsonl | jq -r .event_hash | sort | uniq -d   # must be empty
```

## What is NOT covered here

- Cross-worker chain ordering (P2): chains are per-worker; restoring the
  global timeline requires joining on `timestamp` across chains, which is
  approximate, not strict.
- WORM sink (P2): a true write-once target (object lock, S3 Object Lock in
  compliance mode, or a dedicated WORM appliance) is the next step beyond
  these backups for evidentiary-grade retention.