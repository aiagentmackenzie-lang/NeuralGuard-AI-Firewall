#!/usr/bin/env bash
# generate_sbom.sh — Generate CycloneDX SBOM locally
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

echo "🔧 Generating CycloneDX SBOM..."

# Install cyclonedx-bom if not present
if ! uv run python -c "import cyclonedx" 2>/dev/null; then
    echo "📦 Installing cyclonedx-bom..."
    uv run pip install cyclonedx-bom
fi

# Generate JSON + XML
uv run cyclonedx-py environment --output-format json --output-file sbom.json
uv run cyclonedx-py environment --output-format xml --output-file sbom.xml

# Summary
COMPONENTS=$(uv run python -c "import json; d=json.load(open('sbom.json')); print(len(d.get('components', [])))")
echo "✅ SBOM generated: sbom.json + sbom.xml"
echo "   Components: $COMPONENTS"