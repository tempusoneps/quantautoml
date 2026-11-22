#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
README="$ROOT_DIR/README.md"

DOCS=(
  "docs/OVERVIEW.md"
  "docs/QUICKSTART.md"
  "docs/CONFIGURATION.md"
  "docs/OUTPUTS.md"
  "docs/FEATURES.md"
  "docs/ARCHITECTURE.md"
  "docs/STRUCTURE.md"
)

HEADER="<!-- This file is generated from docs/*.md by scripts/generate_readme.sh. Do not edit it directly. -->"

echo "$HEADER" > "$README"

for i in "${!DOCS[@]}"; do
  doc="${DOCS[$i]}"
  if [[ "$i" -gt 0 ]]; then
    echo "" >> "$README"
    echo "---" >> "$README"
  fi
  echo "" >> "$README"
  cat "$ROOT_DIR/$doc" >> "$README"
done

echo "Generated README.md from ${#DOCS[@]} documentation files in docs/."
