#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root/client"

npm ci
npm run build

echo "PersonaPlex Web UI built at client/dist"
