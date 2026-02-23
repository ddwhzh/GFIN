#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "Building GFIN Docker image..."
docker build -t gfin:latest "$PROJECT_DIR"
echo "Done. Image: gfin:latest"
