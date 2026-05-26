#!/usr/bin/env bash
# scripts/verify-gate.sh — Bash wrapper to run the fish verification gate.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec fish "$SCRIPT_DIR/verify-gate.fish"
