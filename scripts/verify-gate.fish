#!/usr/bin/env fish
# scripts/verify-gate.fish — Runs the full verification suite natively under Gemini/Antigravity.

set -gx CLAUDE_PROJECT_DIR (pwd)
set -l PROJECT_NAME (basename $CLAUDE_PROJECT_DIR)

# Prepend the cargo wrapper to PATH so that hooks automatically use --features gpu
set -gx PATH $CLAUDE_PROJECT_DIR/scripts/.cargo-wrapper $PATH

echo "=================================================="
echo "   🚀 RUNNING INTEGRATED VERIFICATION GATE        "
echo "=================================================="
echo ""

# 1. Stubs Hunter
echo "🔍 [1/5] Running Stubs Hunter (stub-check)..."
fish /home/feanor/Projects/.claude/hooks/stub-check.fish
if test $status -ne 0
    echo "❌ Stubs Hunter failed! Resolve todo!/panic!/unimplemented! statements."
    exit 2
end
echo "✓ Stubs Hunter clean."
echo ""

# 2. Rust Code Verification (Format, Compile Warnings, Clippy, Tests)
echo "🦀 [2/5] Running Comprehensive Rust Verification..."
fish $CLAUDE_PROJECT_DIR/.claude/hooks/verify-rust.fish
if test $status -ne 0
    echo "❌ Rust verification failed! Check formatting, warnings, clippy, or test failures."
    exit 2
end
echo ""

# 3. No-Curl Envoy Dogfooding Check
echo "🛡️ [3/5] Running Envoy Bypass Check..."
fish /home/feanor/.claude/hooks/no-curl-envoy.fish
if test $status -ne 0
    echo "❌ Envoy Bypass Check failed! Replace local curl statements with MCP tools."
    exit 2
end
echo "✓ Envoy Bypass Check clean."
echo ""

# 4. Private Git Dependency Check
echo "📦 [4/5] Running Private Dependency Check..."
fish /home/feanor/.claude/hooks/private-dep-check.fish
if test $status -ne 0
    echo "❌ Private Dependency Check failed! Verify Cargo.toml references."
    exit 2
end
echo "✓ Private Dependency Check clean."
echo ""

# 5. Atheneum Persistent Discoveries
echo "🧠 [5/5] Persisting discoveries to Atheneum knowledge graph..."
fish /home/feanor/.claude/hooks/atheneum-persist.fish
# Do not block the gate if Atheneum persists is failing (non-blocking)
echo ""

echo "=================================================="
echo "   ✅ ALL VERIFICATION GATES PASSED SUCCESSFULLY!"
echo "=================================================="
