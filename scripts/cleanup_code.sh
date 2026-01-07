#!/bin/bash
# ROCmForge Automated Code Cleanup Script
#
# This script automates the cleanup of compiler warnings and code quality issues.
# Run from the project root directory.
#
# Usage:
#   ./scripts/cleanup_code.sh [--dry-run] [--phase <1|2|3|4|all>]
#
# Options:
#   --dry-run     Show what would be changed without making changes
#   --phase N     Run only specific phase (default: all phases)
#
# Author: ROCmForge Development Team
# Version: 1.0
# Date: 2025-01-06

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_LOG="/tmp/rocmforge_cleanup_$(date +%Y%m%d_%H%M%S).log"
DRY_RUN=false
PHASE="all"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --phase)
            PHASE="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [--dry-run] [--phase <1|2|3|4|all>]"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1" | tee -a "$BUILD_LOG"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a "$BUILD_LOG"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "$BUILD_LOG"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$BUILD_LOG"
}

# Header
echo "================================================"
echo "  ROCmForge Automated Code Cleanup"
echo "================================================"
echo "Project: $PROJECT_ROOT"
echo "Log file: $BUILD_LOG"
echo "Dry run: $DRY_RUN"
echo "Phase: $PHASE"
echo "================================================"
echo ""

# Change to project root
cd "$PROJECT_ROOT"

# Phase 1: Auto-fix unused imports and variables
phase1_auto_fix() {
    log_info "Phase 1: Auto-fixing unused imports and variables..."

    if [ "$DRY_RUN" = true ]; then
        log_warning "DRY RUN: Would run 'cargo fix --lib --allow-dirty'"
        log_warning "DRY RUN: Would run 'cargo fix --bin rocmforge_cli --allow-dirty'"
    else
        log_info "Running cargo fix for library..."
        cargo fix --lib --allow-dirty --allow-staged 2>&1 | tee -a "$BUILD_LOG"

        log_info "Running cargo fix for CLI binary..."
        cargo fix --bin rocmforge_cli --allow-dirty --allow-staged 2>&1 | tee -a "$BUILD_LOG"

        log_success "Phase 1 complete"
    fi
}

# Phase 2: Clippy auto-fix
phase2_clippy_fix() {
    log_info "Phase 2: Running clippy auto-fix..."

    if [ "$DRY_RUN" = true ]; then
        log_warning "DRY RUN: Would run 'cargo clippy --fix'"
    else
        log_info "Running cargo clippy with auto-fix..."
        cargo clippy --fix --allow-dirty --allow-staged 2>&1 | tee -a "$BUILD_LOG"

        log_success "Phase 2 complete"
    fi
}

# Phase 3: Format code
phase3_format() {
    log_info "Phase 3: Formatting code with rustfmt..."

    if [ "$DRY_RUN" = true ]; then
        log_warning "DRY RUN: Would run 'cargo fmt'"
        # Check if formatting is needed
        if cargo fmt --check 2>&1 | tee -a "$BUILD_LOG"; then
            log_success "Code is already formatted"
        else
            log_warning "Code needs formatting"
        fi
    else
        log_info "Running cargo fmt..."
        cargo fmt 2>&1 | tee -a "$BUILD_LOG"

        log_success "Phase 3 complete"
    fi
}

# Phase 4: Build and verify
phase4_verify() {
    log_info "Phase 4: Building and verifying..."

    log_info "Running cargo build..."
    cargo build --workspace 2>&1 | tee -a "$BUILD_LOG"

    # Count remaining warnings
    WARNING_COUNT=$(grep -c "warning:" "$BUILD_LOG" || true)
    log_info "Remaining warnings: $WARNING_COUNT"

    log_info "Running cargo test..."
    if [ "$DRY_RUN" = true ]; then
        log_warning "DRY RUN: Would run 'cargo test --workspace'"
    else
        cargo test --workspace 2>&1 | tee -a "$BUILD_LOG"
    fi

    log_success "Phase 4 complete"
}

# Phase 5: Generate report
phase5_report() {
    log_info "Phase 5: Generating cleanup report..."

    echo ""
    echo "================================================"
    echo "  Cleanup Report"
    echo "================================================"
    echo ""

    # Show git diff
    if [ "$DRY_RUN" = false ]; then
        log_info "Git diff summary:"
        git diff --stat 2>&1 | tee -a "$BUILD_LOG"

        echo ""
        log_info "Files changed:"
        git diff --name-only 2>&1 | tee -a "$BUILD_LOG"
    fi

    echo ""
    echo "================================================"
    echo "  Warning Breakdown"
    echo "================================================"
    echo ""

    # Extract warning types
    log_info "Warning types:"
    grep "warning:" "$BUILD_LOG" | \
        sed 's/.*warning: //' | \
        sed 's/ .*//' | \
        sort | uniq -c | sort -rn | \
        tee -a "$BUILD_LOG"

    echo ""
    echo "================================================"
    echo "  Top Files by Warning Count"
    echo "================================================"
    echo ""

    # Extract files with warnings
    grep "warning:" "$BUILD_LOG" | \
        grep -o "src/[^:]*\.rs" | \
        sort | uniq -c | sort -rn | head -10 | \
        tee -a "$BUILD_LOG"

    echo ""
    echo "================================================"
    echo "  Next Steps"
    echo "================================================"
    echo ""

    WARNING_COUNT=$(grep -c "warning:" "$BUILD_LOG" || true)

    if [ "$WARNING_COUNT" -eq 0 ]; then
        log_success "All warnings eliminated! Great job!"
    elif [ "$WARNING_COUNT" -lt 10 ]; then
        log_warning "Only $WARNING_COUNT warnings remaining. Almost done!"
        log_info "Review the warnings above and fix manually."
    else
        log_warning "$WARNING_COUNT warnings remaining."
        log_info "See CODE_CLEANUP_PLAN_DETAILED.md for manual fixes."
    fi

    echo ""
    log_info "Manual fixes needed:"
    log_info "1. Dead code removal (see CODE_CLEANUP_PLAN_DETAILED.md Phase 1)"
    log_info "2. Naming convention fixes (see Phase 3)"
    log_info "3. Clippy warnings requiring refactoring (see Phase 4)"
    echo ""

    if [ "$DRY_RUN" = false ]; then
        log_info "To review changes:"
        echo "  git diff"
        echo ""
        log_info "To commit changes:"
        echo "  git add -A"
        echo "  git commit -m 'chore: cleanup compiler warnings'"
        echo ""
    fi

    log_success "Report saved to: $BUILD_LOG"
}

# Main execution
main() {
    # Run phases based on selection
    case $PHASE in
        1)
            phase1_auto_fix
            ;;
        2)
            phase2_clippy_fix
            ;;
        3)
            phase3_format
            ;;
        4)
            phase4_verify
            ;;
        all)
            phase1_auto_fix
            phase2_clippy_fix
            phase3_format
            phase4_verify
            phase5_report
            ;;
        *)
            log_error "Invalid phase: $PHASE"
            log_info "Valid phases: 1, 2, 3, 4, all"
            exit 1
            ;;
    esac
}

# Run main
main

log_success "Cleanup script completed!"
echo ""
echo "Full log: $BUILD_LOG"
echo ""

# Exit with appropriate code
if [ "$DRY_RUN" = true ]; then
    exit 0
fi

WARNING_COUNT=$(grep -c "warning:" "$BUILD_LOG" || true)
if [ "$WARNING_COUNT" -gt 0 ]; then
    log_warning "Cleanup complete but $WARNING_COUNT warnings remain"
    exit 0
else
    log_success "All warnings eliminated!"
    exit 0
fi
