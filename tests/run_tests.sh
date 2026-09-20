#!/usr/bin/env bash
# run_tests.sh – Run the full stamps test suite
# Usage:
#   ./run_tests.sh            # default: concise output
#   ./run_tests.sh -v         # verbose output
#   ./run_tests.sh -k <expr>  # run tests matching <expr>
#   ./run_tests.sh --cov      # run with coverage report

set -euo pipefail

# ── Resolve the repo root (directory containing this script) ──────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Colour helpers ────────────────────────────────────────────────────────────
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Colour

echo -e "${YELLOW}========================================${NC}"
echo -e "${YELLOW}  stamps – Test Suite Runner            ${NC}"
echo -e "${YELLOW}========================================${NC}"
echo "  Root : $SCRIPT_DIR"
echo "  Date : $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# ── Parse optional flags ──────────────────────────────────────────────────────
COVERAGE=false
EXTRA_ARGS=()

for arg in "$@"; do
    case "$arg" in
        --cov)
            COVERAGE=true
            ;;
        *)
            EXTRA_ARGS+=("$arg")
            ;;
    esac
done

# ── Build the pytest command ──────────────────────────────────────────────────
PYTEST_CMD=(
    python -m pytest
    tests/
    --tb=short          # compact traceback on failure
    -q                  # quiet summary (dots + failures)
    -W default          # show all warnings
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}   # bash 3.2-safe empty-array expansion
)

if $COVERAGE; then
    # requires pytest-cov: pip install pytest-cov
    PYTEST_CMD+=(
        --cov=stamps
        --cov-report=term-missing
        --cov-report=html:htmlcov
    )
    echo -e "${YELLOW}Coverage report enabled  →  htmlcov/index.html${NC}"
    echo ""
fi

# ── Run ───────────────────────────────────────────────────────────────────────
set +e
"${PYTEST_CMD[@]}"
EXIT_CODE=$?
set -e

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✓  All tests passed.${NC}"
else
    echo -e "${RED}✗  One or more tests failed (exit code: $EXIT_CODE).${NC}"
fi

exit $EXIT_CODE
