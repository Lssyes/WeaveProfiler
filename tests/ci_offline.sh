#!/usr/bin/env bash
# Run the legacy offline suite (tests/run_all.sh) and compare its failures with tests/known_failures.txt.
#   - a failure that is not listed      -> exit 1 (regression)
#   - a listed failure that now passes  -> exit 1 (remove it from the list in the same PR)
# Usage: PYTHON=python3.12 bash tests/ci_offline.sh
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 2
PY="$(command -v "${PYTHON:-python3}")"
LOG="$(mktemp)"
bash tests/run_all.sh --python "$PY" | tee "$LOG"

failed="$(grep -E ' FAIL\([0-9]+\) ' "$LOG" | awk '{print $2}' | sort -u)"
known="$(grep -vE '^[[:space:]]*(#|$)' tests/known_failures.txt | awk '{print $1}' | sort -u)"
unexpected="$(comm -23 <(printf '%s\n' "$failed" | sed '/^$/d') <(printf '%s\n' "$known" | sed '/^$/d'))"
fixed="$(comm -13 <(printf '%s\n' "$failed" | sed '/^$/d') <(printf '%s\n' "$known" | sed '/^$/d'))"
rm -f "$LOG"

status=0
if [ -n "$unexpected" ]; then
  echo "NEW FAILURES (not in tests/known_failures.txt):"; printf '  %s\n' $unexpected; status=1
fi
if [ -n "$fixed" ]; then
  echo "KNOWN FAILURES THAT NOW PASS (remove them from tests/known_failures.txt):"; printf '  %s\n' $fixed; status=1
fi
[ "$status" -eq 0 ] && echo "offline suite: only known failures"
exit "$status"
