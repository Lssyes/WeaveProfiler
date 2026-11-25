#!/usr/bin/env bash
# Run the legacy offline suite (tests/run_all.sh) and compare its failures with tests/known_failures.txt.
#   - a failure that is not listed      -> exit 1 (regression)
#   - a listed failure that now passes  -> exit 1 (remove it from the list in the same PR)
# Usage: PYTHON=python3.12 bash tests/ci_offline.sh
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 2
PY="$(command -v "${PYTHON:-python3}")"
LOG="$(mktemp)"
KEEP_DIR=""
if [ "${GITHUB_ACTIONS:-}" = "true" ]; then
  KEEP_DIR="$(mktemp -d)"
  bash tests/run_all.sh --keep-dir "$KEEP_DIR" --python "$PY" | tee "$LOG"
else
  bash tests/run_all.sh --python "$PY" | tee "$LOG"
fi

failed="$(grep -E ' FAIL\([0-9]+\) ' "$LOG" | awk '{print $2}' | sort -u)"
known="$(grep -vE '^[[:space:]]*(#|$)' tests/known_failures.txt | awk '{print $1}' | sort -u)"
unexpected="$(comm -23 <(printf '%s\n' "$failed" | sed '/^$/d') <(printf '%s\n' "$known" | sed '/^$/d'))"
fixed="$(comm -13 <(printf '%s\n' "$failed" | sed '/^$/d') <(printf '%s\n' "$known" | sed '/^$/d'))"
rm -f "$LOG"

gha_escape() {
  local s="$1"
  s="${s//%/%25}"
  s="${s//$'\r'/%0D}"
  s="${s//$'\n'/%0A}"
  printf '%s' "$s"
}

gha_escape_prop() {
  local s
  s="$(gha_escape "$1")"
  s="${s//:/%3A}"
  s="${s//,/%2C}"
  printf '%s' "$s"
}

gha_annotate_item() {
  local tag="$1" safe_tag item_log="" m body
  safe_tag="$(printf '%s' "$tag" | tr -c 'A-Za-z0-9._-' '_')"
  if [ "$tag" = "no" ] && [ -f "$KEEP_DIR/cache_added.txt" ]; then
    item_log="$KEEP_DIR/cache_added.txt"
  else
    for m in "$KEEP_DIR/logs/"[0-9][0-9][0-9]_[A-Z]*_"${safe_tag}".log \
             "$KEEP_DIR/logs/"[0-9][0-9][0-9]_[A-Z]*_"${safe_tag}"_*.log; do
      if [ -f "$m" ]; then
        item_log="$m"
        break
      fi
    done
  fi
  if [ -n "$item_log" ] && [ -f "$item_log" ]; then
    body="$(tail -n 20 "$item_log" 2>/dev/null || true)"
  else
    body="日志文件未找到"
  fi
  printf '::error title=%s::%s\n' "$(gha_escape_prop "$tag")" "$(gha_escape "$body")"
}

status=0
if [ -n "$unexpected" ]; then
  echo "NEW FAILURES (not in tests/known_failures.txt):"; printf '  %s\n' $unexpected; status=1
  if [ "${GITHUB_ACTIONS:-}" = "true" ]; then
    for item in $unexpected; do gha_annotate_item "$item"; done
  fi
fi
if [ -n "$fixed" ]; then
  echo "KNOWN FAILURES THAT NOW PASS (remove them from tests/known_failures.txt):"; printf '  %s\n' $fixed; status=1
  if [ "${GITHUB_ACTIONS:-}" = "true" ]; then
    for item in $fixed; do gha_annotate_item "$item"; done
  fi
fi
[ -n "$KEEP_DIR" ] && rm -rf "$KEEP_DIR"
[ "$status" -eq 0 ] && echo "offline suite: only known failures"
exit "$status"
