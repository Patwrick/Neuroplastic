#!/usr/bin/env bash
set -euo pipefail

DISPLAY_ID="${XVFB_DISPLAY:-:99}"
DISPLAY_SOCKET="/tmp/.X11-unix/X${DISPLAY_ID#:}"
XVFB_LOG="/tmp/xvfb.log"

if [ "$#" -eq 0 ]; then
  echo "No command provided to run_with_xvfb.sh" >&2
  exit 2
fi

echo "Starting Xvfb..." >&2
Xvfb "$DISPLAY_ID" -screen 0 640x360x24 -ac -nolisten tcp +extension GLX -noreset >"$XVFB_LOG" 2>&1 &
XVFB_PID=$!

cleanup() {
  if kill -0 "$XVFB_PID" >/dev/null 2>&1; then
    kill "$XVFB_PID" >/dev/null 2>&1 || true
    wait "$XVFB_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

for _ in $(seq 1 50); do
  if [ -S "$DISPLAY_SOCKET" ]; then
    export DISPLAY="$DISPLAY_ID"
    echo "DISPLAY=$DISPLAY ready" >&2
    echo "Running: $*" >&2
    "$@"
    exit $?
  fi
  sleep 0.1
done

echo "Xvfb failed to become ready within timeout." >&2
if [ -f "$XVFB_LOG" ]; then
  cat "$XVFB_LOG" >&2
fi
exit 1
