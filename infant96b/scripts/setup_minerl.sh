#!/usr/bin/env bash
set -euo pipefail

if ! command -v java >/dev/null 2>&1; then
  echo "Java not found. Install Java 8 (JDK 8) and re-run." >&2
  exit 1
fi

JAVA_VER_OUTPUT="$(java -version 2>&1 | head -n 1)"
if echo "$JAVA_VER_OUTPUT" | grep -Eq '"1\.8|"8'; then
  echo "Java 8 detected: $JAVA_VER_OUTPUT"
else
  echo "Java 8 required. Found: $JAVA_VER_OUTPUT" >&2
  exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
"$PYTHON_BIN" -m pip install --upgrade pip
"$PYTHON_BIN" -m pip install "minerl==0.4.4"

echo "MineRL 0.4.4 installed."
