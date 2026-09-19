#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="${ROOT}/.build"
mkdir -p "$BUILD"
for tool in pandoc xelatex; do
  command -v "$tool" >/dev/null || { echo "Required command not found: $tool" >&2; exit 1; }
done
pandoc "$ROOT/csgn_paper_v0_4_1.md" \
  --standalone --from markdown+tex_math_dollars+autolink_bare_uris --to latex \
  --template "$ROOT/rendering/template.tex" \
  --include-in-header "$ROOT/rendering/style.tex" \
  --lua-filter "$ROOT/rendering/pagebreaks.lua" \
  --toc --toc-depth=2 \
  -V colorlinks=true -V documentclass=article -V fontsize=11pt -V papersize=a4 \
  -V geometry:margin=24mm -V mainfont='Noto Serif' -V sansfont='Inter' \
  -V monofont='DejaVu Sans Mono' -V mathfont='Asana Math' \
  -o "$ROOT/csgn_paper_v0_4_1.tex"
for pass in 1 2 3; do
  xelatex -interaction=nonstopmode -halt-on-error -output-directory="$BUILD" \
    "$ROOT/csgn_paper_v0_4_1.tex" > "$BUILD/compile_stdout.txt" 2>&1 || {
    tail -50 "$BUILD/compile_stdout.txt" >&2; exit 1;
  }
done
cp "$BUILD/csgn_paper_v0_4_1.pdf" "$ROOT/csgn_paper_v0_4_1.pdf"
printf 'Created %s\n' "$ROOT/csgn_paper_v0_4_1.pdf"
