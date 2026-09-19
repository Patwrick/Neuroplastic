# Rebuilding the paper

The editable source is `csgn_paper_v0_4_1.md`. LaTeX source and the rendered PDF are supplied. Equation numbers are retained from v0.4 for traceability; the focused revision is integrated into the body and Appendix A.

The supplied build uses Pandoc and XeLaTeX plus standard TeX packages. Fonts are referenced by installed family name: Noto Serif, Inter, DejaVu Sans Mono, and Asana Math. **Font files are not included.** Substitute locally installed families in `rendering/build_pdf.sh` when necessary; preserve mathematical symbols and inspect output after any font change.

```bash
bash paper/rendering/build_pdf.sh
```

Run from the extracted package root. The script builds in `paper/.build/` and copies the final PDF to `paper/`. Build artifacts are not research evidence. Inspect all pages after a rebuild, including long equations, tables, headers, references and code paths. Package hashes must be regenerated for a new release; do not edit the original manifest and continue to call the files unchanged.

PDF generation does not execute experiment code. See the package README for local verification and Codex implementation instructions.
