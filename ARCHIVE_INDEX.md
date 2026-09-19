# Historical archive index

Repository: Patwrick/Neuroplastic. The only active project folder is
`C:\Users\Patrick\Documents\GitHub\Neuroplastic`, with revised v0.4.1 on `main`.
Legacy source is preserved on local `legacy/pre-v0.4.1` at `f0bd287`, including
the outstanding source changes committed through GitHub Desktop during workspace
correction. The completed M0–M2 implementation and reports remain on local
`research/v0.4.1` at `1cccb5f`. Use GitHub Desktop for repository operations.

Historical source branch: `rewiring`.
Historical base commit: `6fa1c850f067de36db0c25a732873aa4d72ff181`.
Annotated local archive tag: `archive/pre-v041-20260919T135101Z-6fa1c850`.
The former separate `Neuroplastic-v0.4.1` worktree was retired after source and
evidence preservation. Its 88 local M0–M2 evidence files were copied unchanged
into the canonical project's `outputs/v0.4.1/m0-m2-20260919/` and verified by size
and SHA-256. This index points to existing preservation records; no additional
archive folder or archival workflow is required.

Approved private local archive:
`C:\Users\Patrick\Desktop\ResearchArchives\Neuroplastic\2026-09-19`.
The user moved ResearchArchives to Desktop on 2026-09-19. All 347 preservation
payload files and 20 completion files were rechecked successfully after the move.
See `ARCHIVE_LOCATION.md` and `relocations/20260919T142309Z/` for the relocation record.
Historical absolute paths inside sealed records describe the original location.

Start with `RESTORE.md`, `BACKUP_VERIFICATION.md`, `EXCLUSIONS.json`, and
`PRESERVATION_MANIFEST.sha256.json` there. The manifest SHA-256 is
`f8e2bbd247f3d8b69e735a620bf0585eaecefa77a49d67dfe0e40c1f78448f7e`.

- `git/legacy-reachable.bundle`: all seven available local refs, verified by
  bundle verification, independent mirror fsck and exact ref recovery. This is
  not a full remote backup. No additional refs were fetched or pushed.
- `working-tree/research.zip` and its manifest: all 831 original non-environment
  research files, including dirty, untracked and ignored bytes. All extracted
  hashes matched. Staged/unstaged binary patches are separately under `git/`.
- `volumes/`: all four inventoried Docker volumes; all 686 file hashes passed
  extraction and stable-source checks. Empty volumes have empty tar archives.
- `release/csgn_v0_4_1_codex_handoff.zip`: intact original, original manifest
  verified and package validator passed 248 integrity/schema checks.
- `environment/`: legacy environment metadata. Full .venv bytes, external base
  interpreter, Docker image layers and unreachable Git objects are not included;
  these are historical coverage limits, not active-environment instructions.
  Credentials/global config are not ordinary payloads.

The approved path plan/diff are in `plan/`; execution records are in `completion/`.
Historical algorithms, checkpoints and every outcome remain evidence for their
original version only. None is a v0.4.1 specification or learned result.
The active handoff is an explicitly manifested subset; its omitted archive files
remain in the intact original release. See README.md for current workspace
guidance and CLEAN_START_REPORT.md for the dated preparation record. Review
legacy material only for a concrete unresolved issue and record the rationale;
it does not define the new model.
