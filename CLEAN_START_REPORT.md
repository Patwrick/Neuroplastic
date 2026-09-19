# CLEAN_START_REPORT

Preparation complete after explicit approval. No model, M1/M2, reference-math
rerun or learned experiment was implemented or launched.

## Actual workspace and commits

- Branch: `research/v0.4.1` (ordinary branch in the same repository).
- Clean workspace: `C:\Users\Patrick\Documents\GitHub\Neuroplastic-v0.4.1`.
- Reviewed content commit: `58ffe85f3b9d71b056e376684764192ece202282`.
- Cleanup commit: `29a038eb5d7336272b9fb4bc523d6544b8a106f3`.
- Base/history parent: `6fa1c850f067de36db0c25a732873aa4d72ff181`.
- The reviewed content commit's tree is `83ef3180bdcd50633225b4ef4628f374c21a36fc`, exactly
  equal to the approved proposal. Only factual ARCHIVE_INDEX.md and this report
  are committed on top. The containing documentation commit can be read with
  `git log -1 --format=%H -- CLEAN_START_REPORT.md`; its exact final HEAD is also
  recorded in the private `completion/final-state.json` and in the final report
  at the original checkout root. This avoids a self-referential commit hash.
- No push, merge, force operation, history rewrite, orphan history, branch/tag
  deletion, broad cleanup or original-checkout switch was performed.

## Original checkout preservation

Original location: `C:\Users\Patrick\Documents\GitHub\Neuroplastic`.
Original branch/commit: `rewiring` / `6fa1c850f067de36db0c25a732873aa4d72ff181`.
All 831 original non-environment research file hashes and the original Git index
were unchanged through backup/recovery and clean-worktree preparation. Original
staged set remains empty. These five tracked modifications remain in place:

- `csgn/models/__init__.py`
- `readme.md`
- `scripts/eval_plastic_cue.py`
- `scripts/summarize_cue_rollouts.py`
- `scripts/train_plastic_cue.py`

Original untracked `csgn/models/sparse_plastic_mlp.py` and
`csgn_v0_4_1_codex_handoff.zip` remain intact, along with ignored data, checkpoints,
logs and the old .venv. Generated review/report files remain at the source root.
The local annotated archive tag is `archive/pre-v041-20260919T135101Z-6fa1c850`.

## Retained files and removed paths

The approved change removed 94 tracked legacy paths and replaced .gitignore only
in this new worktree. It removed old `csgn/**` modules; old training/evaluation,
rollout, setup and summary `scripts/**`; `docker/**`; `.dockerignore`;
`requirements-core.txt`, `requirements-ml.txt`, `requirements-minestudio.txt`;
`Docs/csgn_paper_v0_2.md`, `Docs/csgn_paper_v0_2.html`; old root `readme.md`;
28 tracked trajectory files under `data/trajectories/**`; and the three tracked
legacy `runs/**` artifacts. The exact 94-path list and old/new blob identities are
in private `completion/clean-tree-application.json` and `plan/archive-review/exact-path-delta.json`.
No ignored/untracked originals were removed. The v0.3 paper and feasibility audit
remain in the saved cached origin/main history, not the active tree.

Retained/added active files are the 47 unchanged selected release files under
`Docs/research/csgn_v0_4_1/`, two subset metadata files, root README.md and AGENTS.md,
ARCHIVE_INDEX.md, .gitattributes, .gitignore, pyproject.toml, the subset verifier,
source/test/config placeholders, and this completion report. No legacy utility
was retained: seed, JSON serialization and CSV logging helpers were reviewed and
their provenance/contract concerns recorded in ARCHIVE_PLAN.md outside this tree.
No old algorithm, checkpoint, raw output or old experimental default was migrated.
Reference mathematics remains supplied research material, not an implemented model.

## Backup coverage and verification

Approved private destination: `C:\Users\Patrick\ResearchArchives\Neuroplastic\2026-09-19`.
It is outside both worktrees. Inheritance is disabled; ACLs grant access only to
the current user, SYSTEM and Administrators. No uploads or visibility changes.

| Artifact/check | Actual result |
|---|---|
| Available reachable Git history | Bundle verify passed; independent mirror fsck passed; all 7 advertised local refs restored exactly; 95 base tracked files restored and hashed |
| Bundle | `git/legacy-reachable.bundle`, 37,401,364 bytes, SHA-256 `5aeffe6823d01366faa2b57b852f9d1c1b09438e53fbb5cb78809ce70da7d69c` |
| Working research snapshot | 831 files / 207,998,969 original bytes; ZIP integrity, extracted hashes and source rehash all passed |
| Snapshot ZIP | `working-tree/research.zip`, 45,736,580 bytes, SHA-256 `1c8ad6a18ec781c600cd08a3ae1051c76c489ac159cdcda35120a0bf6011f865` |
| Local changes | Staged binary patch recorded as 0 bytes; unstaged patch 65,352 bytes; exact modified/untracked bytes preserved; both restored patches reproduce exactly |
| Docker volumes | Four volume archives, 686 files / 759,396,035 bytes; all extracted and source hashes passed; empty volumes preserved too |
| Legacy environment | Python 3.10.11 base path, pyvenv.cfg, 148 package versions and actual METADATA files preserved; old environment unchanged |
| Original handoff | ZIP untouched; all 49 original manifest sizes/hashes matched before use and after separate full extraction; nested historical ZIPs remain opaque |
| Original package validator | 248 checks passed with hash checks on, in a new Python 3.13.5 audit environment with jsonschema 4.26.0; validates integrity/schema consistency only |
| Active subset | 47 unchanged release files pass WORKING_SUBSET_MANIFEST; three archive entries omitted explicitly; no original-completeness claim |
| Preservation payload manifest | 347 payload/metadata files checked, 560,186,684 bytes at sealing; SHA-256 `f8e2bbd247f3d8b69e735a620bf0585eaecefa77a49d67dfe0e40c1f78448f7e` |

The source clone is non-shallow, with no detected partial/promisor config,
alternates, LFS pointers/objects, submodules or symlinks. No refs were fetched.
This is an available-local-history backup, not a full remote/GitHub backup.
The first full fsck was concurrent with archive-tag creation and reported that
new tag transiently; sequential fsck and tag-object checks passed before backup.
Recovery first verified raw Git bytes with core.autocrlf=false, then restored the
source's true setting in the recovery clone to compare exact working patches.
No source setting was changed. Raw restored research bytes always matched hashes.

RESTORE.md documents safe separate-directory recovery, patch application,
byte overlays, line endings and volume restoration. All individual relative paths,
sizes and SHA-256 hashes are recorded in the private manifests. Originals were
kept even after verification. Secret screening printed no values; high-confidence
scans found no matches in inspected research text/reachable text blobs and Docker
volume bytes. This is a heuristic check, not proof about arbitrary binary payloads.

## Fresh setup and isolation checks

- Fresh environment: `.venv-v041`, Python 3.13.5, system-site-packages disabled.
  Installed only the editable empty `neuroplastic-v041` package and pytest 9.1.1
  with its dependencies. No legacy environment or global config was modified.
- Import check resolved `neuroplastic_v041` to this worktree's
  `src/neuroplastic_v041/__init__.py`; `find_spec('csgn')` returned None. The old
  checkout was absent from sys.path. Torch is not installed in this environment.
- Pytest collection uses `tests/v041` and importlib mode. It collected zero tests
  and exited 5, as expected for the empty skeleton; no test suite is claimed to
  pass. No model tests were written or executed.
- New evidence root is `outputs/v0.4.1/<unique-run-id>/`. Output/environment ignore
  rules were checked. Legacy csgn/data/runs/logs/docker runtime paths are absent.
- .gitattributes disables text conversion for all verified release files; the
  reference Python and paper attributes were checked explicitly.
- Full reference dependency installation and reference-math rerun belong to the
  future implementation task. Original verification results were not overwritten.

## Instruction sources and conflicts

Current user/host instructions governed preparation and the explicit “approved”
message authorized the reviewed plan and private destination. The source had no
root/nested/ancestor AGENTS override or project .codex config. Global
`C:\Users\Patrick\.codex\AGENTS.md` is empty; no global override was found.
The global config was inspected read-only for relevant overrides, with none
found for custom/project-document instructions. Its existing trust/model/plugin
settings remain untouched. The OpenAI Docs skill and official instruction-chain
guidance were consulted during inventory; the host's local-first inspection rule
took precedence over that skill's docs-first ordering.

This workspace now has the approved root AGENTS.md, which carries security and
research boundaries into source/test/config scope. The nested unchanged package
AGENTS applies within the package. Its M0-M2 implementation request and starting
prompt do not override this task's preparation-only limit. The paper is research
authority; contracts operationalize it; configs are candidate settings; history
is evidence only. Report a scientific discrepancy rather than importing older
equations or silently changing the paper/reference. No mathematical equivalence
audit or discrepancy repair was performed during preparation.

Cleaning tracked files does not remove global settings or this conversation's
context. Begin implementation in a fresh task rooted in this new worktree and
re-read the root instructions plus current paper/handoff.

## Unresolved coverage and research limits

- Unreachable/reflog-only Git objects remain only in the original Git database;
  they were not included in the reachable-history bundle and were not pruned.
- Full old .venv bytes, external base Python installation, and Docker image layers
  were not exported under the approved minimal scope. Original copies remain;
  their metadata alone is not a complete executable environment backup.
- Docker database db/wal/shm bytes were stable and restored as a set, but no
  application/database recovery or MineStudio experiment was run.
- Original Windows ACLs/alternate data streams are not guaranteed by byte archives.
  No credentials/.env contents were put in ordinary archives. Any separately
  required private/encrypted credential backup still needs distinct handling.
- 98 historical run metadata files contain seeds; 94 have top-level Git/config
  fields. Missing identities and historical dirty-code/dependency limits remain
  documented; they were not retrofitted or presented as v0.4.1 learned evidence.
- No M1/M2 or learned capability claims; no full remote-backup claim.

## Exact authoritative entry paths

All paths below are relative to this new worktree, not the original checkout:

1. `README.md` — root repository README.
2. `AGENTS.md` — root instructions for future implementation.
3. `Docs/research/csgn_v0_4_1/WORKING_SUBSET.md` — subset omissions and verification.
4. `Docs/research/csgn_v0_4_1/README.md` — handoff README, distinct from root.
5. `Docs/research/csgn_v0_4_1/handoff/CODEX_HANDOFF.md`.
6. `Docs/research/csgn_v0_4_1/paper/csgn_paper_v0_4_1.md` — normative mathematics.
7. `Docs/research/csgn_v0_4_1/paper/csgn_paper_v0_4_1.pdf` — preserved PDF.
8. `Docs/research/csgn_v0_4_1/handoff/IMPLEMENTATION_CONTRACT.md`.
9. `Docs/research/csgn_v0_4_1/handoff/EXPERIMENT_PROTOCOL.md`.
10. `Docs/research/csgn_v0_4_1/handoff/CONTRACT_TEST_MAP.md` and `REFERENCE_PROFILE_DETAILS.md`.
11. `Docs/research/csgn_v0_4_1/WORKING_SUBSET_MANIFEST.sha256.json`.
12. `scripts/verify_handoff_subset.py` — integrity-only entrypoint.

The intact complete release is privately preserved at
`C:\Users\Patrick\ResearchArchives\Neuroplastic\2026-09-19\release\csgn_v0_4_1_codex_handoff.zip`
and its complete extraction at the sibling `csgn_v0_4_1_handoff/`. Use that complete
extraction for scripts/validate_package.py; this subset intentionally cannot pass
the original completeness check. Never recursively unpack historical archives here.
