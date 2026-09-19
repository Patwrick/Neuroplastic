# CSGN v0.4.1 workspace correction — report-back

## 1. Identity and status

- Run: `workspace-correction-20260919`; workspace/environment verification completed on 19 September 2026. No new research milestone was started.
- Canonical and only active folder: `C:\Users\Patrick\Documents\GitHub\Neuroplastic`; repository `Patwrick/Neuroplastic`; active branch `main`.
- GitHub Desktop created `legacy/pre-v0.4.1` from `rewiring` and committed seven outstanding source/ignore files as `f0bd287`. The branch retains the original implementation, including the untracked sparse model source.
- Desktop verified `research/v0.4.1` clean at `1cccb5f314d47530395a18fae985e3155005d206`. Its implementation commit remains `c049f23962282fcc348934e2e23b94743752324b`; no outstanding revised source needed another commit there.
- Desktop merged research into original-folder `main` as `7e105fd68fbe11c88bf03763f2993807b1633933`. This report accompanies the subsequent cleanup commit. The actual final Desktop commit identity and clean/dirty observation are recorded in `outputs/v0.4.1/workspace-correction-20260919/final_desktop_state.json` after committing, avoiding a self-referential hash.
- No command-line dirty patch was generated under the Desktop-only restriction. Tests ran on the merged implementation with pending documentation/cleanup changes; tested source hashes are in `canonical_test_execution.json`. This is not a claimed clean-commit experiment run.
- Original release ZIP SHA-256: `15152533d79407f84dd95778e255624fb4c07079f0814772d3819c02b32ccae2`.
- Subset manifest: `720329b2b3995ec43dc7f0e4d4792bd0096095f007ed582373c45f6b1a64eeb2`; paper: `abe9fe9173bfd6fc96c9194d690d88ed795f224fa2fc3c756b0aeca167cf5f36`; smoke config: `a21ac7eaa7689f71448f471e36dc9f82d51cf7c3ab9c076e361b9b4cae6cff75`. Released bytes remain unchanged.
- Windows 11; fresh canonical Python 3.13.5 environment; NumPy 2.3.5, SciPy 1.17.0, Torch 2.10.0 CPU, jsonschema 4.26.0, pytest 9.1.1. CUDA unavailable. Local contract checks only; no new learned or smoke run.
- Instruction sources: explicit workspace-correction request, root README/AGENTS, historical CLEAN_START_REPORT, working-subset note, package AGENTS/README, and handoff REPORT_BACK template. The current request supersedes preparation-only restrictions and authorizes cleanup/merge. Package science remains authoritative.

## 2. What actually changed

All repository mutations used GitHub Desktop: legacy branch creation/commit, worktree retirement, branch switch, merge/conflict resolution, and cleanup commit. Desktop removed the extra `Neuroplastic-v0.4.1` linked worktree after preservation. Both branches/commits remain; the original `.git` directory remains intact. No additional clone, worktree, or archive folder was created. No push, history rewrite, branch deletion, or remote mutation was requested or performed.

Before worktree retirement, all **88 files / 4,494,810 bytes** of local M0–M2 evidence were copied into the original project's `outputs/v0.4.1/m0-m2-20260919/`. Every size and SHA-256 matched; a post-cleanup recheck also passed. Failed checks, intermediate results, and negative findings remain intact.

Desktop's single merge conflict was the old lowercase `readme.md` (modified on main, deleted on research). The revised deletion was selected, and the canonical v0.4.1 README was recreated. Root AGENTS now reflects completed M0–M2, Desktop-only operations, the single original folder, and no M2A/M3. Six historical report bodies are unchanged below clearly labeled current-location addenda. ARCHIVE_INDEX now describes the correct branch/folder arrangement.

Removed obsolete working-tree content: legacy source bytecode remnants, old environment, old data/runs/logs, script bytecode, four superseded Docs files, and superseded preparation copies/instructions. Legacy source is committed on its preservation branch; the existing external archive and Git history preserve the historical material. All 112 preparation-review files were hash-matched to existing external copies; no archive was unpacked or created. No legacy algorithm/checkpoint/results were loaded for model design. Preservation verification read filenames, sizes, hashes and preservation metadata only; no targeted algorithm review was needed.

Restored legacy environment/data/log/secret ignore rules alongside revised exclusions. Kept the original release ZIP locally, ignored, at the repository root. Recreated `.venv-v041` instead of copying absolute editable-install paths; imports now resolve to the original `src/neuroplastic_v041`. No mathematical implementation or test logic changed. All 29 non-root-guidance files in the original smoke source manifest match exactly. Markdown LF handling was added while retaining the immutable research-byte attribute override.

## 3. Commands actually executed

All commands below ran from the canonical folder. `E` below abbreviates `outputs/v0.4.1/workspace-correction-20260919` in this table only; JSON command records contain exact argv, cwd, UTC times, exit codes and stdout/stderr filenames.

| Command | Exit | Wall seconds | Record/log prefix under E |
|---|---:|---:|---|
| `py -3.13 -m venv .venv-v041` | 0 | 5.894 | `environment_create` |
| `.venv-v041/Scripts/python.exe -m pip install -e '.[test]'` | 0 | 88.525 | `environment_install` |
| `.venv-v041/Scripts/python.exe -B E/inspect_environment.py` | 0 | 2.473 | `environment_inspect` |
| `.venv-v041/Scripts/python.exe -m pip freeze` | 0 | 0.625 | `environment_freeze`; exception below |
| `.venv-v041/Scripts/python.exe -I -B scripts/verify_handoff_subset.py` | 0 | 0.102 | `subset_verify` |
| `.venv-v041/Scripts/python.exe -B -m pytest -q -k 'not test_capture_' --junitxml=E/canonical_pytest.xml` | 0 | 6.826 | `canonical_pytest` |
| `.venv-v041/Scripts/python.exe -B Docs/research/csgn_v0_4_1/verification/check_math_v0_4_1.py --output E/canonical_reference_checks.json` | 0 | 5.350 | `canonical_reference` |
| Base Python 3.13.5 `-B E/verify_preservation.py` | 1, then 0 | 0.190, 0.186 | `preservation_verification_initial.json`, `preservation_verification.json` |

Ordinary filesystem operations used scoped `Get-ChildItem`, `Get-Content`, `Get-FileHash`, `Copy-Item`, `Remove-Item -LiteralPath`, and file edits. Recursive deletion targets were resolved and checked against the original workspace; `.git`, revised source/handoff and new outputs were excluded. Inventory: `obsolete_paths_removed.json`, `superseded_preparation_files.json`. These were file cleanup operations, not terminal repository operations. Desktop actions and their actual commit outcomes are described above and in the final receipt.

## 4. Tests

**160 passed, 0 failed, 0 skipped, 3 explicitly deselected**, 5.76 seconds pytest time. The unchanged tests cover the same contract/gradient, causal feedback, graph/delta, and lifecycle behavior as before. No tolerance or test logic was changed. The excluded tests are:

- `test_capture_includes_untracked_bytes_and_exact_patch`
- `test_capture_rejects_recursive_unignored_output`
- `test_capture_rejects_midcapture_source_change`

Their fixtures invoke command-line Git in temporary repositories. Excluding them respects the repository-operation restriction; they are not claimed as executed now. The historical **163-pass** result remains preserved and clearly belongs to the earlier run.

Subset integrity: **47 files passed**, no errors; original release completeness is explicitly **not** claimed. Reference verification: **24 groups / 19,438 cases passed**, seed 20260916, 4.832 seconds checker time. These are local algebra/numerical/gradient checks. Original evidence preservation: **88 files passed**. Implementation preservation: **29 files passed**.

## 5. Results, controls and resources

No new task seeds, theta training, cold-recall metrics, confidence intervals, sweeps, GPU runs, or architecture comparisons were generated. Environment installation and local CPU checks are the only new execution. Commands include interpreter startup in wall times; pytest and reference checks ran concurrently. Peak resource use was not newly measured. The original smoke results and resource limitations remain in REPORT_BACK and the unchanged evidence directory.

## 6. Diagnostics

Import provenance confirms the original folder's `.venv-v041` and `src` paths. The extra checkout is absent; the original `.git` directory exists. No Git metadata contents were directly inspected or edited by correction scripts.

The first preservation check detected six README byte mismatches caused by Windows CRLF checkout conversion, with no mathematical-source mismatch. The failed JSON is preserved. Restoring their original LF bytes made all 29 file hashes match; `.gitattributes` now keeps Markdown LF outside the unchanged release-byte override.

The historical weak graph cold recall, harmful writes, support discontinuity, bounded capacity floors, and unequal baseline geometry remain findings, not repaired claims. No new runtime model diagnostics were collected.

## 7. Deviations and open specification questions

- **WC01 — indirect Git inspection:** the dependency-recording `pip freeze` command automatically resolved editable-package VCS provenance and emitted a repository URL/commit. This was an unintended read-only exception to the intended Desktop-only boundary. No direct Git command was issued by an agent, and all mutations stayed in Desktop. The exception was disclosed immediately, logs retained, and no further freeze/VCS probe was run. Installed versions/import paths use `importlib.metadata` instead. Do not use `pip freeze` here under this restriction.
- **WC02 — excluded provenance tests:** three tests remain unexecuted in this correction; see section 4. The smoke CLI also performs Git provenance queries and was not rerun. No replacement mocked tests were invented.
- **WC03 — line endings:** the six documentation-byte mismatches and repair are recorded above; release and mathematical code bytes were unchanged.
- Original scientific and engineering questions D01–D10 remain in REPORT_BACK. Workspace correction does not resolve them. No new paper discrepancy or legacy-based equation change was introduced.

## 8. Claim assessment

The folder arrangement, preserved evidence, imports and executed local contracts are verified. H1/H2/H3 remain untested as research efficacy claims. Neither this rerun nor the original smoke test validates the architecture.

## 9. Next bounded action

Stop after Desktop cleanup commit and final state verification. **Do not start M2A or M3.** No additional experiment is authorized by this report. A future provenance runner compatible with the current repository-operation restriction is an open tooling question, not part of this correction.

## 10. Artifact inventory

Local correction evidence: `outputs/v0.4.1/workspace-correction-20260919/`.

- `evidence_transfer.json`: 88-file copy manifest, SHA-256 `1a5b233016649f48dd3ee07208738cf0fd249d19ffec7c8049c7a95781a83e07`.
- `legacy_source_before_commit.json`: seven-file legacy source/ignore hashes before Desktop commit.
- `canonical_test_execution.json`: exact commands, results, exclusions, environment and tested source hashes.
- `canonical_test_artifacts_manifest.json`: hashes/sizes of the 28 environment/test evidence files.
- `canonical_pytest.xml`, reference JSON, stdout/stderr and per-command JSON records: executed verification evidence.
- `preservation_verification_initial.json` and `preservation_verification.json`: preserved failing and passing byte checks.
- `obsolete_paths_removed.json`, `superseded_preparation_files.json`: file cleanup inventory.
- `final_desktop_state.json`: final UI-verified branch, commit, cleanliness and folder receipt; `correction_artifact_manifest.json`: final correction evidence hashes.

The original M0–M2 outputs remain unchanged and local. No environments, credentials, raw experiment outputs or ZIP archives are included in the cleanup commit. No data was published.
