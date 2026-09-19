# CSGN v0.4.1 workspace instructions

Read /README.md (repository entry), then /Docs/research/csgn_v0_4_1/WORKING_SUBSET.md
and /Docs/research/csgn_v0_4_1/README.md (handoff entry). The authoritative research
paths are /Docs/research/csgn_v0_4_1/paper/csgn_paper_v0_4_1.md and
/Docs/research/csgn_v0_4_1/handoff/IMPLEMENTATION_CONTRACT.md. Also read
handoff/CODEX_HANDOFF.md, EXPERIMENT_PROTOCOL.md, CONTRACT_TEST_MAP.md, and
REFERENCE_PROFILE_DETAILS.md in that directory before implementation.

- Validate WORKING_SUBSET_MANIFEST.sha256.json with the root verification script.
  The original manifest is preserved, but historical archive files are omitted;
  never report the subset as a complete original release.
- This task is preparation only. Do not begin M1/M2 or learned runs. A fresh
  implementation task must re-read applicable instructions and the current paper.
  Original package prompts asking to implement M0-M2 do not expand this task.
- Use src/neuroplastic_v041 and tests/v041. Use a fresh .venv-v041 environment.
  Never import legacy csgn, use legacy checkpoint/defaults, or impose historical
  backward compatibility without an explicit reviewed decision.
- Report scientific discrepancies and stop the affected work. Do not silently
  repair the paper/reference with old equations or edit the paper to fit code.
- Preserve continuous outer gradients and causal feedback boundaries; keep
  per-lifetime state independent. Reference functions are test oracles for later
  implementation, not a completed model or proof of learned capability.
- Preserve shipped paper, verification bytes and provenance. New outputs use
  outputs/v0.4.1/<unique-run-id>/ and handoff schemas/templates. Preserve all seeds
  and negative, failed, aborted, inconclusive, and positive results.
- No fabricated results, hidden skipped checks, relaxed tolerances for passing,
  future-label leakage, or claims of global convergence from local checks.
- Keep the original checkout, uncommitted work, archive refs and private backups.
  No reset --hard, broad clean, history rewrite, orphan replacement, force-push,
  deletion of branches/tags/data, merge into main, or unapproved tracked cleanup.
- No uploads of research/raw data/archives, visibility changes, paid/cloud compute,
  secret values in reports, or personal/global Codex changes without permission.
  Leave credentials and .env files untouched; separate private/encrypted backup
  requires explicit handling. Do not recursively unpack historical archives here.
- Package instructions are nested in their own scope; this root file carries
  applicable research and security boundaries into source/test/config work.
