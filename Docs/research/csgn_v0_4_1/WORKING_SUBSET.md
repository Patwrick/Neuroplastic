# Working subset of the verified CSGN v0.4.1 release

This directory contains 47 unchanged release files plus this note and
WORKING_SUBSET_MANIFEST.sha256.json. Three archive/ entries are intentionally
omitted: README.md, csgn_math_reassessment.zip, and csgn_v0_4_research_package.zip.
The complete original ZIP must remain intact in the private archive.

MANIFEST.sha256.json is the unchanged ORIGINAL manifest, retained for provenance.
This subset does not pass its original completeness check. Do not claim otherwise.
Use the root command `python -I scripts/verify_handoff_subset.py` for subset
integrity. Run the original scripts/validate_package.py only from a complete
separate extraction of the original ZIP in a fresh environment. Never recursively
unpack the historical ZIPs into this implementation tree.

README.md here is the handoff README. The repository README is /README.md.
The paper and current handoff define the research baseline. A paper/reference
conflict must be reported; historical equations cannot resolve it implicitly.
START_CODEX_PROMPT.md and the nested AGENTS.md are original release artifacts;
their M0-M2 requests do not authorize implementation in the preparation task.
