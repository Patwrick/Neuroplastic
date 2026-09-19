# v0.4.1 change record

Date: 16 September 2026. Full manuscript revision, not an implementation patch.

## Mathematical changes

- Equation (4): replace the conservative absolute-sum slow/fast reserve with four endpoint constraints and the retained absolute-sum fast budget.
- Endpoint lemma: the constraint is necessary and sufficient for effective-weight safety under all independent decay factors in [0,1]. This is a local feasibility statement, not a learner stability theorem.
- Equations (18)–(19): derive asymmetric residual intervals and their effective-weight/step-budget intersection. Projection is still exactly componentwise clipping.
- Proposition 6: update its proof to endpoint convexity, preserving exact transfer and the fast budget.
- Preserve local projected-descent assumptions. Explicitly distinguish fixed inner features from the outer gradient path through their construction and through the functional update.
- Add a bounded-LS diagnostic with upper/lower objective bounds and a primal–dual gap, plus downstream write-utility measurement. A numerical achieved residual is not automatically an irreducible floor.

## Experimental and engineering changes

- Add compatible slow-memory transplantation, matched selective erasure, correct no-information controls and unseen-input rule transfer.
- Promote simple delta memory, with appropriate slow/replay variants, to primary comparator.
- Add stable addressing and gradient-path ablations. Keep contraction as a sufficient reference and sweep its cost inside the bound.
- Separate empirical transaction screens, independent research evaluation and optional formal conditional acceptance; explicitly count the latter's sample cost.
- Keep the small complete lifecycle. Defer structural, precision, async and online-encoder complexity until warranted.
- Supply a non-destructive Codex restart prompt, implementation contracts, task protocols, proposed configs, schemas and a report-back/revision loop.

## Evidence in this release

24 newly executed local groups / 19,438 cases passed programmed tolerances. The checks include NumPy/PyTorch parity and finite differences through a continuous fixed-support update. This is not integrated model training. Original v0.4 and reassessment archives are preserved as historical sources; previous results are not relabeled as new.

## Repository status

Read-only metadata/README inspection recorded commit 281746c9ae962634df04ec34b3ba212af11b3cfd. No repository writes, branch changes, visibility changes, or publication occurred. Codex must inspect its live checkout and preserve newer/user work.
