# Concrete initial profile choices

These choices make the handoff executable without claiming they are optimal or additional theorems. They instantiate the paper and must be recorded in configs/decision records if changed.

## Graph and output route

For the 512-column pilot, use 16 regions of 32 columns, 16 unique outgoing targets per source, four protected and twelve plastic slots, state width 64, message width 16, and four heads. Four designated output receivers are IDs 0–3. For the tiny 64-column smoke, keep four output receivers and use the smaller config widths.

A reproducible protected topology can reserve forward and backward ring edges, a next-region same-offset edge, and one output-delivery edge. Choose output receiver `(source mod 4)` or the first cyclic alternative that is not the source or an already selected target. Fill remaining targets without replacement from valid nodes using the topology RNG. Assign heads deterministically or from the recorded topology RNG. Validate actual connectivity, uniqueness and generation IDs. This is a proposed initial topology, not a proven optimal cortical organization.

Reserve one available output-delivery route per active sender in the executed support, then select the remaining real routes by adjusted logits. Keep null mass; a reserved support entry is not a guarantee of nonzero minimum traffic. Record this support policy rather than pretending the entire support was unconstrained top-k. A no-reserved-delivery ablation may be informative later, but do not start with a random graph where almost no writable edges reach the supervised output.

A maximum of four real routes per sender is allowed. With 64 senders, there are at most 256 executed real edges per inner round. Cap admitted incoming routes and updated node count per config; record rejections. Recurrent computation still uses the graph, not just a direct cue-to-target table.

## Prediction and snapshot

At the final inner round capture receiver aggregates `r[j]` before any plastic write. The association prediction is a fixed linear decoder: average the first eight message coordinates of the four output receivers. If dm<8 in a future profile, choose and record another output interface rather than truncate targets silently. This decoder is common to compatible transplantation agents. The final h update may prepare future state, but the locally guaranteed output for this profile is the captured aggregate decoder.

Construct A using this same decoder and all selected writable contributors; group disjoint receiver blocks only before a mixing decoder couples their outputs. With the four-output average, writing all selected output contributors jointly is straightforward at small scale. Contributions from other fixed edges are subtracted from b. Do not use separate receiver step proofs after a decoder creates cross-block coupling unless the resulting objective is explicitly block-separable.

Targets of scale 0.1 are a cautious engineering starting point. Reachability checks determine whether selected messages and bounds can fit them. Poor rank does not license supplying hidden full correct targets to internal nodes. Learned local teachers are deferred.

## Training and batch state

Use per-lifetime S/Pa/Pu; identical theta may be shared in an offline training minibatch. A meta-learned S initialization may be broadcast via graph-preserving clones into independent lifetimes. No online state averaging across batch elements. Theta is optimized outside the lifetime unroll; a deployed lifetime freezes it.

Initial smoke can use fixed theta and a known feasible overfit diagnostic, but that does not pass H1. M3 trains the encoder/router on task-grounded full-path episodes and evaluates new generated maps. Keep the graph decoder contract and shadow-state isolation intact.

## Slow fitting

The initial slow fitter can use full-graph differentiable replay with residuals disabled, fixed theta/G, a declared optimizer and projected S bound. One proposed pilot choice is Adam with learning rate 0.001, at most 32 steps per sleep, and the replay cap in the config. This nonconvex procedure has no guaranteed global convergence. Charge all replay and optimizer work; offer the same budget to replay-only and delta-sleep baselines.

## Initialization and code ownership

Start S small rather than saturating all synapses. The dedicated saturated-edge test is mandatory regardless of initialization. Fixed descriptors and node-specific bounded drives prevent completely identical initial hidden states. Log norms and rank before and after training. Do not claim output rank merely because descriptors exist.

The reference module examples are unbatched local checks. Codex must implement its own batch-isolated functional operators and test them against those examples. The package does not include a model implementation disguised as a research result.
