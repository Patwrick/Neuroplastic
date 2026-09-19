# Release decisions and open choices

| ID | Decision | Status / authority |
|---|---|---|
| D01 | Four endpoints plus residual absolute-sum budget; asymmetric exact box | Accepted v0.4.1 mathematics |
| D02 | Retain continuous outer write paths; branch-conditional derivative only for hard routing | Accepted v0.4.1 differentiation contract |
| D03 | Shared theta frozen in deployed lifetime; CPU/FP32 correctness start | Accepted reference scope |
| D04 | Synchronous reset-boundary sleep, both residuals off for cold recall | Accepted reference scope |
| D05 | Exact small routing with protected output-delivery slot and fixed output decoder | Proposed initial engineering profile; document any change |
| D06 | Simple delta memory, including slow/replay variant, is a primary comparator | Accepted experimental requirement |
| D07 | Stable addressing versus state-dependent addressing | Unresolved empirical comparison, not a claimed winner |
| D08 | Target scale, message rank, write budget and contraction gain | Pilot engineering choices; diagnose before tuning |
| D09 | Confirmation effect thresholds, sample size and resource limits | Unresolved until preregistration after pilot evidence |
| D10 | Rewiring, INT4, offloading, asynchronous sleep, online encoders | Deferred; not first-task prerequisites |
| D11 | Repository integration path and local dependency environment | Codex resolves after inspecting actual checkout; preserve existing work |

Scientific changes require a report-back issue and explicit acceptance before the authoritative paper is revised. Safe engineering choices within the specification can proceed after a decision record. Do not ask the user to choose a tensor layout or test filename when the contract already determines a safe default.
