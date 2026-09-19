# M0–M2 correctness reference

PyTorch state, routing, recurrent graph, captured snapshots, projected writes,
causal feedback and synchronous slow consolidation implement the v0.4.1 contract.
The T1 task and delta comparator are separated from the learning core. Runtime
math does not call the NumPy reference; SciPy is an offline diagnostic only.
See root README and REPORT_BACK for executed commands and bounded claims.
