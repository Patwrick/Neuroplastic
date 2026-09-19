# Implementation contracts

Pytest discovery is limited to this directory. Tests use the unchanged NumPy
oracle for parity and central finite differences for conditional outer gradients.
Graph, feedback, sleep rejection and cold-recall tests exercise actual code paths.
Use Python `-B` so oracle imports create no unexpected release files. The release
verification script runs separately; local tests do not establish H1/H2/H3.
