# Sampling-scope clarification for the locked reviewer protocol

2026-10-23. This clarifies terminology without changing any samples, settings,
algorithms, candidate families or statistical allocations in the locked protocol.
"Independent calibration images" in its data-access description means held-out
calibration images excluded from fitting/tuning/support/test pools. The identity
and content checks establish separation, not stochastic independence.

The population theorem separately assumes independent correctness pairs with
the prescribed class-conditional target laws. It is not a conditional guarantee
for arbitrary deduplicated finite pools, dependent images, or unmatched shifted
calibration/test distributions. All guarantees are per prespecified experimental
configuration, not simultaneous over the grid or for a post-hoc chosen winner.
