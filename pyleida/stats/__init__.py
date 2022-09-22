"""The module 'pyleida.stats' provides functions
to execute statistical analyses on the dynamical
system theory metrics."""

from ._stats import (
    ks_distance,
    _compute_stats,
    scatter_pvalues,
    permtest_ind,
    permtest_rel,
    hedges_g
)

__all__ = [
    "ks_distance",
    "_compute_stats",
    "scatter_pvalues",
    "permtest_ind",
    "permtest_rel",
    "hedges_g"
]