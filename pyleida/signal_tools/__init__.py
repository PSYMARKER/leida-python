"""The module 'pyleida.signal_tools' provides functions
to compute relevant information from BOLD time series."""

from ._signal_tools import (
    hilbert_phase,
    clean_signals,
    phase_coherence,
    get_eigenvectors,
    txt_matrix
)

__all__ = [
    "hilbert_phase",
    "clean_signals",
    "phase_coherence",
    "get_eigenvectors",
    "txt_matrix"
]