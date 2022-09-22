"""The module 'pyleida.plotting' provides functions to generate
plots and visual representations."""

from ._plotting import (
    brain_states_network,
    brain_states_nodes,
    brain_states_on_surf,
    brain_states_on_surf2,
    states_k_glass,
    brain_dynamics_gif,
    matrices_gif, 
    plot_static_fc_matrices,
    states_in_bold,
    states_in_bold_gif,
    plot_pyramid,
    _explore_state,
    _save_html
)

__all__ = [
    "brain_states_network",
    "brain_states_nodes",
    "brain_states_on_surf",
    "brain_states_on_surf2",
    "states_k_glass",
    "brain_dynamics_gif",
    "matrices_gif",
    "plot_static_fc_matrices",
    "states_in_bold",
    "states_in_bold_gif",
    "plot_pyramid",
    "_explore_state",
    "_save_html"
]