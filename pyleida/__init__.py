"""
Leading Eigenvector Dynamics Analysis (LEiDA) in Python
------------------------------------------------------------------------
Documentation is available in the docstrings and online at the site of the repository
https://github.com/PSYCHOMARK/leida-python

Contents
--------
pyleida is a Python toolbox to apply the Leading Eigenvector Dynamics Analysis (LEiDA)
framework to functional MRI data. It contains all the necessary tools to apply the framework
from the beggining to the end of the pipeline or workflow, save results, generate reports, and
figures.

Reference: Cabral et al. (2017). Cognitive performance in healthy older adults
relates to spontaneous switching between states of functional connectivity during
rest. Scientific reports, 7(1), 5135. https://doi.org/10.1038/s41598-017-05425-7

Modules
---------
clustering              --- Tools to identify the phase-locking states by means of unsupervised learning techniques (k-means).
data_loader             --- Class to load, manage, and explore further the results obtained with the Leida class.
data_utils              --- Functions to transform and handle data.
dynamics_metrics        --- Functions to compute the metrics from Dynamical Systems theory.
leida                   --- Class to run the LEiDA pipeline.
plotting                --- Plotting tools.
signal_tools            --- Utilities for BOLD signals.
"""
__pdoc__ = {}
__pdoc__["_leida"] = True
__pdoc__["_data_loader"] = True

from ._leida import Leida
from ._data_loader import DataLoader