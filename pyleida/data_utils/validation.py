"""Utilities for input validation"""

import numpy as np

def _check_state(k,state):
    """
    Check if the selected state is present
    for the select k partition.

    Params:
    -------
    k : value provided by the user.

    state : value provided by the user.
    """
    if not isinstance(state,int):
        raise TypeError("'state' must be an integer.")
    else:
        if state>k:
            raise ValueError("The selected phase-locking 'state' must be "
                            "present for the selected 'k' partition!")

def _check_k_input(kmin,kmax,k):
    """
    Check if the provided 'k' is valid
    (i.e., if a model for that k was fitted).
    All the class methods that have 'k' as 
    argument are validated with this method.
    """
    if not isinstance(k,int):
        raise TypeError("'k' must be a integer!")
    else:
        if k not in np.arange(kmin,kmax+1):
            raise ValueError(f"'k' must be a fitted model: you selected 'k'= {k}, "
                            f"but K-Means models were fitted for the {kmin} - {kmax} range.")

def _check_metric(metric):
    """
    Validate whether the provided
    metric by the user is available.

    Params:
    -------
    metric : value provided by the user.
    """
    if not isinstance(metric,str):
        raise TypeError("'metric' must be a string!")
    else:
        if metric not in ['occupancies','dwell_times']:
            raise ValueError("'metric' must be either 'occupancies' or 'dwell_times'.")

def _check_isint(dct):
    """
    Validate if each value in
    dictionary is an integer.

    Params:
    -------
    dct : dict
        Keys are parameter's names,
        and values contain the values
        provided by the user for each
        parameter.
    """
    for param in dct.keys():
        if not isinstance(dct[param],int):
            raise TypeError(f"'{param}' must be an integer")
    

