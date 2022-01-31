"""
Codes for creating and manipulating gate filters. New functions: use of trained
Gaussian Mixture Models to remove noise and clutter from CPOL data before 2009.

@title: filtering
@author: Valentin Louf <valentin.louf@monash.edu>
@institutions: Monash University and the Australian Bureau of Meteorology
@date: 20/11/2017
@modification: 09/03/2020

.. autosummary::
    :toctree: generated/

    texture
    do_gatefilter_cpol
    do_gatefilter
"""
# Libraries
import os
import gzip
import pickle

import pyart
import cftime
import numpy as np
import pandas as pd


def do_gatefilter(radar, refl_name='DBZ', phidp_name="PHIDP", rhohv_name='RHOHV_CORR', zdr_name="ZDR", snr_name='SNR'):
    """
    Basic filtering function for dual-polarisation data.

    Parameters:
    ===========
        radar:
            Py-ART radar structure.
        refl_name: str
            Reflectivity field name.
        rhohv_name: str
            Cross correlation ratio field name.
        ncp_name: str
            Name of the normalized_coherent_power field.
        zdr_name: str
            Name of the differential_reflectivity field.

    Returns:
    ========
        gf_despeckeld: GateFilter
            Gate filter (excluding all bad data).
    """
    # Despeckle
    gf = pyart.correct.despeckle_field(radar, refl_name)

    # Remove obviously wrong data.
    gf.exclude_outside(zdr_name, -6.0, 7.0)
    gf.exclude_outside(refl_name, -20.0, 80.0)

    # Compute texture of PHIDP and remove noise.
    # GMM filter should be used here to remove GC
    gf.exclude_below(rhohv_name, 0.7)

    return gf
