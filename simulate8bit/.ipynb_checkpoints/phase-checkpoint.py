import os
import gzip
import copy
import pickle

import cftime
import pandas as pd
import pyart
import scipy
import numpy as np

from scipy import integrate
from csu_radartools import csu_kdp

def texture(data):
    """
    Compute the texture of data.
    Compute the texture of the data by comparing values with a 3x3 neighborhood
    (based on :cite:`Gourley2007`). NaN values in the original array have
    NaN textures. (Wradlib function)

    Parameters:
    ==========
    data : :class:`numpy:numpy.ndarray`
        multi-dimensional array with shape (..., number of beams, number
        of range bins)

    Returns:
    =======
    texture : :class:`numpy:numpy.ndarray`
        array of textures with the same shape as data
    """
    x1 = np.roll(data, 1, -2)  # center:2
    x2 = np.roll(data, 1, -1)  # 4
    x3 = np.roll(data, -1, -2)  # 8
    x4 = np.roll(data, -1, -1)  # 6
    x5 = np.roll(x1, 1, -1)  # 1
    x6 = np.roll(x4, 1, -2)  # 3
    x7 = np.roll(x3, -1, -1)  # 9
    x8 = np.roll(x2, -1, -2)  # 7

    # at least one NaN would give a sum of NaN
    xa = np.array([x1, x2, x3, x4, x5, x6, x7, x8])

    # get count of valid neighboring pixels
    xa_valid = np.ones(np.shape(xa))
    xa_valid[np.isnan(xa)] = 0
    # count number of valid neighbors
    xa_valid_count = np.sum(xa_valid, axis=0)

    num = np.zeros(data.shape)
    for xarr in xa:
        diff = data - xarr
        # difference of NaNs will be converted to zero
        # (to not affect the summation)
        diff[np.isnan(diff)] = 0
        # only those with valid values are considered in the summation
        num += diff ** 2

    # reinforce that NaN values should have NaN textures
    num[np.isnan(data)] = np.nan

    return np.sqrt(num / xa_valid_count)

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
    # Initialize gatefilter
    gf = pyart.filters.GateFilter(radar)

    # Remove obviously wrong data.
    gf.exclude_outside(zdr_name, -6.0, 7.0)
    gf.exclude_outside(refl_name, -20.0, 80.0)

    # Compute texture of PHIDP and remove noise.
    dphi = texture(radar.fields[phidp_name]['data'])
    radar.add_field_like(phidp_name, 'PHITXT', dphi)
    gf.exclude_above('PHITXT', 20)
    gf.exclude_below(rhohv_name, 0.6)

    # Despeckle
    gf_despeckeld = pyart.correct.despeckle_field(radar, refl_name, gatefilter=gf)

    try:
        # Remove PHIDP texture
        radar.fields.pop('PHITXT')
    except Exception:
        pass

    return gf_despeckeld

def _fix_phidp_from_kdp(phidp, kdp, r, gatefilter):
    """
    Correct PHIDP and KDP from spider webs.

    Parameters
    ==========
    r:
        Radar range.
    gatefilter:
        Gate filter.
    kdp_name: str
        Differential phase key name.
    phidp_name: str
        Differential phase key name.

    Returns:
    ========
    phidp: ndarray
        Differential phase array.
    """
    kdp[gatefilter.gate_excluded] = 0
    kdp[(kdp < -4)] = 0
    kdp[kdp > 15] = 0
    interg = integrate.cumtrapz(kdp, r, axis=1)

    phidp[:, :-1] = interg / (len(r))
    return phidp, kdp


def phidp_giangrande_8bit(radar, gatefilter, refl_field='DBZ', ncp_field='NCP',
                     rhv_field='RHOHV_CORR', phidp_field='PHIDP'):
    """
    Phase processing using the LP method in Py-ART. A LP solver is required,

    Parameters:
    ===========
    radar:
        Py-ART radar structure.
    gatefilter:
        Gate filter.
    refl_field: str
        Reflectivity field label.
    ncp_field: str
        Normalised coherent power field label.
    rhv_field: str
        Cross correlation ration field label.
    phidp_field: str
        Differential phase label.

    Returns:
    ========
    phidp_gg: dict
        Field dictionary containing processed differential phase shifts.
    kdp_gg: dict
        Field dictionary containing recalculated differential phases.
    """
    
    #currently in use by RF
    phidp_16_data = copy.deepcopy(radar.fields[phidp_field]['data'])
    gain_8bit = 1.417323
    offset_8bit = np.min(phidp_16_data)
    
    #convert to 8bit
    phidp_8_data_uint = ((phidp_16_data-offset_8bit)/gain_8bit).astype(np.uint8)
    phidp_8_data = (phidp_8_data_uint.astype(float)*gain_8bit)+offset_8bit
    radar.add_field_like(phidp_field, 'PHITMP_8', phidp_8_data)
    
    unfphidic = pyart.correct.dealias_unwrap_phase(radar,
                                                   gatefilter=gatefilter,
                                                   skip_checks=True,
                                                   vel_field='PHITMP_8',
                                                   nyquist_vel=90)

    radar.add_field_like(phidp_field, 'PHITMP_8_unwrapped', unfphidic['data'])


    
    #calc phase
    phidp_gg, kdp_gg = pyart.correct.phase_proc_lp(radar, 0.0,
                                                   LP_solver='cylp',
                                                   ncp_field=ncp_field,
                                                   refl_field=refl_field,
                                                   rhv_field=rhv_field,
                                                   phidp_field='PHITMP_8_unwrapped')

    phidp_gg['data'], kdp_gg['data'] = _fix_phidp_from_kdp(phidp_gg['data'],
                                                           kdp_gg['data'],
                                                           radar.range['data'],
                                                           gatefilter)

    try:
        # Remove temp variables.
        radar.fields.pop('unfolded_differential_phase')
        radar.fields.pop('PHITMP_8')
        radar.fields.pop('PHITMP_8_unwrapped')
    except Exception:
        pass

    phidp_gg['data'] = phidp_gg['data'].astype(np.float32)
    phidp_gg['_Least_significant_digit'] = 4
    kdp_gg['data'] = kdp_gg['data'].astype(np.float32)
    kdp_gg['_Least_significant_digit'] = 4

    return phidp_gg, kdp_gg