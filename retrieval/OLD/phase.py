"""
Codes for correcting the differential phase and estimating KDP.

@title: phase
@author: Valentin Louf <valentin.louf@monash.edu>
@institutions: Monash University and the Australian Bureau of Meteorology
@date: 08/02/2020

.. autosummary::
    :toctree: generated/

    _fix_phidp_from_kdp
    phidp_bringi
    phidp_giangrande
"""
import pyart
import scipy
import numpy as np
from numba import jit
from matplotlib import pyplot as plt

from scipy import integrate
from csu_radartools import csu_kdp

def det_sys_phase_gf(radar, gatefilter, phidp_field=None, first_gate=30, sweep=0):
    """
    Determine the system phase.

    Parameters
    ----------
    radar : Radar
        Radar object for which to determine the system phase.
    gatefilter : Gatefilter
        Gatefilter object highlighting valid gates.
    phidp_field : str, optional
        Field name within the radar object which represents
        differential phase shift. A value of None will use the default
        field name as defined in the Py-ART configuration file.
    first_gate : int, optional
        Gate index for where to being applying the gatefilter.

    Returns
    -------
    sys_phase : float or None
        Estimate of the system phase. None is not estimate can be made.

    """
    # parse the field parameters
    if phidp_field is None:
        phidp_field = get_field_name('differential_phase')
    phidp = radar.fields[phidp_field]['data'][:, first_gate:]
    first_ray_idx = radar.sweep_start_ray_index['data'][sweep]
    last_ray_idx = radar.sweep_end_ray_index['data'][sweep]
    is_meteo = gatefilter.gate_included[:, first_gate:]
    return _det_sys_phase_gf(phidp, first_ray_idx, last_ray_idx, is_meteo)

def _det_sys_phase_gf(phidp, first_ray_idx, last_ray_idx, radar_meteo):
    """ Determine the system phase, see :py:func:`det_sys_phase`. """
    good = False
    phases = []
    for radial in range(first_ray_idx, last_ray_idx + 1):
        meteo = radar_meteo[radial, :]
        mpts = np.where(meteo)
        if len(mpts[0]) > 25:
            good = True
            msmth_phidp = pyart.correct.phase_proc.smooth_and_trim(phidp[radial, mpts[0]], 9)
            phases.append(msmth_phidp[0:25].min())
    if not good:
        return None
    return np.median(phases)

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

@jit
def fill_phi(phi):
    """
    Small function that propagates phidp values forward along rays to fill gaps
    """
    nx, ny = phi.shape
    for i in range(nx):
        phi_val = 0
        for j in range(ny):
            if np.isnan(phi[i, j]):
                phi[i, j] = phi_val
            else:
                phi_val = phi[i, j]
                
    return phi

def phidp_bringi(radar, gatefilter, phidp_field="PHI_UNF", refl_field='DBZ'):
    """
    Compute PHIDP and KDP Bringi.
    Parameters
    ==========
    radar:
        Py-ART radar data structure.
    gatefilter:
        Gate filter.
    unfold_phidp_name: str
        Differential phase key name.
    refl_field: str
        Reflectivity key name.
    Returns:
    ========
    phidpb: ndarray
        Bringi differential phase array.
    kdpb: ndarray
        Bringi specific differential phase array.
    """
    dz = radar.fields[refl_field]['data'].copy().filled(-9999)
    dp = radar.fields[phidp_field]['data'].copy().filled(-9999)

    # Extract dimensions
    rng = radar.range['data']
    azi = radar.azimuth['data']
    dgate = rng[1] - rng[0]
    [R, A] = np.meshgrid(rng, azi)

    # Compute KDP bringi.
    kdpb, phidpb, _ = csu_kdp.calc_kdp_bringi(dp, dz, R / 1e3, gs=dgate, bad=-9999, thsd=12, window=6.0, std_gate=11)

    # Mask array
    phidpb = np.ma.masked_where(phidpb == -9999, phidpb)
    kdpb = np.ma.masked_where(kdpb == -9999, kdpb)
    
    #fill
    phidpb = fill_phi(phidpb.filled(np.NaN))
    
    # Get metadata.
    phimeta = pyart.config.get_metadata("differential_phase")
    phimeta['data'] = phidpb
    kdpmeta = pyart.config.get_metadata("specific_differential_phase")
    kdpmeta['data'] = kdpb

    return phimeta, kdpmeta

def phidp_giangrande(radar, gatefilter, refl_field='DBZ', ncp_field='NCP',
                     rhv_field='RHOHV_CORR', phidp_field='PHIDP', VERBOSE=False):
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

    if VERBOSE:
        print('lp proc phase')
    phidp_gg, kdp_gg = pyart.correct.phase_proc_lp_gf(radar, gatefilter=gatefilter,
                                                   LP_solver='cylp',
                                                   refl_field=refl_field,
                                                   phidp_field=phidp_field)    
    
    if VERBOSE:
        print('fix phase from kdp')
    
    phidp_gg['data'], kdp_gg['data'] = _fix_phidp_from_kdp(phidp_gg['data'],
                                                           kdp_gg['data'],
                                                           radar.range['data'],
                                                           gatefilter)

    try:
        # Remove temp variables.
        radar.fields.pop('unfolded_differential_phase')
        radar.fields.pop('PHITMP')
    except Exception:
        pass

    phidp_gg['data'] = phidp_gg['data'].astype(np.float32)
    phidp_gg['_Least_significant_digit'] = 4
    kdp_gg['data'] = kdp_gg['data'].astype(np.float32)
    kdp_gg['_Least_significant_digit'] = 4

    return phidp_gg, kdp_gg
