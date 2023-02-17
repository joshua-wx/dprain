import os
from glob import glob
import warnings
import argparse
import traceback
from datetime import datetime, timedelta

from concurrent.futures import TimeoutError
from pebble import ProcessPool, ProcessExpired

import matplotlib
matplotlib.use('agg')

import numpy as np
import pyart
from matplotlib import pyplot as plt
import cftime

import openradartools as ort

import radar_codes
import attenuation
import rainrate
import file_util

import dask
import dask.bag as db
import gc

def daterange(date1, date2):
    """
    Generate date list between dates
    """
    date_list = []
    for n in range(int ((date2 - date1).days)+1):
        date_list.append(date1 + timedelta(n))
    return date_list

def buffer(vol_ffn):
    try:
        torrentfields(vol_ffn)
        gc.collect()
    except Exception as e:
        print('failed on', vol_ffn,'with',e)
        
#CUSTOM OPENRADARTOOLS LIBS

def do_gatefilter(radar, gf=None, refl_name='DBZ', phidp_name="PHIDP", rhohv_name='RHOHV_CORR', zdr_name="ZDR", despeckle_field=False):
    """
    Basic filtering function for dual-polarisation data.

    Parameters:
    ===========
        radar:
            Py-ART radar structure.
        gatefilter:
            Py-ART gatefilter object.
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
    if gf is None:
        gf = pyart.correct.GateFilter(radar)
    if despeckle_field:
        # Despeckle
        gf = pyart.correct.despeckle_field(radar, refl_name, gatefilter=gf)

    # Remove obviously wrong data.
    gf.exclude_outside(zdr_name, -4, 4)
    gf.exclude_outside(refl_name, 10, 80.0)
    gf.exclude_below(rhohv_name, 0.98)

    return gf

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
    median, std = _det_sys_phase_gf(phidp, first_ray_idx, last_ray_idx, is_meteo)
    return median, std

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
        return None, None
    return np.median(phases), np.std(phases)
        
###########################################################    

def torrentfields(vol_ffn):

    print('processing', vol_ffn)

    #read radar volume
    radar = pyart.aux_io.read_odim_h5(vol_ffn, file_field_names=True)
    #index index of lowest sweep
    sort_idx = np.argsort(radar.fixed_angle['data'])
    #get time from middle of first sweep
    start_ray_idx = radar.get_start(sort_idx[0])
    end_ray_idx   = radar.get_start(sort_idx[1])
    start_time = cftime.num2pydate(radar.time['data'][start_ray_idx], radar.time['units'])
    end_time = cftime.num2pydate(radar.time['data'][end_ray_idx], radar.time['units'])
    valid_time = start_time + (end_time-start_time)/2
    valid_time = ort.basic.round_to_nearest_minute(valid_time)
    date_str = valid_time.strftime('%Y%m%d')
    
    #get radar band
    wavelength = ort.file.get_wavelength(vol_ffn)
    if wavelength<8:
        band = 'C'
    else:
        band = 'S'
    if VERBOSE:
        print('band', band)
        
        
    #build output filenames
    cf_path = f'{cf_root}/{RADAR_ID}/{date_str}'
    cf_fn = f'{RADAR_ID}_{valid_time.strftime("%Y%m%d_%H%M")}00.vol.nc' #this filename should match
    cf_ffn = f'{cf_path}/{cf_fn}'
    rf_path = f'{rf_root}/{RADAR_ID}/{date_str}'
    rf_fn = f'{RADAR_ID}_{valid_time.strftime("%Y%m%d_%H%M")}00.prcp-rrate.nc' #this filename should match
    rf_ffn = f'{rf_path}/{rf_fn}'
    img_path = f'{img_root}/{RADAR_ID}/{date_str}'
    img_fn = f'{RADAR_ID}_{valid_time.strftime("%Y%m%d_%H%M")}00.jpg' #this filename should match
    img_ffn = f'{img_path}/{img_fn}'
    #check if last file to be created (img_ffn) and SKIP_EXISTING are true
    if os.path.isfile(rf_ffn) and SKIP_EXISTING:
        print('skipping:', vol_ffn, 'already processed and skipping enabled')
        return None
    ##################################################################################################
    #
    # Preprocessing
    #
    ##################################################################################################
    if VERBOSE2:
        print('Corrections')
    # Correct RHOHV
    rho_corr = ort.dp.correct_rhohv(radar, snr_name='SNRH')
    radar.add_field_like('RHOHV', 'RHOHV_CORR', rho_corr, replace_existing=True)

    # Correct ZDR
    corr_zdr = ort.dp.correct_zdr(radar, snr_name='SNRH')
    radar.add_field_like('ZDR', 'ZDR_CORR', corr_zdr, replace_existing=True)

#     from importlib import reload
#     reload(radar_codes)
    
    if VERBOSE2:
        print('Temperature Profile')
    # Temperature    
    height, temperature, isom, profiles, levels = ort.nwp.nwp_profile(radar, source='access')
    radar.add_field('temperature', temperature, replace_existing=True)
    radar.add_field('height', height, replace_existing=True)
    radar.add_field('height_over_isom', isom, replace_existing=True)

    if VERBOSE2:
        print('Gatefilter')
    # GateFilter
    gatefilter = do_gatefilter(radar,
                             refl_name='DBZH_CLEAN',
                             phidp_name="PHIDP",
                             rhohv_name='RHOHV_CORR',
                             zdr_name="ZDR_CORR")
    
    if VERBOSE2:
        print('PHIDP processing')
    # phidp filtering
    rawphi = radar.fields["PHIDP"]['data']
    sysphase_gatefilter = gatefilter.copy()
    sysphase_gatefilter.exclude_below('DBZH_CLEAN', 5)
    sysphase_gatefilter.exclude_below('RHOHV_CORR', 0.95)
    
    sysphase, sysphase_std = det_sys_phase_gf(radar, sysphase_gatefilter, phidp_field="PHIDP", sweep=sort_idx[1])
    if sysphase is None:
        phase_offset_name = "PHIDP"
        if VERBOSE:
            print('WARNING: system phase not found')
        #add metadata on system phase
        kdp_field_name = 'KDP'
        phidp_field_name = 'PHIDP'
        radar.fields[phidp_field_name]['system_phase'] = -999
        radar.fields[phidp_field_name]['system_phase_std'] = -999
    else:
        radar.add_field_like("PHIDP", "PHIDP_offset", rawphi - sysphase, replace_existing=True)
        phase_offset_name = "PHIDP_offset"
        if VERBOSE:
            print('system phase:', round(sysphase))
        #correct phase
        #calculate phidp from bringi technique
        phidp_b, kdp_b = ort.dp.phidp_bringi(radar, gatefilter, phidp_field=phase_offset_name, refl_field='DBZH_CLEAN')
        radar.add_field("PHIDP_B", phidp_b, replace_existing=True)
        radar.add_field('KDP_B', kdp_b, replace_existing=True)
        #add metadata on system phase
        kdp_field_name = 'KDP_B'
        phidp_field_name = 'PHIDP_B'
        #add metadata on system phase
        radar.fields[phidp_field_name]['system_phase'] = sysphase
        radar.fields[phidp_field_name]['system_phase_std'] = sysphase_std
    
    
    if VERBOSE2:
        print('HCA processing')
    
    #first try to use exisiting NCAR HCA into a field. Sometimes it is missing due a missing DP fields.
    try:
        hca_field = ort.dp.insert_ncar_pid(radar, vol_ffn, dbz_name='DBZH_CLEAN')
    except:
        print('failed extracting ncar pid')
        #insert CSU HCA if NCAR PID is not in the file
        hca_field = ort.dp.csu_hca(radar,
                                      gatefilter,
                                      kdp_name=kdp_field_name,
                                      zdr_name='ZDR_CORR',
                                      rhohv_name='RHOHV_CORR',
                                      refl_name='DBZH_CLEAN',
                                      band=band)
    radar.add_field('radar_echo_classification', hca_field, replace_existing=True)
    
    ##################################################################################################
    #
    # Retrievals
    #
    ##################################################################################################
#     from importlib import reload
#     reload(rainrate)
#     reload(attenuation)
    #single pol rainfall
    radar = rainrate.conventional(radar, alpha=92, beta=1.7, refl_field='DBZH_CLEAN')

    if radar.fields[phidp_field_name]['system_phase'] != -999:
        if VERBOSE2:
            print('QPE Estimate')
        #estimate alpha
        alpha, alpha_method = attenuation.estimate_alpha_zhang2020(radar, band, sort_idx[1],
                                               refl_field='DBZH_CLEAN', zdr_field='ZDR_CORR', rhohv_field='RHOHV_CORR',
                                               verbose=VERBOSE)
        if VERBOSE2:
            print('QPE ZPHI')
        #estimate specific attenuation
        radar = attenuation.retrieve_zphi(radar, band, alpha=alpha, alpha_method=alpha_method,
                                         refl_field='DBZH_CLEAN', phidp_field=phidp_field_name, rhohv_field='RHOHV_CORR')
        if VERBOSE2:
            print('QPE Retrieve')
        if RADAR_ID == 2:
            ah_coeff_fitted = False #use default for Melbourne
        else:
            ah_coeff_fitted = True #otherwise use fits for Darwin
        #estimate rainfall
        radar = rainrate.polarimetric(radar, band, refl_field='corrected_reflectivity',
                                      kdp_field=kdp_field_name, phidp_field=phidp_field_name, rhohv_field='RHOHV_CORR',
                                      ah_coeff_fitted=ah_coeff_fitted)
        reflectivity_field_name = 'corrected_reflectivity'
    else:
        #insert nan field when phase is not corrected
        dummy_field = np.empty_like(radar.fields['zr_rainrate']['data'])
        dummy_field[:] = np.nan
        radar.add_field_like('zr_rainrate', 'hybrid_rainrate', dummy_field, replace_existing=True)
        radar.add_field_like('zr_rainrate', 'ah_rainrate', dummy_field, replace_existing=True)
        radar.add_field_like('zr_rainrate', 'kdp_rainrate', dummy_field, replace_existing=True)
        reflectivity_field_name = 'DBZH_CLEAN'
        alpha_method = 'None'
        alpha = 0
    ##################################################################################################
    #
    # Create and write grid
    #
    ##################################################################################################
    
    # grid first two sweeps (second sweep used as a fallback where the lower grid has no data)
    dp_rainrate = radar.get_field(sort_idx[retrieval_sweep], 'hybrid_rainrate', copy=True).filled(np.nan)
    cor_refl = radar.get_field(sort_idx[retrieval_sweep], reflectivity_field_name, copy=True).filled(np.nan)
    kdp_out = radar.get_field(sort_idx[retrieval_sweep], kdp_field_name, copy=True).filled(np.nan)
    zdr_out = radar.get_field(sort_idx[retrieval_sweep], 'ZDR_CORR', copy=True).filled(np.nan)
    
    input_fields = {'rainrate':dp_rainrate, 'reflectivity':cor_refl, 'kdp':kdp_out, 'zdr': zdr_out}
    
    #build metadata and grid
    r = radar.range['data']
    th = 450 - radar.get_azimuth(sort_idx[0], copy=False)
    th[th < 0] += 360
    R, A = np.meshgrid(r, th)
    x = R * np.cos(np.pi * A / 180)
    y = R * np.sin(np.pi * A / 180)
    xgrid = np.linspace(-127750,127750,512)
    xgrid, ygrid = np.meshgrid(xgrid, xgrid)
    output_fields = ort.gridding.KDtree_interp(input_fields, x, y, xgrid, ygrid, nnearest = 50, maxdist = 2500)
    rain_grid_2d = output_fields['rainrate']
    reflectivity_2d = output_fields['reflectivity']
    kdp_2d = output_fields['kdp']
    zdr_2d = output_fields['zdr']
    mask = reflectivity_2d<=0
    reflectivity_2d[mask] = np.nan
    rain_grid_2d[mask] = np.nan
    kdp_2d[mask] = np.nan
    zdr_2d[mask] = np.nan
        
    #extract metadata for RF3 grids
    standard_lat_1, standard_lat_2 = file_util.rf_standard_parallel_lookup(RADAR_ID)
    if standard_lat_1 is None:
        print('failed to lookup standard parallels')
    rf_lon0, rf_lat0 = file_util.rf_grid_centre_lookup(RADAR_ID)
    if rf_lon0 is None:
        print('failed to lookup rf grid centre coordinates')
    
    #create paths
    if not os.path.exists(rf_path):
        os.makedirs(rf_path)
    if os.path.exists(rf_ffn):
        print('rf3 of same name found, removing')
        os.system('rm -f ' + rf_ffn)
    #write to nc
    file_util.write_rf_nc(rf_ffn, RADAR_ID, valid_time.timestamp(), rain_grid_2d, reflectivity_2d, kdp_2d, zdr_2d, rf_lon0, rf_lat0, (standard_lat_1, standard_lat_2))
    
    #create image and save to file
    ###################################################################################################################
    tilt = sort_idx[0] #first tilt
    ylim = [-128, 128]
    xlim = [-128, 128]

    fig = plt.figure(figsize=[16,8])
    display = pyart.graph.RadarDisplay(radar)

    ax = plt.subplot(231)
    display.plot_ppi(reflectivity_field_name, tilt, vmin=0, vmax=60, cmap='pyart_HomeyerRainbow')
    ax.set_xlabel('')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    ax = plt.subplot(232)
    display.plot_ppi(kdp_field_name, tilt, vmin=0, vmax=6, cmap='pyart_HomeyerRainbow')
    ax.set_xlabel('')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    ax = plt.subplot(233)
    display.plot_ppi(phidp_field_name, tilt, vmin=0, vmax=90, cmap='pyart_Wild25')
    ax.set_xlabel('')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    ax = plt.subplot(234)
    display.plot_ppi('zr_rainrate', tilt, vmin=0.2, vmax=75, cmap='pyart_RRate11', title='SP retrieval')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    ax = plt.subplot(235)
    display.plot_ppi('hybrid_rainrate', tilt, vmin=0.2, vmax=75, cmap='pyart_RRate11', title=f'DP retrieval method: {alpha_method} with alpha: {alpha:.3f}')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    ax = plt.subplot(236)
    ax.set_title(f'RF3 grid of DP retrieval')
    rain_grid_2d_plotting = rain_grid_2d.copy()
    rain_grid_2d_plotting[rain_grid_2d_plotting<=0] = np.nan
    img = plt.imshow(np.flipud(rain_grid_2d_plotting), vmin=0.2, vmax=75, cmap=pyart.graph.cm._generate_cmap('RRate11',100))
    cbar = fig.colorbar(img, ax=ax)
    cbar.set_label('mm/hr')
    
    if not os.path.exists(img_path):
        os.makedirs(img_path)
    if os.path.exists(img_ffn):
        print('image of same name found, removing')
        os.system('rm -f ' + img_ffn)
    plt.savefig(img_ffn, dpi=100)
    fig.clf()
    plt.close()
    
    ##################################################################################################
    #
    # Write outputs CF Radial
    #
    ##################################################################################################
    
    #update least sig digit
    for fieldname in radar.fields.keys():
        radar.fields[fieldname]['_Least_significant_digit'] = 2
        radar.fields[fieldname]['_Least_significant_digit'] = 2
    
    #remove uneeded fields
    goodkeys = [kdp_field_name, phidp_field_name, reflectivity_field_name, 'DBZH_CLEAN', 'specific_attenuation', 'ZDR_CORR','hybrid_rainrate','kdp_rainrate','ah_rainrate','zr_rainrate','radar_echo_classification','RHOHV_CORR']
    
    unwanted_keys = []
    for mykey in radar.fields.keys():
        if mykey not in goodkeys:
            unwanted_keys.append(mykey)
    for mykey in unwanted_keys:
        radar.fields.pop(mykey)
        
    #write to cf output
    if VERBOSE2:
        print('Save CFradial')
    #create paths
    if not os.path.exists(cf_path):
        os.makedirs(cf_path)
    if os.path.exists(cf_ffn):
        print('cfradial of same name found, removing')
        os.system('rm -f ' + cf_ffn)
    #write to cf
    pyart.io.write_cfradial(cf_ffn, radar)
    
    ##################################################################################################################
    
    #clean up
    del radar
    del rain_grid_2d
    
def manager(date_str):

    #unpack and list daily zip
    vol_zip = f'{vol_root}/{RADAR_ID}/{date_str[0:4]}/vol/{RADAR_ID}_{date_str}.pvol.zip'
    if not os.path.exists(vol_zip):
        print('archive not found for date:', date_str)
        return None
    temppath = ort.file.unpack_zip(vol_zip)
    vol_ffn_list = sorted(glob(temppath + '/*.h5'))

#     timeout = 300
#     for arg_slice in ort.basic.chunks(vol_ffn_list, NCPU):
#         with ProcessPool() as pool:
#             future = pool.map(buffer, arg_slice, timeout=timeout)
#             iterator = future.result()
#             while True:
#                 try:
#                     _ = next(iterator)
#                 except StopIteration:
#                     break
#                 except TimeoutError as error:
#                     print("function took longer than %d seconds" % timeout, 'for', arg_slice)
#                 except ProcessExpired as error:
#                     print("%s. Exit code: %d" % (error, error.exitcode))
#                 except TypeError as error:
#                     print("%s. Exit code: %d" % (error, error.exitcode))
#                 except Exception:
#                     traceback.print_exc()
                
            
    
    import time    
    for vol_ffn in vol_ffn_list:
        #start = time.time()
        #try:
        torrentfields(vol_ffn)
#         except Exception as e:
#             print('')
#             print('FAILED on', vol_ffn, 'with', e)
#             print('')
#         end = time.time()
#         print('timer', end - start)
#         print('')
#         print('')
#         print('')

#     #run retrieval
#     i            = 0
#     n_files      = len(vol_ffn_list)   
#     for flist_chunk in ort.basic.chunks(vol_ffn_list, NCPU): #CUSTOM RANGE USED
#         bag = db.from_sequence(flist_chunk).map(buffer)
#         _ = bag.compute()
#         i += NCPU
#         del bag
#         print('processed: ' + str(round(i/n_files*100,2)))
        
    #clean up
    temp_vol_dir = os.path.dirname(vol_ffn_list[0])
    if '/tmp' in temp_vol_dir:
        os.system('rm -rf ' + temp_vol_dir)
    return None
        
def main():
    
    #build list of dates for manager
    dt_list = daterange(DT1, DT2)
    
    for dt in dt_list:
        date_str = dt.strftime('%Y%m%d')
        manager(date_str)
    
    
if __name__ == '__main__':
    """
    Global vars
    """    
    #config
    vol_root = '/g/data/rq0/level_1/odim_pvol'
    cf_root = '/scratch/kl02/jss548/dprain/cfradial'
    rf_root = '/scratch/kl02/jss548/dprain/rfgrid'
    img_root = '/scratch/kl02/jss548/dprain/img'
    VERBOSE = False
    VERBOSE2 = False #more info
    SKIP_EXISTING = False
    retrieval_sweep = 1 #second lowest
    
    # Parse arguments
    parser_description = "DP rainfall retrieval"
    parser = argparse.ArgumentParser(description = parser_description)
    parser.add_argument(
        '-j',
        '--cpu',
        dest='ncpu',
        default=16,
        type=int,
        help='Number of process')
    parser.add_argument(
        '-d1',
        '--date1',
        dest='date1',
        default=None,
        type=str,
        help='starting date to process from archive',
        required=True)
    parser.add_argument(
        '-d2',
        '--date2',
        dest='date2',
        default=None,
        type=str,
        help='starting date to process from archive',
        required=True)
    parser.add_argument(
        '-r',
        '--rid',
        dest='rid',
        default=None,
        type=int,
        help='Radar ID',
        required=True)
    
    args = parser.parse_args()
    NCPU         = args.ncpu
    RADAR_ID     = args.rid
    DATE1_STR    = args.date1
    DATE2_STR    = args.date2
    DT1          = datetime.strptime(DATE1_STR,'%Y%m%d')
    DT2          = datetime.strptime(DATE2_STR,'%Y%m%d')
    
    with warnings.catch_warnings():
        # Just ignoring warning messages.
        warnings.simplefilter("ignore")
        main()
        
    #%run op-production.py -r 2 -d1 20220202 -d2 20220202 -j 8