import cftime
import numpy as np
from matplotlib import pyplot as plt

from skimage import morphology
from scipy.integrate import cumtrapz
from sklearn.linear_model import LinearRegression

import pyart

def _rolling_window(a, window):
    """ Create a rolling window object for application of functions
    eg: result=np.ma.std(array, 11), 1). """
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1], )
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def _smooth_masked(raw_data, wind_len=11, min_valid=6, wind_type='median'):
    """
    Smoothes the data using a rolling window.
    data with less than n valid points is masked.
    Parameters
    ----------
    raw_data : float masked array
        The data to smooth.
    win_len : float
        Length of the moving window.
    min_valid : float
        Minimum number of valid points for the smoothing to be valid.
    wind_type : str
        Type of window. Can be median or mean.
    Returns
    -------
    data_smooth : float masked array
        Smoothed data.
    """
    valid_wind = ['median', 'mean']
    if wind_type not in valid_wind:
        raise ValueError(
            "Window " + win_type + " is none of " + ' '.join(valid_wind))

    # we want an odd window
    if wind_len % 2 == 0:
        wind_len += 1
    half_wind = int((wind_len-1)/2)

    # initialize smoothed data
    nrays, nbins = np.shape(raw_data)
    data_smooth = np.ma.zeros((nrays, nbins))
    data_smooth[:] = np.ma.masked
    data_smooth.set_fill_value(np.nan)

    mask = np.ma.getmaskarray(raw_data)
    valid = np.logical_not(mask)

    mask_wind = _rolling_window(mask, wind_len)
    valid_wind = np.logical_not(mask_wind).astype(int)
    nvalid = np.sum(valid_wind, -1)

    data_wind = _rolling_window(raw_data, wind_len)

    # check which gates are valid
    ind_valid = np.logical_and(
        nvalid >= min_valid, valid[:, half_wind:-half_wind]).nonzero()

    data_smooth[ind_valid[0], ind_valid[1]+half_wind] = (
        eval('np.ma.' + wind_type + '(data_wind, axis=-1)')[ind_valid])

    return data_smooth

def _find_z_zdr_slope(alpha_dict):
    
    """
    WHAT: fits slope to Z-ZDR pairs stored in alpha direct object. Used for estimating alpha
    INPUT:
        alpha_dict: dictionary containing Z and ZDR pairs and alpha ts
    OUTPUT:
        slope coefficent (float)
    """
    
    plot_fits=False
    
    #setup bins
    z_bin_width = 2
    z_bin_centres = np.arange(20, 50+z_bin_width, z_bin_width)
    median_zdr = np.zeros_like(z_bin_centres, dtype=float)
    #run binning
    for i, centre in enumerate(z_bin_centres):
        bin_lower = centre - z_bin_width/2
        bin_upper = centre + z_bin_width/2
        bin_mask = np.logical_and(alpha_dict['z_pairs']>=bin_lower, alpha_dict['z_pairs']<bin_upper)
        median_zdr[i] = np.nanmedian(alpha_dict['zdr_pairs'][bin_mask])
        
    #remove nan entries
    nan_mask = np.isnan(median_zdr)
    median_zdr = median_zdr[~nan_mask]
    z_bin_centres = z_bin_centres[~nan_mask]
    
    #linear regression
    LS = LinearRegression()
    LS.fit(z_bin_centres.reshape(-1, 1), median_zdr) #, sample_weight=sample_weight)

    #calculate slope
    if plot_fits:
        plt.plot(alpha_dict['z_pairs'], alpha_dict['zdr_pairs'], 'k.', label='Pairs')
        plt.plot(z_bin_centres, median_zdr, 'r.', label='Median ZDR')
        plt.plot(z_bin_centres, LS.predict(z_bin_centres.reshape(-1, 1)), 'b-', label='LS fit')
        plt.legend()
        plt.xlabel('Z')
        plt.ylabel('ZDR')
    
    return LS.coef_[0]

def _slope_alpha_func(input_x, a, b, max_y):
    #function used for fitting dzdr/dz - alpha slope
    cutoff_x = 0.04 #defines through investiation of dzdr/dr-alpha relationship - appears to remains constant
    if input_x >= cutoff_x:
        #return default (0 slope) alpha value
        fit_y = a/cutoff_x + b
    else:
        #return model alpha - used for tropica rain
        fit_y = a/input_x + b
    #constrain alpha
    if fit_y > max_y:
        return max_y
    else:
        return fit_y

def _alpha_function_from_temperature(band, tempin):
    
    #define temperature for each set of coefficents
    temp_list = np.array([0,10,20])
    #define dzdr/dz value to use for calculating stratiform/tropical default
    strat_k = 0.0183 #Zhang2020 used alpha = 0.035 for stratiform default, which is equal to a K of 0.0183 in formula (6) of Zhang2020
    #find index for upper and lower bounds
    if tempin <= temp_list[0]:
        lower_idx = 0
        upper_idx = 1
    elif tempin >= temp_list[-1]:
        lower_idx = np.size(temp_list)-2
        upper_idx = np.size(temp_list)-1
    else:
        lower_idx = np.where(temp_list<tempin)[0][-1]
        upper_idx = np.where(temp_list>=tempin)[0][0]
    #define function coefficents
    if band == 'C':
        func_a = [0.001404, 0.000718, 0.000643]
        func_b = [0.08310, 0.07797, 0.05789]
#         func_a = [0.001448, 0.000988, 0.000765]
#         func_b = [0.06791, 0.05744, 0.04524]
        max_alpha = 0.2
    elif band == 'S':    
        func_a = [0.000851, 0.000623, 0.000485]
        func_b = [0.01320, 0.009870, 0.007516]
#         func_a = [0.000790, 0.000579, 0.000451]
#         func_b = [0.00991, 0.007217, 0.005449]
        max_alpha = 0.05
    else:
        print('band unknown')
        return None
    #find linear regression a
    m_a = (func_a[upper_idx]-func_a[lower_idx])/(temp_list[upper_idx]-temp_list[lower_idx])
    c_a = func_a[upper_idx]-m_a*temp_list[upper_idx]
    fit_a = tempin*m_a + c_a
    #find linear regression a
    m_b = (func_b[upper_idx]-func_b[lower_idx])/(temp_list[upper_idx]-temp_list[lower_idx])
    c_b = func_b[upper_idx]-m_b*temp_list[upper_idx]
    fit_b = tempin*m_b + c_b        
    
    #determine default alpha by inputting a high slope into the slope function (returns the default)
    default_conv_alpha = _slope_alpha_func(1, fit_a, fit_b, max_alpha)
    default_strat_alpha = _slope_alpha_func(strat_k, fit_a, fit_b, max_alpha)
    return fit_a, fit_b, default_conv_alpha, default_strat_alpha, max_alpha 

def _find_alpha_zhang2020(z_pairs, zdr_pairs, band, t_mean, verbose=False, plot_fits=False):
    
    """
    WHAT: Alpha estimation stragety from Zhang et al 2020 (5 cases).
    INPUT:
        z_pairs: reflectivity array
        zdr_pairs: zdr array
        radar band: either 'C' or 'S'
        t_mean: mean temperature of samples in first tilt
    OUTPUT:
        alpha: factor which relates DSD/temperature/band to total path attenuation (dB, float)
    """
    
    def _linear_regress(z_bin_centres, median_zdr):
        #remove nan entries
        nan_mask = np.isnan(median_zdr)
        median_zdr = median_zdr[~nan_mask]
        z_bin_centres = z_bin_centres[~nan_mask]

        #linear regression for estimating dzdr/dz
        LS = LinearRegression()
        LS.fit(z_bin_centres.reshape(-1, 1), median_zdr) #, sample_weight=sample_weight)
        return LS
    
    def _median_zdr_bins(z_bin_centres, z_bin_width, z_pairs, zdr_pairs):
        #using the zbins, calculate the median zdr values and number of samples
        median_zdr_array = np.zeros_like(z_bin_centres, dtype=float)
        sample_count_array = np.zeros_like(z_bin_centres, dtype=float)
        for i, centre in enumerate(z_bin_centres):
            bin_lower = centre - z_bin_width/2
            bin_upper = centre + z_bin_width/2
            bin_mask = np.logical_and(z_pairs>=bin_lower, z_pairs<bin_upper)
            median_zdr_array[i] = np.nanmedian(zdr_pairs[bin_mask])
            sample_count_array[i] = np.sum(bin_mask)

        return median_zdr_array, sample_count_array
    
    def plot_fits_fun(z_pairs, zdr_pairs, z_bin_centres, median_zdr, LS=None):
        #plot fits
        fig = plt.figure(figsize=[10,10])
        plt.plot(z_pairs, zdr_pairs, 'k.', label='Pairs')
        plt.plot(z_bin_centres, median_zdr, 'r.', label='Median ZDR')
        if LS is not None:
            plt.plot(z_bin_centres, LS.predict(z_bin_centres.reshape(-1, 1)), 'b-', label='LS fit')
        plt.legend()
        plt.xlabel('Z')
        plt.ylabel('ZDR')
    
    #retrieve alpha fits, defaults and maximum
    fit_a, fit_b, default_conv_alpha, default_strat_alpha, max_alpha = _alpha_function_from_temperature(band, t_mean)
    if verbose:
        print('defaults',default_conv_alpha, default_strat_alpha)
        print('temp:', t_mean, 'C')
        print('')
    
    #define z bin width
    z_bin_width = 2
    
    #define scaling to transform bin thresholds provided by Zhang to those suitable for Australian radars
    #NEXRAD data is 0.5deg azimuth, 250m gates and at least 300km range.
    #Australian data is 1.0deg azumuth, 250m dates and 150km range.
    #set to 0.25, seems to work well, needs a sensitivity study on the actual ranges
    rescale = 0.25
    
    #define individual bin thresholds (from email with Jian Zhang 2020/06/25)
    min_10_18dbz = [1200*rescale]*5 #bin centres: 10,12,14,16,18 (len:5)
    min_20_30dbz = [1000*rescale]*6 #bin centres: 20,22,24,26,28,30 (len:6)
    min_32_34dbz = [1000*rescale]*2 #bin centres: 32,34 (len:2)
    min_36_38dbz = [900*rescale]*2  #bin centres: 36,38 (len:2)
    min_40dbz = [800*rescale]*1     #bin centres: 40 (len:1)
    min_42_44dbz = [600*rescale]*2  #bin centres: 42,44 (len:2) (changed from 700 to 600)
    min_46_48dbz = [400*rescale]*2  #bin centres: 46,48 (len:2) (changed from 600 to 400)
    min_50dbz = [240*rescale]*1     #bin centres: 50 (len:1) (changed from 500 to 240 to be less conservative 0> Australian radars sample large volumes, so reduces high refl)
        
    #define z bins and tests for each case
    #case 1: 20-50dBZ bin centres, threshold of 8 valid median zdr values (in 20-50dBZ) and 3 valid median zdr values in 42-50dBZ
    case1_zbins = np.arange(20, 50+z_bin_width, z_bin_width)
    case1_zdrbins, case1_bin_count = _median_zdr_bins(case1_zbins, z_bin_width, z_pairs, zdr_pairs)
    case1_bin_mins = np.array(min_20_30dbz+min_32_34dbz+min_36_38dbz+min_40dbz+min_42_44dbz+min_46_48dbz+min_50dbz)
    case1_valid_bins = 8
    case1a_zbins = np.arange(42, 50+z_bin_width, z_bin_width)
    case1a_zdrbins, case1a_bin_count = _median_zdr_bins(case1a_zbins, z_bin_width, z_pairs, zdr_pairs)
    case1a_bin_mins = np.array(min_42_44dbz+min_46_48dbz+min_50dbz)
    case1a_valid_bins = 3
    #case 2: 10-30dBZ bin centres, threshold of 11 valid median zdr values
    case2_zbins = np.arange(10, 30+z_bin_width, z_bin_width)
    case2_zdrbins, case2_bin_count = _median_zdr_bins(case2_zbins, z_bin_width, z_pairs, zdr_pairs)
    case2_bin_mins = np.array(min_10_18dbz+min_20_30dbz)
    case2_valid_bins = 11
    #case 3: 10-40dBZ bin centres, threshold of 9 valid median zdr values
    case3_zbins = np.arange(10, 40+z_bin_width, z_bin_width)
    case3_zdrbins, case3_bin_count = _median_zdr_bins(case3_zbins, z_bin_width, z_pairs, zdr_pairs)
    case3_bin_mins = np.array(min_10_18dbz+min_20_30dbz+min_32_34dbz+min_36_38dbz+min_40dbz)
    case3_valid_bins = 9
    #case 4: 44-50dBZ bin centres, threshold of 50 samples across range
    case4_zbins = np.arange(44, 50+z_bin_width, z_bin_width)
    case4_zdrbins, case4_bin_count = _median_zdr_bins(case4_zbins, z_bin_width, z_pairs, zdr_pairs)
    case4_min = 50   #(from email with Jian Zhang 2020/06/25)
    
    if verbose:
        print('')
        print('case 1 valid bins', np.sum(np.greater_equal(case1_bin_count, case1_bin_mins)))
        print('case 1 thresholds', case1_valid_bins)
        print('case 1a valid bins', np.sum(np.greater_equal(case1a_bin_count, case1a_bin_mins)))
        print('case 1a thresholds', case1a_valid_bins)
        print('case 2 valid bins', np.sum(np.greater_equal(case2_bin_count, case2_bin_mins)))
        print('case 2 thresholds', case2_valid_bins)
        print('case 3 valid bins', np.sum(np.greater_equal(case3_bin_count, case3_bin_mins)))
        print('case 3 thresholds', case3_valid_bins)
        print('case 4', np.sum(case4_bin_count) >= case4_min)
        print('case 4 total count', np.sum(case4_bin_count))
        print('case 4 min', case4_min)
        print('')
    
    #case 1
    if np.sum(np.greater_equal(case1_bin_count, case1_bin_mins)) >= case1_valid_bins and np.sum(np.greater_equal(case1a_bin_count, case1a_bin_mins)) >= case1a_valid_bins:
        if verbose:
            print('1: fitted 20-50dbZ')
        LS = _linear_regress(case1_zbins, case1_zdrbins)
        alpha = _slope_alpha_func(LS.coef_[0], fit_a, fit_b, max_alpha)
        alpha_method = 1
        if plot_fits:
            plot_fits_fun(z_pairs, zdr_pairs, case1_zbins, case1_zdrbins, LS)
    #case 2
    elif np.sum(np.greater_equal(case2_bin_count, case2_bin_mins)) >= case2_valid_bins:
        if verbose:
            print('2: 10-30dBZ valid, used default stratiform')
        alpha = default_strat_alpha
        alpha_method = 2
        if plot_fits:
            plot_fits_fun(z_pairs, zdr_pairs, case2_zbins, case2_zdrbins)
    #case 3
    elif np.sum(np.greater_equal(case3_bin_count, case3_bin_mins)) >= case3_valid_bins:
        if verbose:
            print('3: fitted 10-40dbz')
        LS = _linear_regress(case3_zbins, case3_zdrbins)
        alpha = _slope_alpha_func(LS.coef_[0], fit_a, fit_b, max_alpha)
        alpha_method = 3
        if plot_fits:
            plot_fits_fun(z_pairs, zdr_pairs, case3_zbins, case3_zdrbins, LS)
    #case 4
    elif np.sum(case4_bin_count) >= case4_min:
        if verbose:
            print('4: 44-55dBZ valid, used default convective with', np.sum(case4_bin_count), 'convetive samples')
        alpha = default_conv_alpha
        alpha_method = 4
        if plot_fits:
            plot_fits_fun(z_pairs, zdr_pairs, case4_zbins, case4_zdrbins)
    #case 5
    else:
        if verbose:
            print('5: default stratiform')
        alpha = default_strat_alpha
        alpha_method = 5
        if plot_fits:
            plot_fits_fun(z_pairs, zdr_pairs, case4_zbins, case4_zdrbins)
    return alpha, alpha_method

def estimate_alpha_zhang2020(radar, band, scan_idx,
                   min_z=10, max_z=50, min_zdr=-4, max_zdr=4, min_rhohv=0.98,
                   min_r=20, max_r=120,
                   refl_field='reflectivity', zdr_field='corrected_differential_reflectivity', rhohv_field='corrected_cross_correlation_ratio',
                   isom_field='height_over_isom', temp_field='temperature', verbose=False,
                   z_offset=0, zdr_offset=0):
    
    """
    WHAT: Estimate alpha by accumulating Z - ZDR pairs across scans until the pair threshold has been reaches,
            and then fitting a slope to these pairs using _find_z_zdr_slope.
    INPUT:
        radar: pyart radar object
        alpha_dict: dictionary containing Z and ZDR pairs and alpha ts
        min_z: minimum reflectivity for pairs (float, dB)
        max_z: maximum reflectivity for pairs (float, dB)
        max_zdr: minimum differential reflectivity for pairs (float, dB)
        min_zdr: maximum differential reflectivity for pairs (float, dB)
        min_rhohv: minimum cross correlation for pairs (float)
        min_r: minimum range (km)
        max_r: maximum range (km)
        
    OUTPUT:
        alpha: alpha value (float)
    """
    
    #get radar time
    radar_starttime = cftime.num2pydate(radar.time['data'][0], radar.time['units'])
    #extract data
    t_data = radar.get_field(scan_idx, temp_field)
    z_data = radar.get_field(scan_idx, refl_field).filled() + z_offset
    zdr_data = radar.get_field(scan_idx, zdr_field).filled() + zdr_offset
    rhohv_data = radar.get_field(scan_idx, rhohv_field).filled()
    isom_data = radar.get_field(scan_idx, isom_field)
    range_vec = radar.range['data']/1000
    azi_vec = radar.get_azimuth(scan_idx)
    range_data, _ = np.meshgrid(range_vec, azi_vec)
    
    #build masks
    z_mask = np.logical_and(z_data>=min_z, z_data<=max_z)
    zdr_mask = np.logical_and(zdr_data>=min_zdr, zdr_data<=max_zdr)
    nan_mask = np.logical_and(~np.isnan(z_data), ~np.isnan(zdr_data))
    rhv_mask = rhohv_data>min_rhohv
    h_mask = isom_data==0 #below melting level
    r_mask = np.logical_and(range_data>=min_r, range_data<=max_r)
    final_mask = z_mask & zdr_mask & rhv_mask & h_mask & r_mask & nan_mask
    
    #get mean temperature of first tilt
    try:
        t_mean = np.nanmean(t_data[final_mask]) #this will crash if no valid areas
    except:
        t_mean = np.nanmean(t_data)
    
    #find alpha
    alpha, alpha_method = _find_alpha_zhang2020(z_data[final_mask], zdr_data[final_mask], band, t_mean, verbose=verbose)
    if verbose:
        print('alpha value', alpha)
                           
    return alpha, alpha_method

def estimate_alpha_wang2019(radar, alpha_dict, band, pair_threshold=30000, min_pairs=500,
                   min_z=20, max_z=50, min_zdr=-4, max_zdr=4, min_rhohv=0.98,
                  refl_field='reflectivity', zdr_field='corrected_differential_reflectivity', rhohv_field='corrected_cross_correlation_ratio',
                  isom_field='height_over_isom', temp_field='temperature', verbose=False):
    
    """
    WHAT: Estimate alpha by accumulating Z - ZDR pairs across scans until the pair threshold has been reaches,
            and then fitting a slope to these pairs using _find_z_zdr_slope.
    INPUT:
        radar: pyart radar object
        alpha_dict: dictionary containing Z and ZDR pairs and alpha ts
        pair_threshold: number of pairs required for Z-ZDR slope calculation (int)
        min_z: minimum reflectivity for pairs (float, dB)
        max_z: maximum reflectivity for pairs (float, dB)
        max_zdr: minimum differential reflectivity for pairs (float, dB)
        min_zdr: maximum differential reflectivity for pairs (float, dB)
        min_rhohv: minimum cross correlation for pairs (float)
        
    OUTPUT:
        alpha_dict: dictionary containing Z and ZDR pairs and alpha ts
    """
    
    #define default alpha and fitting parameters
    #if band == 'S':
        #default_alpha = 0.015
        #z_zdr_slope_threshold = 0.045
        #fit_c = 0.049 Wang 2020, no longer used
        #fit_m = -0.75
    #elif band == 'C':
        #quadratic model: 31.94 k**2 - 3.103 k + 0.134
        #default_alpha = 0.1
        #default_alpha = 0.063
        #z_zdr_slope_threshold = 0.035
        #fit_c = 0.1168
        #fit_m = -1.532
    
    #get radar time
    radar_starttime = cftime.num2pydate(radar.time['data'][0], radar.time['units'])
    #extract data
    t_data = radar.get_field(scan_idx, temp_field)
    z_data = radar.get_field(scan_idx, refl_field).filled()
    zdr_data = radar.get_field(scan_idx, zdr_field).filled()
    rhohv_data = radar.get_field(scan_idx, rhohv_field).filled()
    isom_data = radar.get_field(scan_idx, isom_field)
    
    #build masks
    z_mask = np.logical_and(z_data>=min_z, z_data<=max_z)
    zdr_mask = np.logical_and(zdr_data>=min_zdr, zdr_data<=max_zdr)
    nan_mask = np.logical_and(~np.isnan(z_data), ~np.isnan(zdr_data))
    rhv_mask = rhohv_data>min_rhohv
    h_mask = isom_data==0 #below melting level
    final_mask = z_mask & zdr_mask & rhv_mask & h_mask & nan_mask
    
    #get mean temperature of first tilt
    t_data = radar.get_field(0, temp_field)
    try:
        t_mean = np.mean(t_data[final_mask]) #this will crash if no valid areas
    except:
        t_mean = np.mean(t_data)
    
    #generate dzdr/dz - alpha fit coefficents
    fit_a, fit_b, default_alpha, _, max_alpha = _alpha_function_from_temperature(band, t_mean)
    if verbose:
        print(fit_a, fit_b, default_alpha, max_alpha, t_mean)
        
    #don't collate for small numbers of samples
    if len(z_data[final_mask]) < min_pairs:
        final_mask = np.zeros_like(final_mask)
    
    #collate z and zdr pairs
    alpha_dict['z_pairs'] = np.append(alpha_dict['z_pairs'] , z_data[final_mask])
    alpha_dict['zdr_pairs']  = np.append(alpha_dict['zdr_pairs'] , zdr_data[final_mask])
    
    #halt if insufficent number of pairs
    n_pairs = len(alpha_dict['z_pairs'])
    if n_pairs < pair_threshold:
        #update alpha timeseries
        if len(alpha_dict['alpha_ts'])>0:
            if verbose:
                print('insufficent pairs', n_pairs, '- Using previous alpha of', alpha_dict['alpha_ts'][-1])
            alpha_dict['alpha_ts'].append(alpha_dict['alpha_ts'][-1]) #update using last alpha
        else:
            if verbose:
                print('insufficent pairs', n_pairs, '- Using default alpha of', default_alpha)
            alpha_dict['alpha_ts'].append(default_alpha)#update to default alpha
        alpha_dict['dt_ts'].append(radar_starttime)
        return alpha_dict

    #find z-zdr slope
    if verbose:
        print(n_pairs, 'pairs found, finding Z-ZDR slope')
    K = _find_z_zdr_slope(alpha_dict)
    if verbose:
        print('slope value', K)

    #update alpha
#     if K < z_zdr_slope_threshold:
#         alpha = fit_m*K + fit_c
#     else:
#         alpha = default_alpha
    alpha = _slope_alpha_func(K, fit_a, fit_b, max_alpha)

    if verbose:
        print('alpha value', alpha)

    #update timeseries
    alpha_dict['alpha_ts'].append(alpha)
    alpha_dict['dt_ts'].append(radar_starttime)
    #reset pairs
    alpha_dict['z_pairs'] = []
    alpha_dict['zdr_pairs'] = []
                           
    return alpha_dict


def retrieve_zphi(radar, band, alpha, alpha_method=1, beta=0.64884, smooth_window_len=5, rhohv_edge_threshold=0.98, refl_edge_threshold=5,
         refl_field='reflectivity', phidp_field='corrected_differential_phase', rhohv_field='corrected_cross_correlation_ratio',
         hca_field='radar_echo_classification', isom_field='height_over_isom', ah_field='specific_attenuation', corz_field='corrected_reflectivity',
         z_offset=0, ncar_pid_values=[6,7], csu_pid_values=[9, 10]):
        
    """
    WHAT: Implementation of zphi technique for estimating specific attenuation from Ryzhkov et al.
    Adpated from pyart.
    
    INPUTS:
        radar: pyart radar object
        alpha: coefficent that is dependent on wavelength and DSD (float)
        beta: coefficent that's dependent on wavelength (float)
        smooth_window_len: used for calculating a moving average in the radial direction for the reflectivity field (int)
        rhohv_edge_threshold: threshold for detecting first and last gates used for total PIA calculation (float)
        refl_edge_threshold: threshold for detecting first and last gates used for total PIA calculation (float, dBZ)
        various field names
    
    OUTPUTS:
        radar: pyart radar object with specific attenuation field
    
    https://arm-doe.github.io/pyart/_modules/pyart/correct/attenuation.html#calculate_attenuation
    
    Ryzhkov et al. Potential Utilization of Specific Attenuation for Rainfall
    Estimation, Mitigation of Partial Beam Blockage, and Radar Networking,
    JAOT, 2014, 31, 599-619.
    """
    
    # extract fields and parameters from radar if they exist
    # reflectivity and differential phase must exist
    # create array to hold the output data
    radar.check_field_exists(refl_field)
    refl = radar.fields[refl_field]['data'].copy() + z_offset
    radar.check_field_exists(phidp_field)
    phidp = radar.fields[phidp_field]['data'].copy()
    radar.check_field_exists(rhohv_field)
    rhohv = radar.fields[rhohv_field]['data'].copy()
    
    ah = np.ma.zeros(refl.shape, dtype='float64')
    pia = np.ma.zeros(refl.shape, dtype='float64')
    
    #smooth reflectivity
    sm_refl = _smooth_masked(refl, wind_len=smooth_window_len,
                            min_valid=1, wind_type='mean')
    radar.add_field_like(refl_field, 'sm_reflectivity', sm_refl, replace_existing=True)

    #load gatefilter
    gatefilter = pyart.correct.GateFilter(radar)
    
    #mask clutter
    pid = radar.fields[hca_field]['data']
    gatefilter.exclude_gates(np.ma.getmask(pid))
    #mask hail
    if radar.fields[hca_field]['long_name'] == 'NCAR Hydrometeor classification':
        hail_pid_values=ncar_pid_values
    else:
        hail_pid_values=csu_pid_values
        
    for hail_value in hail_pid_values:
        gatefilter.exclude_gates(pid==hail_value)
    #mask data above melting level
    isom = radar.fields[isom_field]['data']
    gatefilter.exclude_gates(isom > 0)

    #create rhohv and z mask for determining r1 and r2
    edge_mask = np.logical_or(rhohv.filled(fill_value=0) < rhohv_edge_threshold, sm_refl.filled(fill_value=0) < refl_edge_threshold)

    #despeckle gatefilter (both foles and valid regions)
    valid_mask = gatefilter.gate_included
    valid_mask_filt = morphology.remove_small_holes(valid_mask, area_threshold=10)
    valid_mask_filt = morphology.remove_small_objects(valid_mask_filt, min_size=10)
    gatefilter.include_gates(valid_mask_filt)
                                                            
    #prepare phidp
    mask_phidp = np.ma.getmaskarray(phidp)
    mask_phidp = np.logical_and(mask_phidp, ~valid_mask_filt)
    corr_phidp = np.ma.masked_where(mask_phidp, phidp).filled(fill_value=0)
    
    #convert refl to z and gate spacing (in km)
    refl_linear = np.ma.power(10.0, 0.1 * beta * sm_refl).filled(fill_value=0)
    dr = (radar.range['data'][1] - radar.range['data'][0]) / 1000.0

    #find end indicies in reject_mask
    end_gate_arr = np.zeros(radar.nrays, dtype='int32')
    start_gate_arr = np.zeros(radar.nrays, dtype='int32')
    #combine edge + gatefilter
    gate_mask = np.logical_and(gatefilter.gate_included, ~edge_mask)
    
    for ray in range(radar.nrays):
        ind_rng = np.where(gate_mask[ray, :] == 1)[0]
#         if len(ind_rng) > 1:
#             #CP2 experences invalid data in the first 5 gates. ignore these gates
#             ind_rng = ind_rng[ind_rng>6]
        if len(ind_rng) > 1:
            # there are filtered gates: The last valid gate is one
            # before the first filter gate
            end_gate_arr[ray] = ind_rng[-1]-1 #ensures that index is -1 if all rays are masked
            start_gate_arr[ray] = ind_rng[0]
            
    for ray in range(radar.nrays):
        # perform attenuation calculation on a single ray

        # if number of valid range bins larger than smoothing window
        if end_gate_arr[ray]-start_gate_arr[ray] > smooth_window_len:
            # extract the ray's phase shift,
            # init. refl. correction and mask
            ray_phase_shift = corr_phidp[ray, start_gate_arr[ray]:end_gate_arr[ray]]
            ray_mask = valid_mask_filt[ray, start_gate_arr[ray]:end_gate_arr[ray]]
            ray_refl_linear = refl_linear[ray, start_gate_arr[ray]:end_gate_arr[ray]]

            # perform calculation if there is valid data
            last_six_good = np.where(np.ndarray.flatten(ray_mask) == 1)[0][-6:]
            if(len(last_six_good)) == 6:
                phidp_max = np.median(ray_phase_shift[last_six_good])
                #abort if phase change is equal to zero
                if phidp_max <= 0:
                    continue
                self_cons_number = (
                    np.exp(0.23 * beta * alpha * phidp_max) - 1.0)
                I_indef = cumtrapz(0.46 * beta * dr * ray_refl_linear[::-1])
                I_indef = np.append(I_indef, I_indef[-1])[::-1]

                # set the specific attenutation and attenuation
                ah[ray, start_gate_arr[ray]:end_gate_arr[ray]] = (
                    ray_refl_linear * self_cons_number /
                    (I_indef[0] + self_cons_number * I_indef))
                
                pia[ray, :-1] = cumtrapz(ah[ray, :]) * dr * 2.0
                pia[ray, -1] = pia[ray, -2]
                
                
    #add ah into radar
    spec_at = pyart.config.get_metadata('specific_attenuation')
    ah_masked = np.ma.masked_where(gatefilter.gate_excluded, ah)
    spec_at['data'] = ah_masked.astype(np.float32)
    spec_at['alpha'] = alpha
    spec_at['alpha_method'] = alpha_method
    spec_at['_FillValue'] = ah_masked.fill_value
    radar.add_field(ah_field, spec_at, replace_existing=True)
    
    #add corrected refl to radar
    cor_z = pyart.config.get_metadata('corrected_reflectivity')
    cor_z_masked = np.ma.masked_where(gatefilter.gate_excluded, pia + refl)
    cor_z['data'] = cor_z_masked.astype(np.float32)
    cor_z['_FillValue'] = cor_z_masked.fill_value
    radar.add_field(corz_field, cor_z, replace_existing=True)

    #add PIA
    cor_z = pyart.config.get_metadata('corrected_reflectivity')
    cor_z_masked = np.ma.masked_where(gatefilter.gate_excluded, pia + refl)
    cor_z['data'] = cor_z_masked.astype(np.float32)
    cor_z['_FillValue'] = cor_z_masked.fill_value
    radar.add_field(corz_field, cor_z, replace_existing=True)
    
    
    return radar