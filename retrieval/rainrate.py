import numpy as np

import pyart

def _invpower_func(z, a, b):
    return (z / a) ** (1. / b)

def _power_func(data, a, b):
    return a*(data**b)


def conventional(radar, alpha=60, beta=1.7, refl_field='sm_reflectivity', zr_field='zr_rainrate',
                z_offset=0):
    """
    WHAT: retrieve conventional rain rates using ZR technique
    INPUTS:
        radar: pyart radar object
        alpha/beta: coefficents used in inverse powerlaw function to derive rainrate from Z (float)
        various field names for input and output
    OUTPUTS:
        radar: pyart radar object
    """
        
    #get reflectivity field
    radar.check_field_exists(refl_field)
    refl = radar.fields[refl_field]['data'].copy() + z_offset
    refl_z = 10.**(refl/10.)
    
    #calculate rain
    rr_data = _invpower_func(refl_z, alpha, beta)
    
    #empty rain fields
    rain_field = pyart.config.get_metadata('radar_estimated_rain_rate')
    rain_field['data'] = rr_data
    
    #add to radar
    radar.add_field(zr_field, rain_field, replace_existing=True)
    radar.fields[zr_field]['standard_name'] = 'zr_rainrate'
    radar.fields[zr_field]['long_name'] = 'Rainrate from R(Z)'
    
    return radar

def _interpolate_coeff(temp_list, coeff_list, temp_field):
    
    coeff_array = np.zeros_like(temp_field, dtype=float)
    
    #for temps warmer than the max in temp_list
    mask = temp_field >= temp_list[-1]
    coeff_array[mask] = coeff_list[-1]
    #for temps cooler than the min in temp list
    mask = temp_field < temp_list[0]
    coeff_array[mask] = coeff_list[0]    
    #for temps between temp_list values
    #loop through every temp pair (skipping n=0)
    for i, _ in enumerate(temp_list):
        if i==0:
            continue
        #extract upper and lower temps/coeffs
        upper_temp = temp_list[i]
        lower_temp = temp_list[i-1]
        upper_coeff = coeff_list[i]
        lower_coeff = coeff_list[i-1]
        #calculate linear fit slope/intercept
        slope = (upper_coeff - lower_coeff)/(upper_temp - lower_temp)
        intercept = upper_coeff-slope*upper_temp
        #create mask for points which use this fit
        mask = np.logical_and(temp_field>lower_temp, temp_field<=upper_temp)
        #assign coeff values using the mask and linear fit
        coeff_array[mask] = temp_field[mask]*slope+intercept
    
    return coeff_array

def polarimetric(radar, band, refl_field='sm_reflectivity', ah_field='specific_attenuation',
                 kdp_field='corrected_specific_differential_phase', phidp_field='corrected_differential_phase',
                 rhohv_field='corrected_cross_correlation_ratio', temp_field='temperature', isom_field='height_over_isom',
                 zr_field='zr_rainrate', ahr_field='ah_rainrate', kdpr_field='kdp_rainrate', hybridr_field='hybrid_rainrate',
                 hca_field='radar_echo_classification',
                 beamb_data=None, pid_ncar_clutter=17, pid_csu_clutter=1,
                 refl_lower_threshold=45., refl_upper_threshold=50., z_offset=0,
                 min_delta_phidp=2., ah_coeff_fitted=True):
    
    """
    WHAT: retrieve polarimetric rain rates for ah, kdp and hybrid kdp/ah technique
    INPUTS:
        radar: pyart radar object
        refl_threshold: threshold to define transition from ah to kdp rainrate retrieval (dB, float)
        various field names used for input and output
    OUTPUTS:
        radar: pyart radar object
    """
    
    
    
    if band == 'S':
        if ah_coeff_fitted:
            #fitted coefficents
            ah_coeff = {'t':[0,10,20],'a':[2121, 2908, 3836], 'b':[0.99, 0.99, 0.99]}
        else:
            #Ryzhkov et al. 2014 table 1
            ah_coeff = {'t':[0,10,20,30],'a':[2230, 3100, 4120, 5330], 'b':[1.03, 1.03, 1.03, 1.03]}
        #Wang et al. 2019
        #kdp_coeff = {'a':27.00, 'b':0.77}
        #Giangrande and Ryzhkov 2008
        kdp_coeff = {'a':44.00, 'b':0.82}
        
    elif band == 'C':
        if ah_coeff_fitted:
            #fitted coefficents
            ah_coeff = {'t':[0,10,20],'a':[193, 203, 208], 'b':[0.82, 0.77, 0.71]}
        else:
            #Ryzhkov et al. 2014 table 1
            ah_coeff = {'t':[0,10,20,30],'a':[221, 250, 294, 352], 'b':[0.92, 0.91, 0.89, 0.89]}
        #Ryzhkov et al. 2013 coefficients: Polarimetric radar characteristics of melting hail. part II: Practical implications
        kdp_coeff = {'a':25.3, 'b':0.776}
    
    #get fields
    radar.check_field_exists(refl_field)
    refl = radar.fields[refl_field]['data'].copy() + z_offset
    radar.check_field_exists(ah_field)
    ah = radar.fields[ah_field]['data'].copy()
    radar.check_field_exists(kdp_field)
    kdp = radar.fields[kdp_field]['data'].copy()
    radar.check_field_exists(phidp_field)
    phidp = radar.fields[phidp_field]['data'].copy()    
    radar.check_field_exists(temp_field)
    temperature = radar.fields[temp_field]['data'].copy()
    radar.check_field_exists(isom_field)
    height_over_isom = radar.fields[isom_field]['data'].copy()
    radar.check_field_exists(rhohv_field)
    rhohv = radar.fields[rhohv_field]['data'].copy()
    radar.check_field_exists(hca_field)
    pid = radar.fields[hca_field]['data']
    
    #assign ah coefficent assignment using temperature
    ah_a_array = _interpolate_coeff(ah_coeff['t'], ah_coeff['a'], temperature)
    ah_b_array = _interpolate_coeff(ah_coeff['t'], ah_coeff['b'], temperature)
    
    radar.add_field_like(zr_field, 'ah_a_array', ah_a_array, replace_existing=True)
    radar.add_field_like(zr_field, 'ah_b_array', ah_b_array, replace_existing=True)
    
    #retrieve rainrates
    ah_rain  = _power_func(ah, ah_a_array, ah_b_array)
    kdp_rain = _power_func(kdp, kdp_coeff['a'], kdp_coeff['b'])
    kdp_rain = np.ma.masked_where(height_over_isom > 0, kdp_rain)
    
    #mask using beamblockage
    if beamb_data is not None:
        beamb_shape = np.shape(beamb_data)
        data_shape = np.shape(ah_rain)
        #check number of bins in beamb is the same as what's expected
        if beamb_shape[1] < data_shape[1]: #pad beamb shape
            beamb_data = np.pad(beamb_data,  ((0,0),(0,data_shape[1]-beamb_shape[1])), mode='edge')
        elif beamb_shape[1] > data_shape[1]: #cut beamb shape
            beamb_data = beamb_data[:,:data_shape[1]]
            
        if beamb_shape[0] < data_shape[0]: #pad beamb shape
            beamb_data = np.pad(beamb_data,  ((0,0),(data_shape[0]-beamb_shape[0], 0)), mode='edge')
        elif beamb_shape[0] > data_shape[0]: #cut beamb shape
            beamb_data = beamb_data[:data_shape[0],:]
            
        #apply beamblock mask suggested by Zhang et al. 2020
        beamb_mask = beamb_data>0.9
        ah_rain[beamb_mask] = np.nan
        kdp_rain[beamb_mask] = np.nan
        
    #create rain and hail masks, and weighting arrays
    #identify regions of high reflecitity, below the melting layer and not clutter
    kdp_lower_mask = np.logical_and(refl>refl_lower_threshold, height_over_isom == 0)
    if radar.fields[hca_field]['long_name'] == 'NCAR Hydrometeor classification':
        kdp_lower_mask[pid==pid_ncar_clutter] = False
    else:
        kdp_lower_mask[pid==pid_csu_clutter] = False
    
    kdp_weight = (refl.copy()-refl_lower_threshold)/(refl_upper_threshold-refl_lower_threshold)
    kdp_weight[kdp_weight<0] = 0
    kdp_weight[kdp_weight>1] = 1
    ah_weight = 1 - kdp_weight.copy()
    
    #mask rays where the total span is less than 3 degrees
    #find max phidp and pad back to array size
    phidp_shape = np.shape(phidp)
    phidp_span = np.amax(phidp, axis=1)
    phidp_span = np.rot90(np.tile(phidp_span, (phidp_shape[1], 1)), 3)
    phidp_mask = phidp_span<min_delta_phidp
    ah_rain[phidp_mask] = np.nan
    kdp_rain[phidp_mask] = np.nan
#     radar.add_field_like(zr_field, 'kdp_weight', kdp_weight, replace_existing=True)
#     radar.add_field_like(zr_field, 'ah_weight', ah_weight, replace_existing=True)
    
    #crate hybrid kdp/ah rainrate
    hybrid_rain = ah_rain.copy()
    hybrid_rain[kdp_lower_mask] = kdp_rain[kdp_lower_mask]*kdp_weight[kdp_lower_mask] + ah_rain[kdp_lower_mask].filled(0)*ah_weight[kdp_lower_mask]
    
    #add fields to radar object
    radar.add_field_like(zr_field, hybridr_field, hybrid_rain.astype(np.float32), replace_existing=True)
    radar.add_field_like(zr_field, ahr_field, ah_rain.astype(np.float32), replace_existing=True)
    radar.add_field_like(zr_field, kdpr_field, kdp_rain.astype(np.float32), replace_existing=True)
    
    #update names
    radar.fields[hybridr_field]['standard_name'] = 'hydrid_a_and_kdp_rainrate'
    radar.fields[hybridr_field]['long_name'] = 'Rainrate from R(A) and R(kdp)'
    radar.fields[ahr_field]['standard_name'] = 'a_rainrate'
    radar.fields[ahr_field]['long_name'] = 'Rainrate from R(A)'
    radar.fields[kdpr_field]['standard_name'] = 'kdp_rainrate'
    radar.fields[kdpr_field]['long_name'] = 'Rainrate from R(kdp)'
        
    return radar