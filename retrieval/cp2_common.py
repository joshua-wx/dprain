# Python Standard Library
import os
import re
import glob
import time
import fnmatch
import datetime

# Other Libraries
import pyart
import scipy
import cftime
import netCDF4
import numpy as np
import pandas as pd
import xarray as xr

import radar_codes

def generate_isom(radar):
    """
    Generate fields for radar object that is height relative to the
    melting level (at the radar site using era5 data)
    Parameters:
    ===========
    radar:
        Py-ART radar object.
    Returns:
    ========
    iso0_info_dict: dict
        Height field relative to melting level
    """
    grlat = radar.latitude['data'][0]
    grlon = radar.longitude['data'][0]
    dtime = pd.Timestamp(cftime.num2pydate(radar.time['data'][0], radar.time['units']))

    year = dtime.year
    era5 = f'/g/data/rq0/admin/temperature_profiles/era5_data/{year}_openradar_temp_geopot.nc'
    if not os.path.isfile(era5):
        raise FileNotFoundError(f'{era5}: no such file for temperature.')

    # Getting the temperature
    dset = xr.open_dataset(era5)
    temp = dset.sel(longitude=grlon, latitude=grlat, time=dtime, method='nearest')
    
    #extract data
    geopot_profile = np.array(temp.z.values/9.80665) #geopot -> geopotH
    temp_profile = np.array(temp.t.values - 273.15)
    
    #append surface data using lowest level
    geopot_profile = np.append(geopot_profile,[0])
    temp_profile = np.append(temp_profile, temp_profile[-1])
        
    #find melting level
    melting_level = radar_codes.find_melting_level(temp_profile, geopot_profile)
    
    # retrieve the Z coordinates of the radar gates
    rg, azg = np.meshgrid(radar.range['data'], radar.azimuth['data'])
    rg, eleg = np.meshgrid(radar.range['data'], radar.elevation['data'])
    _, _, z = pyart.core.antenna_to_cartesian(rg / 1000.0, azg, eleg)
    
    #calculate height above melting level
    isom_data = (radar.altitude['data'] + z) - melting_level
    isom_data[isom_data<0] = 0
    
    isom_info_dict = {'data': isom_data, # relative to melting level
                      'long_name': 'Height relative to (H0+H10)/2 level',
                      'standard_name': 'relative_melting_level_height',
                      'units': 'm'}
    
    radar.add_field('height_over_isom', isom_info_dict)

    return radar

def add_ncar_pid(radar, derived_path, vol_ffn):
    
    """
    pid.cl	    (1)	 # Cloud                        
    pid.drz     (2)  # Drizzle                      
    pid.lr	    (3)  # Light_Rain                   
    pid.mr	    (4)  # Moderate_Rain                
    pid.hr	    (5)  # Heavy_Rain                   
    pid.ha	    (6)  # Hail                         
    pid.rh	    (7)  # Rain_Hail_Mixture            
    pid.gsh     (8)  # Graupel_Small_Hail           
    pid.grr     (9)  # Graupel_Rain                 
    pid.ds	    (10) # Dry_Snow                     
    pid.ws	    (11) # Wet_Snow                     
    pid.ic	    (12) # Ice_Crystals                 
    pid.iic     (13) # Irreg_Ice_Crystals           
    pid.sld     (14) # Supercooled_Liquid_Droplets  
    pid.bgs     (15) # Flying_Insects               
    pid.trip2   (16) # Second trip                  
    pid.gcl     (17) # Ground_Clutter          
    pid.sat     (18) # Receiver saturation
    """
    
    #build derived filename
    vol_fn = os.path.basename(vol_ffn)[6:-12]
    derived_fn = 'cp2-derived_' + vol_fn + '.mdv'
    derived_ffn = derived_path + '/' + derived_fn
    #load
    derived = pyart.io.read(derived_ffn, file_field_names=True)
    #extract PID
    pid = derived.fields['PID']
    #add to radar
    radar.add_field('PID', pid, replace_existing=True)
    return radar
    