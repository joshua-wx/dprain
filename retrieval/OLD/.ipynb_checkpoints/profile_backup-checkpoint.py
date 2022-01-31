def temperature_profile_access(radar, source='access', model_timestep_hr=6):
    """
    Compute the signal-to-noise ratio as well as interpolating the radiosounding
    temperature on to the radar grid. The function looks for the radiosoundings
    that happened at the closest time from the radar. There is no time
    difference limit.
    Parameters:
    ===========
    radar:
        Py-ART radar object.
    source:
        string of either access (access g) or era5
    Returns:
    ========
    z_dict: dict
        Altitude in m, interpolated at each radar gates.
    temp_info_dict: dict
        Temperature in Celsius, interpolated at each radar gates.
    """
    grlat = radar.latitude['data'][0]
    grlon = radar.longitude['data'][0]
    dtime = pd.Timestamp(cftime.num2pydate(radar.time['data'][0], radar.time['units']))
    
    if source == 'access':
        if dtime < datetime.strptime('20200924', '%Y%m%d'):
            #APS2
            access_root = '/g/data/lb4/ops_aps2/access-g/1' #access g
        else:
            #APS3
            access_root = '/g/data/wr45/ops_aps3/access-g/1' #access g
        #build folder for access data
        model_timestep_hr = 6
        hour_folder = str(round(dtime.hour/model_timestep_hr)*model_timestep_hr).zfill(2) + '00'
        if hour_folder == '2400':
            hour_folder = '0000'
        access_folder = '/'.join([access_root, datetime.strftime(dtime, '%Y%m%d'), hour_folder, 'an', 'pl'])
        #build filenames
        temp_ffn = access_folder + '/air_temp.nc'
        geop_ffn = access_folder + '/geop_ht.nc'
        if not os.path.isfile(temp_ffn):
            raise FileNotFoundError(f'{temp_ffn}: no such file for temperature.')
        if not os.path.isfile(geop_ffn):
            raise FileNotFoundError(f'{geop_ffn}: no such file for geopotential.')
        #extract data
        with xr.open_dataset(temp_ffn) as temp_ds:
            temp_profile = temp_ds.air_temp.sel(lon=grlon, method='nearest').sel(lat=grlat, method='nearest').data[0] - 273.15 #units: deg C
        with xr.open_dataset(geop_ffn) as geop_ds:
            geopot_profile = geop_ds.geop_ht.sel(lon=grlon, method='nearest').sel(lat=grlat, method='nearest').data[0] #units: m
        #flipdata (ground is first row)
        temp_profile = np.flipud(temp_profile)
        geopot_profile = np.flipud(geopot_profile)
        
    elif source == "era5":
        #set era path
        era5_root = '/g/data/rt52/era5/pressure-levels/reanalysis'
        #build file paths
        month_str = dtime.month
        year_str = dtime.year
        temp_ffn = glob(f'{era5_root}/t/{year_str}/t_era5_oper_pl_{year_str}{month_str:02}*.nc')[0]
        geop_ffn = glob(f'{era5_root}/z/{year_str}/z_era5_oper_pl_{year_str}{month_str:02}*.nc')[0]
        #extract data
        with xr.open_dataset(temp_ffn) as temp_ds:
            temp_data = temp_ds.t.sel(longitude=grlon, method='nearest').sel(latitude=grlat, method='nearest').sel(time=dtime, method='nearest').data[:] - 273.15 #units: deg K -> C
        with xr.open_dataset(geop_ffn) as geop_ds:
            geop_data = geop_ds.z.sel(longitude=grlon, method='nearest').sel(latitude=grlat, method='nearest').sel(time=dtime, method='nearest').data[:]/9.80665 #units: m**2 s**-2 -> m
        #flipdata (ground is first row)
        temp_profile = np.flipud(temp_data)
        geopot_profile = np.flipud(geop_data)
        

    
    #append surface data using lowest level
    geopot_profile = np.append([0], geopot_profile)
    temp_profile = np.append(temp_profile[0], temp_profile)
    
    z_dict, temp_dict = pyart.retrieve.map_profile_to_gates(temp_profile, geopot_profile, radar)
    
    temp_info_dict = {'data': temp_dict['data'],  # Switch to celsius.
                      'long_name': 'Sounding temperature at gate',
                      'standard_name': 'temperature',
                      'valid_min': -100, 'valid_max': 100,
                      'units': 'degrees Celsius',
                      'comment': 'Radiosounding date: %s' % (dtime.strftime("%Y/%m/%d"))}
    
    #generate isom dataset
    melting_level = find_melting_level(temp_profile, geopot_profile)
    
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
    

    return z_dict, temp_info_dict, isom_info_dict


def _sounding_interp(snd_temp,snd_height,target_temp):
    """
    Provides an linear interpolated height for a target temperature using a sounding vertical profile. 
    Looks for first instance of temperature below target_temp from surface upward.

    Parameters:
    ===========
    snd_temp: ndarray
        temperature data (degrees C)
    snd_height: ndarray
        relative height data (m)
    target_temp: float
        target temperature to find height at (m)

    Returns:
    ========
    intp_h: float
        interpolated height of target_temp (m)
    """

    intp_h = np.nan
    #check if target_temp is warmer than lowest level in sounding
    if target_temp>snd_temp[0]:
        print('warning, target temp level below sounding, returning ground level (0m)')
        return 0.
    
    # find index above and below target level
    mask = np.where(snd_temp < target_temp)
    above_ind = mask[0][0]

    # index below
    below_ind = above_ind - 1
    # apply linear interplation to points above and below target_temp
    set_interp = interp1d(
        snd_temp[below_ind:above_ind+1],
        snd_height[below_ind:above_ind+1], kind='linear')
    # apply interpolant
    intp_h = set_interp(target_temp)
    
    return intp_h

def find_melting_level(temp_profile, geop_profile):
    #interpolate to required levels
    plus10_h = _sounding_interp(temp_profile, geop_profile, 10.)
    fz_h = _sounding_interp(temp_profile, geop_profile, 0.)
    #calculate base of melting level
    return (plus10_h+fz_h)/2