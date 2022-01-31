from glob import glob

import numpy as np
import netCDF4

def rf_standard_parallel_lookup(radar_id):
    #lookup standard parallels for each radar site.
    #copied from http://ho-rainfields-sw.bom.gov.au/rainfields/sites.php)
    ref_list = [
        (2,-36.3, -39.4),
        (64,-33.1,-36.2),
        (66,-26.2,-29.3),
        (71,-32.2,-35.2),
        (31,-33.4,-36.5),
        (32,-32.3,-35.4),
        (95,-34.4,-37.5)
    ]
    
    for item in ref_list:
        if item[0] == radar_id:
            return item[1], item[2]
    
    return None, None

def rf_grid_centre_lookup(radar_id):
    #lookup lon0 and lat0 for each radar site.
    #copied from http://ho-rainfields-sw.bom.gov.au/rainfields/sites.php)
    ref_list = [
        (2,144.752,-37.852),
        (64,138.469,-34.6169),
        (66,153.24,-27.7178),
        (71,151.209,-33.7008),
        (31,117.816,34.9418),
        (32,121.892,-33.83),
        (95,142.0133,-35.9976)
    ]
    
    for item in ref_list:
        if item[0] == radar_id:
            return item[1], item[2]
    
    return None, None

def write_rf_nc(rf_ffn, rid, valid_time, rain_rate, reflectivity, kdp, zdr, origin_lon, origin_lat, parallels):

    """
    WHAT: Write a RF3 instaneous rainrate file
    rain_rate grid must be a 512x512 array. With extent 128m.
    """
    
    #create RF3 dims
    y_bounds1 = np.linspace(-128, 127.5, 512)
    y_bounds2 = y_bounds1.copy() + 0.5
    y_bounds = np.rot90(np.stack([y_bounds2, y_bounds1]))
    x_bounds = np.fliplr(np.flipud(y_bounds.copy()))
    x = np.linspace(-127.75,127.75,512)
    y = np.linspace(127.75,-127.75,512)
    
    #flip rain field into image coordinates
    rain_rate = np.flipud(rain_rate)
    reflectivity = np.flipud(reflectivity)
    kdp = np.flipud(kdp)
    zdr = np.flipud(zdr)
    
    # Write data
    with netCDF4.Dataset(rf_ffn, 'w') as ncid:
        #create dimensions
        ncid.createDimension('n2', 2)
        ncid.createDimension("x", 512)
        ncid.createDimension("y", 512)

        #set global
        ncid.setncattr('licence', 'http://www.bom.gov.au/other/copyright.shtml')
        ncid.setncattr('source', 'dprain testing')
        ncid.setncattr('station_id', np.intc(rid))
        ncid.setncattr('institution', 'Commonwealth of Australia, Bureau of Meteorology (ABN 92 637 533 532)')
        ncid.setncattr('station_name', '')
        ncid.setncattr('Conventions', 'CF-1.7')
        ncid.setncattr('title', 'Radar derived rain rate')

        #create x/y bounds
        ncx_bounds = ncid.createVariable('x_bounds', np.float, ('x','n2'))
        ncy_bounds = ncid.createVariable('y_bounds', np.float, ('y','n2'))    
        ncx_bounds[:] = x_bounds
        ncy_bounds[:] = y_bounds

        #create x/y vars
        ncx = ncid.createVariable('x', np.float, ('x'))
        ncy = ncid.createVariable('y', np.float, ('y'))
        ncx[:] = x
        ncx.units = 'km'
        ncx.bounds = 'x_bounds'
        ncx.standard_name = 'projection_x_coordinate'
        ncy[:] = y
        ncy.units = 'km'
        ncy.bounds = 'y_bounds'
        ncy.standard_name = 'projection_y_coordinate'

        #create time var
        nct = ncid.createVariable('valid_time', np.int_)
        nct[:] = valid_time
        nct.long_name = 'Valid time'
        nct.standard_name = 'time'
        nct.units = 'seconds since 1970-01-01 00:00:00 UTC'
        
        #write rain rate
        ncrain = ncid.createVariable('rain_rate', np.float, ("y", "x"), zlib=True,
                                       fill_value=np.nan, chunksizes=(512,512))
        ncrain[:] = rain_rate
        ncrain.grid_mapping = 'proj'
        ncrain.long_name = 'Rainfall rate'
        ncrain.standard_name = 'rainfall_rate'
        ncrain.units = 'mm hr-1'
        ncrain.scale_factor = 1
        ncrain.add_offset = 0.0
        
        #write reflectivity var
        ncrefl = ncid.createVariable('reflectivity', np.float, ("y", "x"), zlib=True,
                                       fill_value=np.nan, chunksizes=(512,512))
        ncrefl[:] = reflectivity
        ncrefl.grid_mapping = 'proj'
        ncrefl.long_name = 'Radar Measured Reflectivity rate'
        ncrefl.standard_name = 'reflectivity'
        ncrefl.units = 'dBZ'
        ncrefl.scale_factor = 1
        ncrefl.add_offset = 0.0

        #write kdp var
        nckdp = ncid.createVariable('specific_differential_phase', np.float, ("y", "x"), zlib=True,
                                       fill_value=np.nan, chunksizes=(512,512))
        nckdp[:] = kdp
        nckdp.grid_mapping = 'proj'
        nckdp.long_name = 'corrected specific differential phase'
        nckdp.standard_name = 'kdp'
        nckdp.units = 'deg/km'
        nckdp.scale_factor = 1
        nckdp.add_offset = 0.0
        
        #write zdr var
        nczdr = ncid.createVariable('differential_reflectivity', np.float, ("y", "x"), zlib=True,
                                       fill_value=np.nan, chunksizes=(512,512))
        nczdr[:] = zdr
        nczdr.grid_mapping = 'proj'
        nczdr.long_name = 'corrected differential reflectivity'
        nczdr.standard_name = 'zdr'
        nczdr.units = 'dB'
        nczdr.scale_factor = 1
        nczdr.add_offset = 0.0
        
        #write proj
        ncproj = ncid.createVariable('proj', np.byte)
        ncproj[:] = 0
        ncproj.grid_mapping_name = 'albers_conical_equal_area'
        ncproj.false_easting = 0.0
        ncproj.false_northing = 0.0
        ncproj.semi_major_axis = 6378137.0
        ncproj.semi_minor_axis = 6356752.31414
        ncproj.longitude_of_central_meridian = origin_lon
        ncproj.latitude_of_projection_origin = origin_lat
        ncproj.standard_parallel = parallels