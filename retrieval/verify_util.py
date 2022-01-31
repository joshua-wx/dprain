from datetime import datetime, timedelta

import pandas
import numpy as np

def read_csv(csv_ffn, header_line):
    """
    CSV reader used for the radar locations file (comma delimited)
    
    Parameters:
    ===========
        csv_ffn: str
            Full filename to csv file
            
        header_line: int or None
            to use first line of csv as header = 0, use None to use column index
            
    Returns:
    ========
        as_dict: dict
            csv columns are dictionary
    
    """
    df = pandas.read_csv(csv_ffn, header=header_line, dtype=str)
    as_dict = df.to_dict(orient='list')
    return as_dict

def read_rain_1min(rain_file_list):
    
    """
    WHAT: This function takes a list of HD01D csv files continaing 
    1 minute rainfall data and reads out the specific columns for rainfall since last measurement.
    An offset is made of -1 minute to move the timestamp forward to the start of the measurement minute.
    Empty and zero data is removed.
    """
    
    dt_array = np.array([])
    data_array = np.array([])
    
    #loo through sorted list
    for rain_file in rain_file_list:

        #load rain data
        csv_dict = read_csv(rain_file, 0)  
        
        #extract data
        dtstr_list = csv_dict['Year Month Day Hours Minutes in YYYYMMDDHH24MI format in Universal coordinated time']
        data_list = csv_dict['Precipitation since last (AWS) observation in mm']
          
        filter_dt_list = []
        filter_data_list = []
        #remove empty and zero arrays
        for i, data_str_value in enumerate(data_list):
            if data_str_value.strip() in ['0.0','']:
                continue
            #convert to float and assign
            data_value = float(data_str_value)
            filter_data_list.append(data_value)
            #parse time data to datetime and subtract one hour to convert to minute starting observations
            dt_value  = datetime.strptime(dtstr_list[i], '%Y%m%d%H%M') - timedelta(minutes=1)
            filter_dt_list.append(dt_value)

        #convert lists to arrays
        filter_dt_array = np.array(filter_dt_list)
        filter_data_array = np.array(filter_data_list)
        
        #append to master arrays
        dt_array = np.append(dt_array, filter_dt_array)
        data_array = np.append(data_array, filter_data_array)
        
    data_dict = {'dt':dt_array, 'rain':data_array}
    
    return data_dict