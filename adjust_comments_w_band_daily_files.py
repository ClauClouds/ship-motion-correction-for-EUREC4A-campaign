#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 09:54:37 2021
code to change comment in the files and to cut hours that cannot be published 
@author: claudia
"""

import numpy as np
import glob
import xarray as xr
import pandas as pd
from datetime import datetime

# code to read 5 s int time data and resample them on 10 s resolution
path = '/Volumes/Extreme SSD/ship_motion_correction_merian/corrected_data/wband_daily_with_DOI/'
file_list = np.sort(glob.glob(path+'*.nc'))

for indFile, file in enumerate(file_list):

    WbandData = xr.open_dataset(file)
    file_out = file[len(path):]
    
    
    print(file_out)
    
    if indFile == 20:
        
        
        # set to nans all data collected between 11 and 13:59 utc
        time_start = datetime(2020,2,8,11,0,0)
        time_end = datetime(2020,2,8,13,59,59)
        
        # select time indeces that I want to keep
        ds = WbandData.loc[{'time':(pd.to_datetime(WbandData.time.values)<=time_start) | (pd.to_datetime(WbandData.time.values)>=time_end)}]
        
        # reindexing on the original time array and filling with nans
        WbandData = ds.reindex({'time':pd.to_datetime(WbandData.time.values)}, method=None)
        
        #da.loc["2000-01-01", ["IL", "IN"]] = -10
        # add comment on with explained gaps
        attrs = {'CREATED_BY'     : 'Claudia Acquistapace',
                        'CREATED_ON'       : str(datetime.now()),
                        'ORCID-AUTHOR'     : "Claudia Acquistapace: 0000-0002-1144-4753",
                        'DOI'              : '10.25326/235 (https://doi.org/10.25326/235)',
                        'FILL_VALUE'       : 'NaN',
                        'PI_NAME'          : 'Claudia Acquistapace',
                        'PI_AFFILIATION'   : 'University of Cologne (UNI), Germany',
                        'PI_ADDRESS'       : 'Institute for geophysics and meteorology, Pohligstrasse 3, 50969 Koeln',
                        'PI_MAIL'          : 'cacquist@meteo.uni-koeln.de',
                        'DO_NAME'          : 'University of Cologne - Germany',
                        'DO_AFFILIATION'   : 'University of Cologne - Germany',
                        'DO_address'       : 'Institute for geophysics and meteorology, Pohligstrasse 3, 50696 Koeln',
                        'DO_MAIL'          : 'cacquist@meteo.uni-koeln.de',
                        'DS_NAME'          : 'University of Cologne - Germany',
                        'DS_AFFILIATION'   : 'University of Cologne - Germany',
                        'DS_address'       : 'Institute for geophysics and meteorology, Pohligstrasse 3, 50696 Koeln',
                        'DS_MAIL'          : 'cacquist@meteo.uni-koeln.de',
                        'DATA_DESCRIPTION' : 'daily w-band radar Doppler moments and surface weather station variables',
                        'DATA_DISCIPLINE'  : 'Atmospheric Physics - Remote Sensing Radar Profiler',
                        'DATA_GROUP'       : 'Experimental;Profile;Moving',
                        'DATA_LOCATION'    : 'Research vessel Maria S. Merian - Atlantic Ocean',
                        'DATA_SOURCE'      : 'wband data postprocessed',
                        'DATA_PROCESSING'  : 'ship motion correction, the code used is available at https://github.com/ClauClouds/ship-motion-correction-for-EUREC4A-campaign',
                        'INSTRUMENT_MODEL' : '94 GHz (W-band) radar, manufactured by RPG GmbH',
                        'COMMENT'          : 'data from 11:00 to 13:59 UTC have been removed because collected in Barbados territorial waters' }
        
    else:
        # add comment on fall speed variable
        attrs = {'CREATED_BY'     : 'Claudia Acquistapace',
                    'CREATED_ON'       : str(datetime.now()),
                    'ORCID-AUTHOR'     : "Claudia Acquistapace: 0000-0002-1144-4753",
                    'DOI'              : '10.25326/235 (https://doi.org/10.25326/235)',
                    'FILL_VALUE'       : 'NaN',
                    'PI_NAME'          : 'Claudia Acquistapace',
                    'PI_AFFILIATION'   : 'University of Cologne (UNI), Germany',
                    'PI_ADDRESS'       : 'Institute for geophysics and meteorology, Pohligstrasse 3, 50969 Koeln',
                    'PI_MAIL'          : 'cacquist@meteo.uni-koeln.de',
                    'DO_NAME'          : 'University of Cologne - Germany',
                    'DO_AFFILIATION'   : 'University of Cologne - Germany',
                    'DO_address'       : 'Institute for geophysics and meteorology, Pohligstrasse 3, 50696 Koeln',
                    'DO_MAIL'          : 'cacquist@meteo.uni-koeln.de',
                    'DS_NAME'          : 'University of Cologne - Germany',
                    'DS_AFFILIATION'   : 'University of Cologne - Germany',
                    'DS_address'       : 'Institute for geophysics and meteorology, Pohligstrasse 3, 50696 Koeln',
                    'DS_MAIL'          : 'cacquist@meteo.uni-koeln.de',
                    'DATA_DESCRIPTION' : 'daily w-band radar Doppler moments and surface weather station variables',
                    'DATA_DISCIPLINE'  : 'Atmospheric Physics - Remote Sensing Radar Profiler',
                    'DATA_GROUP'       : 'Experimental;Profile;Moving',
                    'DATA_LOCATION'    : 'Research vessel Maria S. Merian - Atlantic Ocean',
                    'DATA_SOURCE'      : 'wband data postprocessed',
                    'DATA_PROCESSING'  : 'ship motion correction, the code used is available at https://github.com/ClauClouds/ship-motion-correction-for-EUREC4A-campaign',
                    'INSTRUMENT_MODEL' : '94 GHz (W-band) radar, manufactured by RPG GmbH',
                    'COMMENT'          : '' }
    
    
    # assign additional attributes following CF convention
    WbandData.attrs  = attrs
    # assign additional attributes following CF convention
    WbandData = WbandData.assign_attrs({
             "Conventions": "CF-1.8",
             "title": WbandData.attrs["DATA_DESCRIPTION"],
             "institution": WbandData.attrs["DS_AFFILIATION"],
             "history": "".join([
                 "source: " + WbandData.attrs["DATA_SOURCE"] + "\n",
                 "processing: " + WbandData.attrs["DATA_PROCESSING"] + "\n",
                 " adapted to enhance CF compatibility\n",
             ]),  # the idea of this attribute is that each applied transformation is appended to create something like a log
             "featureType": "trajectoryProfile",
         })
 
    # storing ncdf data
    WbandData.to_netcdf(path+'/corrected_comments/'+file_out, encoding={"radar_reflectivity":{"zlib":True, "complevel":9},\
                                                                                              "mean_doppler_velocity": {"dtype": "f4", "zlib": True, "complevel":9}, \
                                                                                              "spectral_width": {"zlib": True, "complevel":9}, \
                                                                                              "skewness": {"zlib": True, "complevel":9}, \
                                                                                              "lat": {"dtype": "f4"} , \
                                                                                              "lon": {"dtype": "f4"}, \
                                                                                              "time": {"units": "seconds since 2020-01-01", "dtype": "i4"}})
