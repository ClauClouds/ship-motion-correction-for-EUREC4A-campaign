#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 10:09:11 2021

@author: claudia
"""


# importing necessary libraries
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import rcParams
import matplotlib
import netCDF4 as nc4
from netCDF4 import Dataset
import glob
import os.path
import pandas as pd
import numpy as np
import xarray as xr
import scipy.integrate as integrate
from datetime import datetime
from datetime import timedelta
from pathlib import Path
from scipy.signal import chirp, find_peaks, peak_widths, peak_prominences

data_10 = xr.open_dataset('/Volumes/Extreme SSD/ship_motion_correction_merian/corrected_data/mrr_final/20200122_MRR_PRO_msm_eurec4a.nc')
path_out = '/Volumes/Extreme SSD/ship_motion_correction_merian/corrected_data/mrr_final/formatted/'

# defining height array for completing the matrix that has less heights 
height_above = np.arange(start=660., stop=1300., step=10.)
height_below =np.arange(start=0., stop=20., step=10.)



# reading file list of data
file_list = np.sort(glob.glob('/Volumes/Extreme SSD/ship_motion_correction_merian/corrected_data/mrr_final/*.nc'))


# loop on files
for ind_file, filename in enumerate(file_list):
    
    #defining proper time grid for the day
    yy = int(filename[-31:][0:4])
    mm = int(filename[-31:][4:6])
    dd = int(filename[-31:][6:8])
    
    if len(dd) == 1:
        dd = '0'+dd
        
    date = str(yy)+'0'+str(mm)+str(dd)
    print('processing day ', date)
    
    
    # read file and time and height coordinate
    data = xr.open_dataset(filename)
    height = data.height.values
    time = data.time.values
    # defining nan matrix to fill matrices 
    nan_matrix_below =  np.empty((len(time),len(height_below)))
    nan_matrix_below[:]= np.nan
    nan_matrix_above =  np.empty((len(time),len(height_above)))
    nan_matrix_above[:]= np.nan
    
    # defining nan dataset to fill matrices with higher heights
    empty_dataset_above = xr.Dataset(
            data_vars = dict(
               fall_speed = (['time', 'height'], nan_matrix_above),
               skewness = (['time', 'height'], nan_matrix_above),
               liquid_water_content = (['time'], np.repeat(np.nan, len(time))),
               rain_rate = (['time', 'height'], nan_matrix_above),
               Z = (['time', 'height'],nan_matrix_above),
               Ze = (['time', 'height'], nan_matrix_above),
               Zea = (['time', 'height'], nan_matrix_above),
               drop_size_distribution = (['time', 'height'], nan_matrix_above),
               mean_mass_weigthed_raindrop_diameter = (['time', 'height'], nan_matrix_above),
               spectral_width = (['time', 'height'], nan_matrix_above),
               Kurtosis = (['time', 'height'], nan_matrix_above),
        ), 
            coords=dict(
             time=time,
             height=height_above,
         ),
    )
    
    # assign istrument id
    instrument_id = xr.DataArray("msm_mrr_pro",dims=(),attrs={"cf_role": "trajectory_id"},)
    empty_dataset_above = empty_dataset_above.assign({"instrument": instrument_id,})
 
    # defining nan dataset to fill matrices with lower heights
    empty_dataset_below = xr.Dataset(
            data_vars = dict(
               fall_speed = (['time', 'height'], nan_matrix_below),
               skewness = (['time', 'height'], nan_matrix_below),
               liquid_water_content = (['time'], np.repeat(np.nan, len(time))),
               rain_rate = (['time', 'height'], nan_matrix_below),
               Z = (['time', 'height'],nan_matrix_below),
               Ze = (['time', 'height'], nan_matrix_below),
               Zea = (['time', 'height'], nan_matrix_below),
               drop_size_distribution = (['time', 'height'], nan_matrix_below),
               mean_mass_weigthed_raindrop_diameter = (['time', 'height'], nan_matrix_below),
               spectral_width = (['time', 'height'], nan_matrix_below),
               Kurtosis = (['time', 'height'], nan_matrix_below),
        ), 
            coords=dict(
             time=time,
             height=height_below,
         ),
    )
    
    # assign istrument id
    instrument_id = xr.DataArray("msm_mrr_pro",dims=(),attrs={"cf_role": "trajectory_id"},)
    empty_dataset_below = empty_dataset_below.assign({"instrument": instrument_id,})
            

    if (height[0] !=0.) * (height[-1]!= 1290.):
        # merging nans above and below
        print('* adding nans below and above')
        data = xr.merge([empty_dataset_below, data,empty_dataset_above])
    elif (height[0] ==0.) * (height[-1]!= 1290.):
        print('*  adding nans above')
        data = xr.merge([data,empty_dataset_above])
    elif (height[0] !=0.) * (height[-1] == 1290.):
        print('*  adding nans below')
        data = xr.merge([empty_dataset_below, data])
    else:
        print('height of the correct shape')
        
            
    # resampling data on a time grid of 1 s
    print('* resampling now on time resolution of 1s')
    time_grid = pd.date_range(start=datetime(yy,mm,dd,0,0,0), end=datetime(yy,mm,dd,23,59,59), freq='1s')
    
    # resample in time
    data_final = data.reindex(time=time_grid, method='nearest')
    print('* saving data to ncdf')
    # saving data final on output ncdf file
    MRRdata = xr.Dataset(
        data_vars={
            "fall_speed": (('time','height'), data_final.fall_speed.values, {'long_name': 'hydrometeor fall speed alias corrected', 'units':'m s-1'}),
            'skewness': (('time','height'), data_final.skewness.values, {'long_name': 'skewness', 'units':'none'}),
            'Kurtosis':(('time','height'), data_final.Kurtosis.values, {'long_name': 'kurtosis', 'units':'none'}),
            'liquid_water_content':(('time','height'), data_final.liquid_water_content.values, {'long_name': 'liquid water content', "standard_name": "atmosphere_mass_content_of_cloud_liquid_water", 'units':'g m-3'}),
            'rain_rate':(('time','height'), data_final.rain_rate.values, {'long_name': 'rainfall rate', "standard_name":'rainfall_rate', 'units':'mm h-1'}),
            'Z':(('time','height'), data_final.Z.values, {'long_name': 'reflectivity considering only liquid drops', 'units':'dBz'}),
            'Ze':(('time','height'), data_final.Ze.values, {'long_name': 'equivalent reflectivity non attenuated', 'units':'dBz'}),
            'Zea':(('time','height'), data_final.Zea.values, {'long_name': 'equivalent reflectivity attenuated', 'units':'dBz'}),
            'drop_size_distribution':(('time','height'), data_final.drop_size_distribution.values, {'long_name': 'rain drop size distribution', 'units':'log10(m -3 mm -1)'}),
            'mean_mass_weigthed_raindrop_diameter':(('time','height'), data_final.mean_mass_weigthed_raindrop_diameter.values,{'long_name': 'mean mass weighted raindrop diameter', 'units':'mm'}),
            'spectral_width':(('time','height'), data_final.spectral_width.values, {'long_name': 'spectral width', 'units':'m s-1'}),
        },  
        coords={
            "time": (('time',), data_final.time.values, {"axis": "T","standard_name": "time"}), # leave units intentionally blank, to be defined in the encoding
            "height": (('height',), data_final.height.values, {"axis": "Z","positive": "up","units": "m", "long_name":'radar range height'}),
            "lat": (('time',), data_final.lat.values, {"axis": "Y", "standard_name": "latitude", "units": "degree_north"}),
            "lon": (('time',), data_final.lon.values, { "axis": "X", "standard_name": "longitude","units": "degree_east"}),
        },  
        attrs={'CREATED_BY'     : 'Claudia Acquistapace and Albert Garcia Benadi',
                        'ORCID-AUTHORS'   : "Claudia Acquistapace: 0000-0002-1144-4753, Albert Garcia Benadi : 0000-0002-5560-4392", 
                        'CREATED_ON'       : str(datetime.now()),
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
                        'DATA_DESCRIPTION' : 'daily MRR measurements on Maria S. Merian (msm) ship during EUREC4A campaign',
                        'DATA_DISCIPLINE'  : 'Atmospheric Physics - Remote Sensing Radar Profiler',
                        'DATA_GROUP'       : 'Experimental;Profile;Moving',
                        'DATA_LOCATION'    : 'Research vessel Maria S. Merian - Atlantic Ocean',
                        'DATA_SOURCE'      : 'MRR-PRO data postprocessed',
                        'DATA_PROCESSING'  : 'ship motion correction and filtering of interference the code used is available at https://github.com/ClauClouds/ship-motion-correction-for-EUREC4A-campaign',
                        'INSTRUMENT_MODEL' : 'MRR PRO (24 Ghz radar)',
                        'COMMENT'          : 'The MRR pro belongs to Jun. Prof. Heike Kalesse, University of Leipzig (DE)' }
    )
    
    
    
    # assign istrument id
    instrument_id = xr.DataArray("msm_mrr_pro",dims=(),attrs={"cf_role": "trajectory_id"},)
    MRRdata = MRRdata.assign({"instrument": instrument_id,})
    
    # assign additional attributes following CF convention
    MRRdata = MRRdata.assign_attrs({
            "Conventions": "CF-1.8",
            "title": MRRdata.attrs["DATA_DESCRIPTION"],
            "institution": MRRdata.attrs["DS_AFFILIATION"],
            "history": "".join([
                "source: " + MRRdata.attrs["DATA_SOURCE"] + "\n",
                "processing: " + MRRdata.attrs["DATA_PROCESSING"] + "\n",
                "postprocessing with de-aliasing developed by Albert Garcia Benadi " + '\n', 
                "adapted to enhance CF compatibility\n",
            ]),  # the idea of this attribute is that each applied transformation is appended to create something like a log
            "featureType": "trajectoryProfile",
        })
    
    # storing ncdf data
    MRRdata.to_netcdf(path_out+date+'_MRR_PRO_msm_eurec4a.nc', encoding={"Z":{"zlib":True, "complevel":9},\
                                                                                             "Ze": {"dtype": "f4", "zlib": True, "complevel":9}, \
                                                                                             "Zea": {"zlib": True, "complevel":9}, \
                                                                                             "drop_size_distribution": {"zlib": True, "complevel":9}, \
                                                                                             "liquid_water_content": {"zlib": True, "complevel":9}, \
                                                                                             "Kurtosis": {"zlib": True, "complevel":9}, \
                                                                                             "fall_speed": {"zlib": True, "complevel":9}, \
                                                                                             "skewness": {"zlib": True, "complevel":9}, \
                                                                                             "mean_mass_weigthed_raindrop_diameter": {"zlib": True, "complevel":9}, \
                                                                                             "spectral_width": {"zlib": True, "complevel":9}, \
                                                                                             "rain_rate": {"zlib": True, "complevel":9}, \
                                                                                             "lat": {"dtype": "f4"} , \
                                                                                             "lon": {"dtype": "f4"}, \
                                                                                             "time": {"units": "seconds since 2020-01-01", "dtype": "i4"}})

#%%

data_merged.Ze.plot()
#%%
file_list = np.sort(glob.glob('/Volumes/Extreme SSD/ship_motion_correction_merian/corrected_data/mrr_final/*.nc'))

data_1 = xr.open_dataset(file_list[15])
data_resized = data_10.reindex_like(data_1)