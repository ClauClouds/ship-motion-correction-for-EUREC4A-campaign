#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 21 April 2021 15:51 CET
@ date; 21 April 2021
@author: Claudia Acquistapace
@goal: Read files from second step folder provided by Albert, do quality checks and
save them as ncdf files with CF conventions applied, as done for W-band radar data

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


# Preprocessing of the data compact.nc
def generate_preprocess(): # can take arguments

    '''
    author: Claudia Acquistapace
    date  : 29/10/2020
    goal  : generate a function that  does the preprocessing of the data
    '''
    import numpy as np
    import xarray as xr
    
    

    # Generate a preprocess function that takes a dataset as only argument and does all the processing needed
    def preprocess(ds):
        '''
        author: Claudia Acquistapace
        date : 21.04.2021, modified from version in eurec4a_intake_data_preparation.py
        goal : preprocess MRR ncdf data to make them readable with xr.open_fmdataset instruction
        The preprocessing includes:
            - removal of non useful variables :

            - redefinition of scalar variables as time series

        input:
            dataset xarray of data from radar

        output:
            new dataset with var2DropList variables removed and varList variables modified

        '''
        
        # redefining time coordinate 
        # reading time variables ( seconds since the beginning of the hour)
        #secs = ds['time'].values    
        
        # assign a new time coordinate defined as seconds since 
        #ds = ds.assign_coords({'time':[datetime_val + timedelta(seconds=int(n_sec)) for ind_sec, n_sec in enumerate(secs)]})

        # assign new coordinate for dependencies not specified
        #ds = ds.assign_coords({'scalar':np.arange(1)})

        # retrieving the list of variables of the dataset
        var2DropList = list(ds.keys())
        #print('vars to drop:', var2DropList)
        # escluding from the list of variables to remove the  moments to be plotted
        Var2keep = ['Dm', 'Kurtosis', 'LWC', 'N(D)',\
                    'RR', 'Skewness', 'spectral width', 'W', "Z", "Ze", "Zea", "mask"]
        for ind in range(len(Var2keep)):
            #print(Var2keep[ind])
            var2DropList.remove(Var2keep[ind])
        #print('variable list ', var2DropList)
        # drop variables that have no interest
        #var2DropList = ['source_rr', 'source_rh', 'source_ta', 'source_pa', 'source_wspeed', 'source_wdir', 'wl', 'nqv']

        for indDrop in range(len(var2DropList)):
            ds = ds.drop(var2DropList[indDrop])
        
        ds.drop('Dm_ax')
        ds.drop('Nw_ax')
        ds.drop('PIA_Height')
        ds.drop('3Range')
        ds.drop('DropSize')
        # Convert all variables with scalar dimension to time
        # list of variables to convert
        #varList = ['lat','lon','zsl','freq_sb','hpbw']
        #varUnits = ['degrees', 'degrees', 'm', 's-1', 'degrees']
        #varLongNames = ['latitude',
        #        'longitude',
        ##        'Altitude above mean sea level the instrument is located',
        #        'Central transmission frequency ',
        #        'Antenna half power beam width']

        #loop on variables to convert
        #for iVar in range(len(varList)):
        #    var = varList[iVar]
        #
        #    # storing single value of the variable
        #    varValue = ds[var].values
        #
        #    # deleting the variable from the dataset
        #    ds = ds.drop(var)#

            # creating new variable time serie
        #    varTimeSerie = np.repeat(varValue, len(ds['time'].values))#

            # saving variable in the dataset
        #    ds[var] = (['time'],varTimeSerie)
        #    dim     = ['time']
        #    coord   = {'time':ds['time'].values}
        #    VarDataArray = xr.DataArray(dims=dim, coords=coord, data=varTimeSerie,
        #                                  attrs={'long_name':varLongNames[iVar],
        #                                         'units':varUnits[iVar]})
            # adding new Var variable to the dataset
       #     ds = ds.assign({var:VarDataArray})

        #ds['time'] = pd.to_datetime(ds.time.values.astype(int).astype(str), format='%Y%m%d') + pd.to_timedelta(ds.time.values%1, unit='D').round('1s')
        return ds
    return preprocess


# reading ship data for lat/lon
shipFile = '/Volumes/Extreme SSD/ship_motion_correction_merian/ship_data/new/shipData_all2.nc'
shipData = xr.open_dataset(shipFile)

folders = ['20200127']
# days already processed '20200122' ,'20200123', '20200125', '20200126', '20200127', '20200128', '20200129', '20200130', '20200131',

for i_folder, fold in enumerate(folders):

    # read list of hourly files for the day corresponding to the folder
    fileList = np.sort(glob.glob('/Volumes/Extreme SSD/ship_motion_correction_merian/mrr/second_step/'+fold+'/*.nc'))
    
    # setting output path for final file
    path_out = '/Volumes/Extreme SSD/ship_motion_correction_merian/mrr/second_step/'
    
    # changing time coordinat and adding a proper coordinate
    for indDay, fileStr in enumerate(fileList):
        print(indDay, fileStr)
        lenstr = len('/Volumes/Extreme SSD/ship_motion_correction_merian/mrr/second_step/'+fold+'/')
        print('reading files for the hour: ', fileStr[lenstr:lenstr+11])
        file_test_out = '/Users/claudia/Downloads/test/'+fileStr[lenstr:]
        print(file_test_out)
        
        date_hour       = fileStr[lenstr:lenstr+11]
        yy              = date_hour[4:8]
        mm              = date_hour[2:4]
        dd              = date_hour[0:2]
        hh              = date_hour[9:11]

        # setting dates strings
        date            = dd+mm+yy      #'04022020'
        dateRadar       = yy[0:2]+mm+dd #'200204'
        dateReverse     = yy+mm+dd      #'20200204'

        datetime_val = pd.to_datetime(datetime(int(yy), int(mm), int(dd), int(hh), 0, 0))
    
        # reading data from the file
        ds = xr.open_dataset(fileStr)
        secs = ds['time'].values        
        # rename time as time old
        #ds = ds.rename({'time': 'time_old'})
        
        # define new time coordinate
        ds = ds.assign_coords({'time':[datetime_val + timedelta(seconds=int(n_sec)) for ind_sec, n_sec in enumerate(secs)]})

        ds.to_netcdf('/Volumes/Extreme SSD/ship_motion_correction_merian/mrr/second_step/'+fold+'/temp/'+dateReverse+'_'+hh+'_mrr.nc')
            
    
    fileListTest = np.sort(glob.glob('/Volumes/Extreme SSD/ship_motion_correction_merian/mrr/second_step/'+fold+'/temp/*.nc'))
    
   
    # creating daily file for the dataset
    #print('processing day : '+yy+'/'+mm+'/'+dd)
    print("*************************************************")
    day_data = xr.open_mfdataset(fileListTest,
                                   concat_dim = 'time_utc',
                                   data_vars = 'minimal',
                                   preprocess = generate_preprocess(),
                                  )
    #removing wrong time coordinate
    day_data = day_data.drop('time')
    
    # renaming as time the time_utc coordinate
    day_data = day_data.rename({'time_utc':'time'})

    # filtering data using the interference noise mask developed 
    day_data = day_data.where(day_data.mask !=  0)
    
    # defining a new mask to filter interference present in lowest 200 m
    ds = day_data.sel(Height=slice(0,200.))
    
    # defining the new mask as zeros/ones based on Ze ==np.nan/!=np.nan
    newmask = np.zeros((len(ds.time.values), len(ds.Height.values)))
    newmask[~np.isnan(ds.Zea.values)] = 1
    
    # calculating sum of mask elements in the lowest 200 m
    sumMask = np.sum(newmask, axis=1)
    
    # when the sum of the  number of pixels where mask == 1 is smaller than 10, 
    # we consider that as interference. The threshold is taken considering that 
    # in 200 m we have 19 range gates, hence we ask that at least half of them have 
    # data in it, for the signal to be not noise
    day_data.mask.values[sumMask < 10,:] = np.nan
    mask_new = day_data.mask.values
    mask_new[sumMask < 10,:] = np.nan
    mask_new[:, day_data.Height.values>=200.] = day_data.mask.values[:, day_data.Height.values>=200.]
    
    # assign the new mask as a dataset variable
    day_data = day_data.assign({"mask2": (('time','Height'), mask_new)})
    
    # filter all data points where the mask is nan (keeping all non nans)
    day_data = day_data.where(~np.isnan(day_data.mask2))

    # selecting ship data corresponding to the selected day
    ship_data_day = shipData.sel(time=slice(pd.to_datetime(day_data.time.values[0]), pd.to_datetime(day_data.time.values[-1])))
    
    # interpolating ship data on radar time stamps
    ship_day_interp = ship_data_day.interp(time=day_data.time.values, method='nearest')
    

    # reading lat/lon for the trajectory
    lat_day = ship_day_interp['lat'].values
    lon_day = ship_day_interp['lon'].values
    
    
    # saving the data in CF compliant conventions
    MRRdata = xr.Dataset(
        data_vars={
            "fall_speed": (('time','height'), day_data.W.values, {'long_name': 'hydrometeor fall speed alias corrected', 'units':'m s-1'}),
            'skewness': (('time','height'), day_data.Skewness.values, {'long_name': 'skewness', 'units':'none'}),
            'Kurtosis':(('time','height'), day_data.Kurtosis.values, {'long_name': 'kurtosis', 'units':'none'}),
            'liquid_water_content':(('time','height'), day_data.LWC.values, {'long_name': 'liquid water content', "standard_name": "atmosphere_mass_content_of_cloud_liquid_water", 'units':'g m-3'}),
            'rain_rate':(('time','height'), day_data.RR.values, {'long_name': 'rainfall rate', "standard_name":'rainfall_rate', 'units':'mm h-1'}),
            'Z':(('time','height'), day_data.Z.values, {'long_name': 'reflectivity considering only liquid drops', 'units':'dBz'}),
            'Ze':(('time','height'), day_data.Ze.values, {'long_name': 'equivalent reflectivity non attenuated', 'units':'dBz'}),
            'Zea':(('time','height'), day_data.Zea.values, {'long_name': 'equivalent reflectivity attenuated', 'units':'dBz'}),
            'drop_size_distribution':(('time','height'), day_data['N(D)'].values, {'long_name': 'rain drop size distribution', 'units':'log10(m -3 mm -1)'}),
            'mean_mass_weigthed_raindrop_diameter':(('time','height'), day_data['Dm'].values,{'long_name': 'mean mass weighted raindrop diameter', 'units':'mm'}),
            'spectral_width':(('time','height'), day_data['spectral width'].values, {'long_name': 'spectral width', 'units':'m s-1'}),
        },  
        coords={
            "time": (('time',), day_data['time'].values, {"axis": "T","standard_name": "time"}), # leave units intentionally blank, to be defined in the encoding
            "height": (('height',), day_data.Height.values, {"axis": "Z","positive": "up","units": "m", "long_name":'radar range height'}),
            "lat": (('time',), lat_day, {"axis": "Y", "standard_name": "latitude", "units": "degree_north"}),
            "lon": (('time',), lon_day, { "axis": "X", "standard_name": "longitude","units": "degree_east"}),
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
    MRRdata.to_netcdf(path_out+dateReverse+'_MRR_PRO_msm_eurec4a.nc', encoding={"Z":{"zlib":True, "complevel":9},\
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
