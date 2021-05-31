#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 31 13:06:39 2021

@author: claudia
"""

import xarray as xr
import pandas as pd
# code to read 5 s int time data and resample them on 10 s resolution

file = '/Volumes/Extreme SSD/ship_motion_correction_merian/mrr/second_step/20200120_150000_MRR-FromManufacturer-processed.nc'

data            = xr.open_dataset(file)
time            = data['time'].values


units_time      = 'seconds since 1970-01-01 00:00:00' 
datetimeM       = pd.to_datetime(time, unit ='s', origin='unix') 

data['time']    = datetimeM
datetime_10     = pd.date_range(start=datetimeM[0], end=datetimeM[-1], freq='10s')


data_10 = data.interp(time=datetime_10)
                V_windS_hour = V_windS_hour.interp(time=timeInterp) 

# call function to calculate time resolution and processing parameters for all dates 
timeRes, dates, navgArr, nnoise_arr = f_calcTimeResParam(DataList, pathRadar)