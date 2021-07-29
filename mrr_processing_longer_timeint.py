#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 31 13:06:39 2021

@author: claudia
"""
import numpy as np
import glob
import xarray as xr
import pandas as pd
# code to read 5 s int time data and resample them on 10 s resolution
path = '/Volumes/Extreme SSD/ship_motion_correction_merian/corrected_data/mrr_final/formatted/'
file_list = np.sort(glob.glob('/Volumes/Extreme SSD/ship_motion_correction_merian/corrected_data/mrr_final/formatted/*.nc'))


for indFile, file in enumerate(file_list):
    
    data = xr.open_dataset(file)
    file_out = file[len(path):]
    print(file_out)
    # add comment on fall speed variable
    data['fall_speed'].attrs['positive'] = 'down'
    data['fall_speed'].attrs['comment'] = 'fall speed positive towards the radar'
    
    print(data.fall_speed)
    
    data.to_netcdf('/Volumes/Extreme SSD/ship_motion_correction_merian/corrected_data/mrr_final/formatted/commented/'+file_out)
    