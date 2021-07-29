#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 09:43:33 2021

@author: claudia
"""
import numpy as np
import glob
import xarray as xr

path_files = '/Volumes/Extreme SSD/ship_motion_correction_merian/22_merge_test/'
file_list = np.sort(glob.glob(path_files+'*.nc'))



for ind_file, filename in enumerate(file_list):
    
    print(filename)
    
    # reading new file
    data = xr.open_dataset(filename) 
    print(data)
    print("*************************************************************")
    #if ind_file == 0:
    #    data_merged = data
    #else:
    #    data_merged = xr.merge([data_merged, data], combine_attrs='drop')