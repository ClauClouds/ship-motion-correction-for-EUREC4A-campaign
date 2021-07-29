#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 14:32:18 2021
code to resample w band radar data on a common vertical grid
@author: claudia
"""
import numpy as np
import glob
import xarray as xr


fileListProcess_WBAND = np.sort(glob.glob('/Volumes/Extreme SSD/ship_motion_correction_merian/corrected_data/eurec4a_intake/*.nc'))
Nchars = len('/Volumes/Extreme SSD/ship_motion_correction_merian/corrected_data/eurec4a_intake/')


# read height sample
height_data = xr.open_dataset('/Volumes/Extreme SSD/ship_motion_correction_merian/corrected_data/eurec4a_intake/20200218_wband_radar_msm_eurec4a_intake.nc')
height_grid = height_data.height.values

for ind_file, filename in enumerate(fileListProcess_WBAND):
    
    print('file:', filename)
    #reading the data
    data = xr.open_dataset(filename)
    
    #regridding on height grid established
    if np.array_equal(data.height.values, height_grid) == False:
        print('data to be regridded:', filename[Nchars:Nchars+8])
        data_new = data.reindex(height=height_grid)
        data_new.to_netcdf('/Volumes/Extreme SSD/ship_motion_correction_merian/corrected_data/eurec4a_intake/cfad/'+filename[Nchars:])
    else:
        print('same height grid - skip ')