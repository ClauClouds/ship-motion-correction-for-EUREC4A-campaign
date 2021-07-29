#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon 19/04/2021

@author: cacquist
@date: 19.04.2021
@goal: calculate for essd paper
the mean max and min LWP daily values

"""
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib


# paths to the different data files and output directories for plots
path_input_data  = '/Volumes/Extreme SSD/ship_motion_correction_merian/corrected_data/daily_files_intake/daily_files/'
#pathFig         = '/Volumes/Extreme SSD/ship_motion_correction_merian/plots_essd/'


file_list = np.sort(glob.glob(path_input_data+'*.nc'))
print(file_list)

mean_lwp = []
median_lwp = []
max_lwp = []
min_lwp = []

for ind_file, filename in enumerate(file_list):
    # reading daily file
    ds = xr.open_dataset(filename)
    
    # reading LWP values
    lwp_serie = ds.liquid_water_path.values
    
    # calculating max, min, mean, median
    median_lwp.append(np.nanmedian(lwp_serie))
    mean_lwp.append(np.nanmean(lwp_serie))
    max_lwp.append(np.nanmax(lwp_serie))    
    min_lwp.append(np.nanmin(lwp_serie))
    
    print(np.nanmax(lwp_serie), np.nanmin(lwp_serie),  np.nanmean(lwp_serie))

#%%
print('mean')
print(np.array(mean_lwp).T)
print('****************')
print('max')
print(max_lwp)
print('****************')
print('min')
print(min_lwp)
print('****************')
print('median')
print(median_lwp)
print('****************')
