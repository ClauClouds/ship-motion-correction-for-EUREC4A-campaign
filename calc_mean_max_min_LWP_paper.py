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
import glob

# paths to the different data files and output directories for plots
path_input_data  = '/Volumes/Extreme SSD/ship_motion_correction_merian/corrected_data/wband_daily_with_DOI/latest/with_DOI/daily_intake/'
#pathFig         = '/Volumes/Extreme SSD/ship_motion_correction_merian/plots_essd/'


file_list = np.sort(glob.glob(path_input_data+'*.nc'))

# calculate mean ship speed
from functions_essd import f_haversineFormula

#(lat1_deg,lon1_deg,lat2_deg,lon2_deg)
#lat = data_merged['lat'].values
#lon = data_merged['lon'].values
#time = pd.to_datetime(data_merged['time'].values)

for indFile in range(len(file_list)):
    ship_speed = []
    data = xr.open_dataset(file_list[indFile])
    lat = data['lat'].values
    lon = data['lon'].values
    time = pd.to_datetime(data['time'].values)
    for ind in range(len(lat)-1):
        lat_1 = lat[ind]
        lat_2 = lat[ind+1]
        lon_1 = lon[ind]
        lon_2 = lon[ind+1]
        time_1 = time[ind]
        time_2 = time[ind+1]
        #print(f_haversineFormula(lat_1, lon_1, lat_2, lon_2))
        #print((time_2 - time_1).total_seconds())
        #print(f_haversineFormula(lat_1, lon_1, lat_2, lon_2)[0]/(time_2 - time_1).total_seconds())
        ship_speed.append(1000*f_haversineFormula(lat_1, lon_1, lat_2, lon_2)[0]/(time_2 - time_1).total_seconds())

    print(np.nanmean(ship_speed))
    print(np.nanmax(ship_speed))
    print(np.nanmin(ship_speed))
    print('***************************')

strasuka


data_merged = xr.open_mfdataset(file_list)
# = np.nanmean(data_merged.liquid_water_path.values)

LWP_median = np.nanmedian(data_merged.liquid_water_path.values)
LWP_std = np.nanstd(data_merged.liquid_water_path.values)
#print('LWP mean all days ', LWP_mean) #LWP mean all days  61.760258
print('LWP median all days ', LWP_median)#LWP median all days  2.6817865
print('LWP std all days ', LWP_std) #LWP std all days  320.44022
strasuka2
mean_lwp = []
mean_P = []
mean_T = []
mean_RR = []
mean_RH = []
mean_h_wind = []
mean_wind_dir = []

for ind_file, filename in enumerate(file_list):
    # reading daily file
    ds = xr.open_dataset(filename)

    # reading LWP values
    LWP_serie = ds.liquid_water_path.values
    P_serie = ds.air_pressure.values
    T_serie = ds.air_temperature.values
    RR_serie = ds.rain_rate.values
    WS_serie = ds.wind_speed.values
    WD_serie = ds.wind_direction.values
    RH_serie = ds.relative_humidity.values

    # calculating max, min, mean, median
    mean_lwp.append(np.nanmean(LWP_serie))
    mean_P.append(np.nanmean(P_serie))
    mean_T.append(np.nanmean(T_serie))
    mean_RR.append(np.nanmean(RR_serie))
    mean_RH.append(np.nanmean(RH_serie))
    mean_h_wind.append(np.nanmean(WS_serie))
    mean_wind_dir.append(np.nanmean(WD_serie))

#%%
print('mean')
print(np.array(mean_lwp).T)
print('****************')
print(np.array(mean_P).T)
print('****************')
print(np.array(mean_T).T)
print('****************')
print(np.array(mean_RR).T)
print('****************')
print(np.array(mean_RH).T)
print('****************')
print(np.array(mean_h_wind).T)
print('****************')
print(np.array(mean_wind_dir).T)
print('****************')
print('****************')
