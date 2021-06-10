#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 10:51:08 2021

@author: claudia
"""
import os.path
import pandas as pd
import netCDF4 as nc4
from netCDF4 import Dataset
import xarray as xr
import numpy as np
from functions_essd import f_interpShipData
from functions_essd import lcl
from functions_essd import f_readShipDataset
from datetime import datetime
from datetime import timedelta
import glob
# reading ship data for the entire campaign
# paths to the different data files and output directories for plots
path_input_data  = '/Volumes/Extreme SSD/ship_motion_correction_merian/corrected_data/eurec4a_intake/'
pathFig         = '/Volumes/Extreme SSD/ship_motion_correction_merian/plots_essd/'



# set up variables of weather station 
file_list = np.sort(glob.glob(path_input_data+'*.nc'))
print(file_list)
n_files = len(file_list)


n_times = 28800

# defining variables of interest
P = np.zeros((n_times, n_files))
T_air = np.zeros((n_times, n_files))
RH_air = np.zeros((n_times, n_files))
rr = np.zeros((n_times, n_files))
wind_speed = np.zeros((n_times, n_files))
lcl_matrix = np.zeros((n_times, n_files))



# reading files in a loop
for indfile, filename in enumerate(file_list):
    
    print('processing ', yy,mm,dd)
    data = xr.open_dataset(filename)
    time_data = data.time.values
    
    # defining grid time array of 3s resolution
    yy = pd.to_datetime(time_data[0]).year
    mm = pd.to_datetime(time_data[0]).month
    dd = pd.to_datetime(time_data[0]).day
    time_grid = pd.date_range(start= datetime(yy,mm,dd,0,0,0), end=datetime(yy,mm,dd,23,59,59), freq='3s')
    
    # interpolating data on the fixed time grid
    if (len(time_data) != len(time_grid)):
        
        data = data.interp(time=time_grid)
        
        
        # copying variables P, T, in the corresponding matrix line
        P[:,indfile] = data.air_pressure.values
        T_air[:,indfile] = data.air_temperature.values
        RH_air[:,indfile] = data.relative_humidity.values
        wind_speed[:,indfile] = data.wind_speed.values
        rr[:,indfile] = data.rain_rate.values
        
    
        # calculating lcl 
        P_serie = data.air_pressure.values * 100. #pa
        T_serie = data.air_temperature.values
        RH_serie = data.relative_humidity.values
        
        # selecting data from nans
        i_valid = np.where((~np.isnan(P_serie)) * (~np.isnan(T_serie)) * (~np.isnan(RH_serie)))
        
        # vectorization of the function to calculate lcl
        vecDist = np.vectorize(lcl)
        # calculation of lcl for the whole dataset
        lcl_matrix[i_valid,indfile] =  vecDist(P_serie[i_valid], T_serie[i_valid], RH_serie[i_valid])
        print('lcl calculated')

    else:
        
        # copying variables P, T, in the corresponding matrix line
        P[:,indfile] = data.air_pressure.values
        T_air[:,indfile] = data.air_temperature.values
        RH_air[:,indfile] = data.relative_humidity.values
        wind_speed[:,indfile] = data.wind_speed.values
        rr[:,indfile] = data.rain_rate.values
        
        
        # calculating lcl 
        P_serie = data.air_pressure.values * 100. #pa
        T_serie = data.air_temperature.values
        RH_serie = data.relative_humidity.values
        
        # selecting data from nans
        i_valid = np.where((~np.isnan(P_serie)) * (~np.isnan(T_serie)) * (~np.isnan(RH_serie)))
        
        # vectorization of the function to calculate lcl
        vecDist = np.vectorize(lcl)
        # calculation of lcl for the whole dataset
        lcl_matrix[i_valid,indfile] =  vecDist(P_serie[i_valid], T_serie[i_valid], RH_serie[i_valid])
        
#%%

# setting to nan all rain rate values that are zero.
rr[rr ==0.] = np.nan
# store dat
from matplotlib import rcParams
import matplotlib
import os.path
import pandas as pd
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
# calculating stats
P_mean = np.nanmedian(P, axis=1)
P_std = np.nanstd(P, axis=1)
T_mean = np.nanmedian(T_air, axis=1)
T_std = np.nanstd(T_air, axis=1)
RH_mean = np.nanmedian(RH_air, axis=1)
RH_std = np.nanstd(RH_air, axis=1)
RR_mean = np.nanmedian(rr, axis=1)
RR_std = np.nanstd(rr, axis=1)
LCL_mean = np.nanmedian(lcl_matrix, axis=1)
LCL_std = np.nanstd(lcl_matrix, axis=1)
wind_speed_mean = np.nanmedian(wind_speed, axis=1)
wind_speed_std = np.nanstd(wind_speed, axis=1)
#%%
# plotting variables
fig, axs = plt.subplots(3, 2, figsize=(24,14), sharex=True, constrained_layout=True)
from matplotlib.path import Path
import matplotlib.patches as patches
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
# setting dates formatter 
matplotlib.rc('xtick', labelsize=22)  # sets dimension of ticks in the plots
matplotlib.rc('ytick', labelsize=22)  # sets dimension of ticks in the plots
rcParams['font.sans-serif'] = ['Tahoma']
fontSizeTitle = 22
fontSizeX = 22
fontSizeY = 22

timeLocal = pd.to_datetime(time_grid)
timeStartDay = timeLocal[0]
timeEndDay = timeLocal[-1]
[a.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M')) for a in axs[:].flatten()]


axs[0,0].spines["top"].set_visible(False)  
axs[0,0].spines["right"].set_visible(False)  
#axs[0,0].get_xaxis().tick_bottom()  
#axs[0,0].get_yaxis().tick_left() 
axs[0,0].xaxis.set_major_formatter(mdates.DateFormatter("%H")) # set the label format
axs[0,0].xaxis.set_minor_formatter(mdates.DateFormatter(" "))     # empty prints no label
major_ticks = np.arange(timeStartDay, timeEndDay, 7200000000, dtype='datetime64')
minor_ticks = np.arange(timeStartDay, timeEndDay, 900000000, dtype='datetime64')
axs[0,0].tick_params(which = 'major', direction = 'out')
axs[0,0].set_xticks(minor_ticks, minor=True)
axs[0,0].set_xticks(major_ticks)
axs[0,0].grid(which='minor', alpha=0.2)
axs[0,0].grid(which='major', alpha=0.5)
axs[2,0].xaxis.set_major_locator(MultipleLocator(4))

axs[0,0].xaxis_date()
axs[0,0].plot(time_grid, LCL_mean, color='red')
axs[0,0].fill_between(time_grid, LCL_mean-LCL_std, LCL_mean+LCL_std, alpha=0.2, color='red')
axs[0,0].axvline(x=pd.to_datetime(datetime(yy,mm,dd,10,30,0,0)), color='black',linewidth=2, linestyle='dashed')
axs[0,0].axvline(x=pd.to_datetime(datetime(yy,mm,dd,23,15,0,0)), color='black', linewidth=2, linestyle=':')
axs[0,0].set_ylim(500.,1100.)                                               # limits of the y-axesn  cmap=plt.cm.get_cmap("viridis", 256)
axs[0,0].set_xlim(timeStartDay, timeEndDay)                                 # limits of the x-axes
#axs[0,0].set_title('diurnal cycle of lifting condensation level (LCL)', fontsize=fontSizeTitle, loc='left')
axs[0,0].set_ylabel('height [m]', fontsize=fontSizeY)

axs[1,0].spines["top"].set_visible(False)  
axs[1,0].spines["right"].set_visible(False)  
#axs[0,0].get_xaxis().tick_bottom()  
#axs[0,0].get_yaxis().tick_left() 
axs[1,0].xaxis.set_major_formatter(mdates.DateFormatter("%H")) # set the label format
axs[1,0].xaxis.set_minor_formatter(mdates.DateFormatter(" "))     # empty prints no label
major_ticks = np.arange(timeStartDay, timeEndDay, 7200000000, dtype='datetime64')
minor_ticks = np.arange(timeStartDay, timeEndDay, 900000000, dtype='datetime64')
axs[1,0].tick_params(which = 'major', direction = 'out')
axs[1,0].set_xticks(minor_ticks, minor=True)
axs[1,0].set_xticks(major_ticks)
axs[1,0].grid(which='minor', alpha=0.2)
axs[1,0].grid(which='major', alpha=0.5)
axs[1,0].xaxis_date()
axs[1,0].plot(time_grid, T_mean-273.15, color='black')
axs[1,0].fill_between(time_grid, T_mean-273.15-T_std, T_mean-273.15+T_std, alpha=0.2, color='black')
axs[1,0].axvline(x=pd.to_datetime(datetime(yy,mm,dd,10,30,0,0)), color='black',linewidth=2, linestyle='dashed')
axs[1,0].axvline(x=pd.to_datetime(datetime(yy,mm,dd,23,15,0,0)), color='black', linewidth=2, linestyle=':')
axs[1,0].set_ylim(25.,29.)                                               # limits of the y-axesn  cmap=plt.cm.get_cmap("viridis", 256)
axs[1,0].set_xlim(timeStartDay, timeEndDay)                                 # limits of the x-axes
#axs[1,0].set_title('diurnal cycle of surface air temperature', fontsize=fontSizeTitle, loc='left')
axs[1,0].set_ylabel('temperature [$^{\circ} C$]', fontsize=fontSizeY)


axs[2,0].spines["top"].set_visible(False)  
axs[2,0].spines["right"].set_visible(False)  
#axs[0,0].get_xaxis().tick_bottom()  
#axs[0,0].get_yaxis().tick_left() 
axs[2,0].xaxis.set_major_formatter(mdates.DateFormatter("%H")) # set the label format
axs[2,0].xaxis.set_minor_formatter(mdates.DateFormatter(" "))     # empty prints no label
major_ticks = np.arange(timeStartDay, timeEndDay, 7200000000, dtype='datetime64')
minor_ticks = np.arange(timeStartDay, timeEndDay, 900000000, dtype='datetime64')
axs[2,0].tick_params(which = 'major', direction = 'out')
axs[2,0].set_xticks(minor_ticks, minor=True)
axs[2,0].set_xticks(major_ticks)
axs[2,0].grid(which='minor', alpha=0.2)
axs[2,0].grid(which='major', alpha=0.5)
axs[2,0].xaxis_date()
axs[2,0].plot(time_grid, RH_mean*100, color='blue')
axs[2,0].fill_between(time_grid, RH_mean*100-RH_std*100, RH_mean*100+RH_std*100, alpha=0.2, color='blue')
axs[2,0].axvline(x=pd.to_datetime(datetime(yy,mm,dd,10,30,0,0)), color='black',linewidth=2, linestyle='dashed', label='sunrise')
axs[2,0].axvline(x=pd.to_datetime(datetime(yy,mm,dd,23,15,0,0)), color='black', linewidth=2, linestyle='dotted', label='sunset')
axs[2,0].set_ylim(60.,80.)                                               # limits of the y-axesn  cmap=plt.cm.get_cmap("viridis", 256)
axs[2,0].set_xlim(timeStartDay, timeEndDay)                                 # limits of the x-axes
#axs[2,0].set_title('diurnal cycle of surface relative humidity', fontsize=fontSizeTitle, loc='left')
axs[2,0].legend(frameon=False, fontsize=20)
axs[2,0].set_ylabel('relative humidity [$\%$]', fontsize=fontSizeY)
axs[2,0].set_xlabel("time UTC (LT +4h) [hh]", fontsize=fontSizeX)

####################################
axs[0,1].spines["top"].set_visible(False)  
axs[0,1].spines["right"].set_visible(False)  
#axs[0,0].get_xaxis().tick_bottom()  
#axs[0,0].get_yaxis().tick_left() 
axs[0,1].xaxis.set_major_formatter(mdates.DateFormatter("%H")) # set the label format
axs[0,1].xaxis.set_minor_formatter(mdates.DateFormatter(" "))     # empty prints no label
major_ticks = np.arange(timeStartDay, timeEndDay, 7200000000, dtype='datetime64')
minor_ticks = np.arange(timeStartDay, timeEndDay, 900000000, dtype='datetime64')
axs[0,1].tick_params(which = 'major', direction = 'out')
axs[0,1].set_xticks(minor_ticks, minor=True)
axs[0,1].set_xticks(major_ticks)
axs[0,1].grid(which='minor', alpha=0.2)
axs[0,1].grid(which='major', alpha=0.5)
axs[0,1].xaxis_date()
axs[0,1].plot(time_grid, RR_mean, color='purple')
axs[0,1].fill_between(time_grid, RR_mean-RR_std, RR_mean+RR_std, alpha=0.2, color='purple')
axs[0,1].axvline(x=pd.to_datetime(datetime(yy,mm,dd,10,30,0,0)), color='black',linewidth=2, linestyle='dashed')
axs[0,1].axvline(x=pd.to_datetime(datetime(yy,mm,dd,23,15,0,0)), color='black', linewidth=2, linestyle=':')
axs[0,1].set_ylim(0.,40.)                                               # limits of the y-axesn  cmap=plt.cm.get_cmap("viridis", 256)
axs[0,1].set_xlim(timeStartDay, timeEndDay)                                 # limits of the x-axes
#axs[0,1].set_title('diurnal cycle of surface rain rate ', fontsize=fontSizeTitle, loc='left')
axs[0,1].set_ylabel('rain rate [mmh$^{-1}$]', fontsize=fontSizeY)



axs[1,1].spines["top"].set_visible(False)  
axs[1,1].spines["right"].set_visible(False)  
#axs[0,0].get_xaxis().tick_bottom()  
#axs[0,0].get_yaxis().tick_left() 
axs[1,1].xaxis.set_major_formatter(mdates.DateFormatter("%H")) # set the label format
axs[1,1].xaxis.set_minor_formatter(mdates.DateFormatter(" "))     # empty prints no label
major_ticks = np.arange(timeStartDay, timeEndDay, 7200000000, dtype='datetime64')
minor_ticks = np.arange(timeStartDay, timeEndDay, 900000000, dtype='datetime64')
axs[1,1].tick_params(which = 'major', direction = 'out')
axs[1,1].set_xticks(minor_ticks, minor=True)
axs[1,1].set_xticks(major_ticks)
axs[1,1].grid(which='minor', alpha=0.2)
axs[1,1].grid(which='major', alpha=0.5)
axs[1,1].xaxis_date()
axs[1,1].plot(time_grid, wind_speed_mean, color='green')
axs[1,1].fill_between(time_grid, wind_speed_mean-wind_speed_std, wind_speed_mean+wind_speed_std, alpha=0.2, color='green')
axs[1,1].axvline(x=pd.to_datetime(datetime(yy,mm,dd,10,30,0,0)), color='black',linewidth=2, linestyle='dashed')
axs[1,1].axvline(x=pd.to_datetime(datetime(yy,mm,dd,23,15,0,0)), color='black', linewidth=2, linestyle=':')
axs[1,1].set_ylim(0.,40.)                                               # limits of the y-axesn  cmap=plt.cm.get_cmap("viridis", 256)
axs[1,1].set_xlim(timeStartDay, timeEndDay)                                 # limits of the x-axes
#axs[1,1].set_title('diurnal cycle of surface horizontal wind speed', fontsize=fontSizeTitle, loc='left')
axs[1,1].set_ylabel('wind speed [$ms^{-1}$]', fontsize=fontSizeY)


axs[2,1].spines["top"].set_visible(False)  
axs[2,1].spines["right"].set_visible(False)  
axs[2,1].xaxis.set_major_formatter(mdates.DateFormatter("%H")) # set the label format
axs[2,1].xaxis.set_minor_formatter(mdates.DateFormatter(" "))     # empty prints no label
major_ticks = np.arange(timeStartDay, timeEndDay, 7200000000, dtype='datetime64')
minor_ticks = np.arange(timeStartDay, timeEndDay, 900000000, dtype='datetime64')
axs[2,1].tick_params(which = 'major', direction = 'out')
axs[2,1].set_xticks(minor_ticks, minor=True)
axs[2,1].set_xticks(major_ticks)
axs[2,1].grid(which='minor', alpha=0.2)
axs[2,1].grid(which='major', alpha=0.5)
axs[2,1].xaxis_date()
axs[2,1].plot(time_grid, P_mean, color='orange')
axs[2,1].fill_between(time_grid, P_mean-P_std, P_mean+P_std, alpha=0.2, color='orange')
axs[2,1].axvline(x=pd.to_datetime(datetime(yy,mm,dd,10,30,0,0)), color='black',linewidth=2, linestyle='dashed', label='sunrise')
axs[2,1].axvline(x=pd.to_datetime(datetime(yy,mm,dd,23,15,0,0)), color='black', linewidth=2, linestyle='dotted', label='sunset')
axs[2,1].set_ylim(1010.,1016.)                                               # limits of the y-axesn  cmap=plt.cm.get_cmap("viridis", 256)
axs[2,1].set_xlim(timeStartDay, timeEndDay)                                 # limits of the x-axes
#axs[2,1].set_title('diurnal cycle of surface pressure', fontsize=fontSizeTitle, loc='left')
axs[2,1].set_ylabel('surface pressure [$Hpa$]', fontsize=fontSizeY)
axs[2,1].set_xlabel("time UTC (LT +4h) [hh]", fontsize=fontSizeX)
for ax, l in zip(axs.flatten(), ['(a) Lifting condensation level (LCL)','(b) Surface rain rate', '(c) Surface air temperature', '(d) Surface horizontal wind speed','(e) Surface relative humidity', '(f) Surface pressure']):
    ax.text(-0.05, 1.05, l,  fontweight='black', fontsize=20, transform=ax.transAxes)
fig.savefig(pathFig+'Figure_diurnal_cycle_surface_vars.png')

#%%
grid            = True
axs[0,0].plot(time_grid pd.Series(profile_mrr).rolling(window=N).mean().iloc[N-1:].values, color='white', linestyle='dotted', linewidth=2, label='MRR highest signal')
axs[0,0].plot(timeLocal, lcl, color='black', label='Lifting condensation level')
#axs[0,0].plot(time_mrr, profile_mrr, color='white')
axs[0,0].spines["top"].set_visible(False)
axs[0,0].spines["right"].set_visible(False)
axs[0,0].get_xaxis().tick_bottom()
axs[0,0].get_yaxis().tick_left()
axs[0,0].set_xlim(time_start, time_end)
axs[0,0].set_ylim(ymin_w, ymax_w)
axs[0,0].add_patch(patch1)   
axs[0,0].xaxis.grid(True, which='minor')
axs[0,0].xaxis.set_minor_locator(MultipleLocator(5))

cbar = fig.colorbar(mesh, ax=axs[0], location='right', aspect=10, use_gridspec=grid)
cbar.set_label(label='Ze [dBZ]',  size=20)
axs[0,0].set_ylabel('Height [m]', fontsize=fontSizeX)

#mrr_cs.Zea.plot(x='time', y='height', cmap=cmap_ze_mrr, vmin=-10., vmax=40.)
mesh = axs[1,0].pcolormesh(timeLocal, range_mrr, mrr_interp.Zea.values.T, vmin=mincm_ze_mrr, vmax=maxcm_ze_mrr, cmap='viridis', rasterized=True)
axs[1,0].plot(lcl_time, lcl, color='black', label='Lifting condensation level (LCL)')
axs[1].spines["top"].set_visible(False)
axs[1].spines["right"].set_visible(False)
axs[1].get_xaxis().tick_bottom()
axs[1].get_yaxis().tick_left()
axs[1].set_xlim(time_start, time_end)
axs[1].set_ylim(ymin_mrr, ymax_mrr)
cbar = fig.colorbar(mesh, ax=axs[1], location='right', aspect=10, use_gridspec=grid)
cbar.set_label(label='Ze - MRR [dBZ]',  size=20)
axs[1].set_ylabel('Height [m]', fontsize=fontSizeX)
axs[1].set_xlabel('Time UTC [hh:mm]', fontsize=fontSizeX)
axs[1].add_patch(patch2)   
axs[1].legend(frameon=False, fontsize=20)
axs[1].xaxis.grid(True, which='minor')
axs[1].xaxis.set_minor_locator(MultipleLocator(5))
