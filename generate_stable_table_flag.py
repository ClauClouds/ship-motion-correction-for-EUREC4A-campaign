#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 11:51:27 2021
create a ncdf file containing a flag for when the stable table is working and when it is not. When it is not working, we save roll and pitch last recorded values 
@author: claudia
"""
import numpy as np
import glob
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.dates as mdates
import numpy as np
import netCDF4 as nc4
from netCDF4 import Dataset
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import rcParams
import matplotlib
import glob
import pandas as pd
from datetime import datetime
from datetime import timedelta
#import atmos
import xarray as xr
from pathlib import Path
import os.path
from matplotlib import rcParams

# paths to the different data files and output directories for plots
pathFolderTree  = '/Volumes/Extreme SSD/ship_motion_correction_merian/'

PathFigHour = pathFolderTree+'plots/paperPlots/'
Path(PathFigHour).mkdir(parents=True, exist_ok=True)

file_list = np.sort(glob.glob('/Volumes/Extreme SSD/ship_motion_correction_merian/stable_table_processed_data/*_stableTableFlag.nc'))

data = xr.open_mfdataset(file_list)


#defining flag for table working
flag = np.repeat(0.,len(data.time.values))

# setting flag to 1 when the angles are nan            
flag[np.where(~np.isnan(data.pitch.values))[0]] = 1

flag[np.where(~np.isnan(data['roll'].values))[0]] = 1

# add new variable flag_table_working to the data
dims             = ['time']
coords          = {"time":data.time.values}
flag_datarray   = xr.DataArray(dims=dims, coords=coords, data=flag,
                         attrs={'long_name':'flag indicating if stabilization platform is working flag == 0: working,  flag == 1: not working',
                                'units':''})
    
data['flag_table_working']          = flag_datarray



data['roll'].attrs={'long_name':'Last recorded roll angle when the table gets stuck. The value is constant until the table starts working again',
                                'units':'degrees'}
data['pitch'].attrs={'long_name':'Last recorded pitch angle when the table gets stuck. The value is constant until the table starts working again',
                                'units':'degrees'}
data.to_netcdf(pathFolderTree+'/stable_table_processed_data/stabilization_platform_status_eurec4a.nc')
#%%

# producing hourly and daily quicklooks for the 
Eurec4aDays     = pd.date_range(datetime(2020,2,6),datetime(2020,2,19),freq='d')
NdaysEurec4a    = len(Eurec4aDays)
hours_arr = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23']
radar_name = 'msm'


for n_day, dayEu in enumerate(Eurec4aDays):
    
    # reading yy, mm, dd
    yy              = str(dayEu)[0:4]
    mm              = str(dayEu)[5:7]
    dd              = str(dayEu)[8:10]
    date            = yy+mm+dd
    
    #defining output path and creating it if not existing
    path_out = '/Volumes/Extreme SSD/ship_motion_correction_merian/plots/quicklooks_hourly_ST/'+date+'/'
    Path(path_out).mkdir(parents=True, exist_ok=True)
    
    
    
    date_start = datetime(int(yy), int(mm), int(dd), 0, 0, 0)
    date_end = datetime(int(yy), int(mm), int(dd), 23, 59, 59)
    
    # conversion of UTC to local time
    timeLocal = pd.to_datetime(data.time.values)-timedelta(hours=4)
    timeStartDay = timeLocal[0]
    timeEndDay = timeLocal[-1]
    
    date_start_local = date_start-timedelta(hours=4)
    date_end_local = date_end-timedelta(hours=4)
    
    #plot a quicklook panel showing the table status and the last recorded angles when position is stuck
    labelsizeaxes   = 26
    fontSizeTitle   = 26
    fontSizeX       = 26
    fontSizeY       = 26
    cbarAspect      = 10
    fontSizeCbar    = 26
    rcParams['font.sans-serif'] = ['Tahoma']
    matplotlib.rcParams['savefig.dpi'] = 100
    plt.rcParams.update({'font.size':14})
    grid            = True
    matplotlib.rc('xtick', labelsize=26)  # sets dimension of ticks in the plots
    matplotlib.rc('ytick', labelsize=26)  # sets dimension of ticks in the plots
    fig, axs = plt.subplots(3, 1, figsize=(14,14), sharex=True, constrained_layout=True)
    
    # build colorbar
    mesh = axs[0].scatter(timeLocal, data["flag_table_working"].values, marker='D', color='red')
    
    mesh =axs[1].scatter(timeLocal, data["pitch"].values, marker='D', color='blue')
    
    
    axs[2].set_xlabel('Local Time (UTC - 4h) [hh:mm]', fontsize=fontSizeX)
    
    [a.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M')) for a in axs[:].flatten()]
    
    mesh = axs[2].scatter(timeLocal, data["roll"].values, marker='D', color='green')
    #axs[2].set_title('Corrected and smoothed', fontsize=fontSizeX, loc='left')
    axs[2].set_xlim(date_start_local, date_end_local)
    
    for ax, l in zip(axs.flatten(), ['(a) Flag stabilization platform (working=0, not working=1)', '(b) Pitch recorded in stuck positions ', '(c) Roll recorded in stuck positions']):
        ax.text(-0.05, 1.05, l,  fontweight='black', fontsize=26, transform=ax.transAxes)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_linewidth(2)
        ax.spines["left"].set_linewidth(2)
        ax.tick_params(which='minor', length=7, width=3)
        ax.tick_params(which='major', length=7, width=3)
    fig.savefig(path_out+'/'+yy+mm+dd+'_'+radar_name+'_stabilization_platform_status.png')


    
    for n_hour, hour in enumerate(hours_arr):
        
        date_start = datetime(int(yy), int(mm), int(dd), int(hour), 0, 0)
        date_end = datetime(int(yy), int(mm), int(dd), int(hour), 59, 59)
        
        # conversion of UTC to local time
        timeLocal = pd.to_datetime(data.time.values)-timedelta(hours=4)
        timeStartDay = timeLocal[0]
        timeEndDay = timeLocal[-1]
        
        date_start_local = date_start-timedelta(hours=4)
        date_end_local = date_end-timedelta(hours=4)
        
        #plot a quicklook panel showing the table status and the last recorded angles when position is stuck
        labelsizeaxes   = 26
        fontSizeTitle   = 26
        fontSizeX       = 26
        fontSizeY       = 26
        cbarAspect      = 10
        fontSizeCbar    = 26
        rcParams['font.sans-serif'] = ['Tahoma']
        matplotlib.rcParams['savefig.dpi'] = 100
        plt.rcParams.update({'font.size':14})
        grid            = True
        matplotlib.rc('xtick', labelsize=26)  # sets dimension of ticks in the plots
        matplotlib.rc('ytick', labelsize=26)  # sets dimension of ticks in the plots
        fig, axs = plt.subplots(3, 1, figsize=(14,14), sharex=True, constrained_layout=True)
        
        # build colorbar
        mesh = axs[0].scatter(timeLocal, data["flag_table_working"].values, marker='D', color='red')
        
        mesh =axs[1].scatter(timeLocal, data["pitch"].values, marker='D', color='blue')
        
        
        axs[2].set_xlabel('Local Time (UTC - 4h) [hh:mm]', fontsize=fontSizeX)
        
        [a.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M')) for a in axs[:].flatten()]
        
        mesh = axs[2].scatter(timeLocal, data["roll"].values, marker='D', color='green')
        #axs[2].set_title('Corrected and smoothed', fontsize=fontSizeX, loc='left')
        axs[2].set_xlim(date_start_local, date_end_local)
        
        for ax, l in zip(axs.flatten(), ['(a) Flag stabilization platform (working=0, not working=1)', '(b) Pitch recorded in stuck positions ', '(c) Roll recorded in stuck positions']):
            ax.text(-0.05, 1.05, l,  fontweight='black', fontsize=26, transform=ax.transAxes)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["bottom"].set_linewidth(2)
            ax.spines["left"].set_linewidth(2)
            ax.tick_params(which='minor', length=7, width=3)
            ax.tick_params(which='major', length=7, width=3)
        fig.savefig(path_out+'/'+yy+mm+dd+'_'+hour+'_'+radar_name+'_stabilization_platform_status.png')
        
