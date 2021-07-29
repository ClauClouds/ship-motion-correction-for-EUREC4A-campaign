#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 20:10:13 2021

@author: cacquist
"""

# importing necessary libraries
from matplotlib import rcParams
import matplotlib
import os.path
import pandas as pd
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import netCDF4 as nc4
from netCDF4 import Dataset
import glob
import pandas as pd
from datetime import datetime, timedelta
# import atmos
import xarray as xr
from functions_essd import f_calculateMomentsCol
from functions_essd import f_readAndMergeRadarDataDay_DopplerCorrection
from functions_essd import f_readAndMergeRadarDataDay
from functions_essd import generate_preprocess
from scipy.interpolate import CubicSpline
import custom_color_palette as ccp


def f_defineSingleColorPalette(colors, minVal, maxVal, step):
    """
    author: Claudia Acquistapace
    date : 13 Jan 2020
    goal: define color palette and connected variables associated to a given
    range for a variable
    input:
        colors: list of strings identifying the colors selected (HEX)
        minVal: min value for the color palette
        maxVal: max value to be assigned to the color paletter
        step  : increment for the color palette
    output:
        cmap: personalized color map
        ticks
        norm
        bounds
    """

    Intervals = ccp.range(minVal, maxVal, step)

    # sublist with colors that will be used to create a custom palette
    palette = [colors, Intervals]

    # we pass the parm_color inside a list to the creates_palette module
    cmap, ticks, norm, bounds = ccp.creates_palette([palette])

    return(cmap, ticks, norm, bounds)
def f_defineDoubleColorPalette(colorsLower, colorsUpper, minVal, maxVal, step, thrs):
    """
    author: Claudia Acquistapace
    date : 13 Jan 2020
    goal: define dual color palette (i.e. based on two sets of colors) and its parameters based on one ensemble of colors based on the input parameters
    input:
        lower_palette: list of strings identifying the colors selected (HEX) for the lower part of the palette i.e. Values <Thr
        upper_palette: list of strings identifying the colors selected (HEX) for the upper part of the palette i.e. Values >Thr
        minVal: min value for the color palette
        maxVal: max value to be assigned to the color paletter
        step  : increment for the color palette
        thrs  : threshold value to be used to as separation for the upper and lower color palette
    output:
        cmap: personalized color map
        ticks
        norm
        bounds
    """
    lower_palette = [colorsLower, ccp.range(minVal, thrs, step)] # grigio: 8c8fab
    upper_palette = [colorsUpper, ccp.range(thrs, maxVal, step)] #sk # last old color 987d7b

    # we pass the parm_color inside a list to the creates_palette module
    cmap, ticks, norm, bounds = ccp.creates_palette([lower_palette, upper_palette])

    return(cmap, ticks, norm, bounds)



# paths to the different data files and output directories for plots
pathFolderTree  = '/Volumes/Extreme SSD/ship_motion_correction_merian/'
pathFig         = pathFolderTree+'/plots/quicklooks_hourly/'
pathNcDataAnc   = pathFolderTree+'/ncdf_ancillary/'
Path(pathFig).mkdir(parents=True, exist_ok=True)

PathFigHour = pathFolderTree+'plots/paperPlots/'
Path(PathFigHour).mkdir(parents=True, exist_ok=True)
#Path(pathOutData).mkdir(parents=True, exist_ok=True)


# generating array of days for the dataset
Eurec4aDays  = pd.date_range(datetime(2020,1,19),datetime(2020,2,19),freq='d')
NdaysEurec4a = len(Eurec4aDays)
radar_name = 'msm'


#%%

dayEu = '2020-01-20'
#dayEu = '2020-02-10'

# extracting strings for yy, dd, mm
yy = str(dayEu)[0:4]
mm = str(dayEu)[5:7]
dd = str(dayEu)[8:10]



# setting dates strings
date = dd+mm+yy      # '04022020'
dateRadar = yy[0:2]+mm+dd   # '200204'
dateReverse = yy+mm+dd      # '20200204'

# creating output path for figures
Path(pathFig+'/'+dateReverse+'/').mkdir(parents=True, exist_ok=True)

pathRadar = pathFolderTree+'/corrected_data/'+yy+'/'+mm+'/'+dd+'/'
filename = pathRadar+date+'_06msm94_msm_ZEN_corrected.nc'


# read hour from the selected files
n = len(pathRadar)
hh = filename[n+9:n+11]   # 19012020_00msm94_msm_ZEN_corrected
hourInt = int(hh)

# setting starting and ending time for the hour an
timeStart    = datetime(int(yy),int(mm), int(dd),hourInt,0,0)
if hh != '23':
    timeEnd      = datetime(int(yy),int(mm), int(dd),hourInt+1,0,0)
else:
    timeEnd      = datetime(int(yy),int(mm), int(dd),hourInt,59,59)
    
    

MeanDopVelDataset = xr.open_dataset(filename)
datetimeRadar = pd.to_datetime(MeanDopVelDataset['time'].values)

# reading second file
filename2 = pathFolderTree+'/corrected_data/2020/02/10/10022020_02msm94_msm_ZEN_corrected.nc'
MeanDopVelDataset_2 = xr.open_dataset(filename2)
datetimeRadar_2 = pd.to_datetime(MeanDopVelDataset_2['time'].values)

#%%
time_diff = np.ediff1d(datetimeRadar)

#converting time differences to seconds
diff_list = [pd.Timedelta(t).total_seconds() for t in time_diff]

# find where diff_list > 4 seconds:
diff_arr = np.asarray(diff_list)
i_gaps = np.where(diff_arr > 4.)[0][:]

# defining new list of time stamps where we add the new missing times
new_time_arr = datetimeRadar.tolist()
len_added_times = []
# finding how many time stamps have to be introduced.
for i, i_gaps_val in enumerate(i_gaps):
    #print(i, i_gaps_val)
    time_stop = datetimeRadar[i_gaps_val]
    time_restart = datetimeRadar[i_gaps_val+1]

    # calculate number of time stamps to add
    deltaT = diff_arr[i_gaps_val]
    n_times_to_add = deltaT//3

    # calculate time stamps to add in the gaps
    time_to_add = [time_stop+i_n*(timedelta(seconds=3)) for i_n in np.arange(1,n_times_to_add+1)]

    # storing amount of inserted values
    len_added_times.append(len(time_to_add))

    #print('time stamps to add: ', time_to_add)
    # loop on time to add elements for inserting them in the list one by one
    for ind in range(len(time_to_add)):
        # read value to insert
        val_to_insert = time_to_add[ind]

        # find index where to insert
        if i == 0:
            ind_start = i_gaps_val+1
        else:
            ind_start = new_time_arr.index(time_stop)+1
        #print(i_gaps_val, ind_start)
        new_time_arr.insert(ind_start+ind, val_to_insert)


new_time_arr = pd.to_datetime(np.asarray(new_time_arr))
print('gaps found: ', len(i_gaps))
print('dim new time array ', len(new_time_arr))
print('dim old time array ', len(datetimeRadar))
print('******************')

# resampling radar data on new time array
MeanDopVelDataset_res = MeanDopVelDataset.reindex({"time":new_time_arr}, method=None)
print('resampling on new axis for time, done. ')

#%%
# reading ST flag file
ST_flag_file = '/Volumes/Extreme SSD/ship_motion_correction_merian/stable_table_processed_data/stabilization_platform_status_eurec4a.nc'
ST_flag = xr.open_dataset(ST_flag_file)



Vd = MeanDopVelDataset_res['vm'].values
rangeRadar = MeanDopVelDataset_res['range'].values
Vd[Vd == -999.] = np.nan

Vd_corr = MeanDopVelDataset_res['vm_corrected'].values
rangeRadar = MeanDopVelDataset_res['range'].values
Vd_corr[Vd_corr == -999.] = np.nan

Vd_corr_smooth = MeanDopVelDataset_res['vm_corrected_smoothed'].values
rangeRadar = MeanDopVelDataset_res['range'].values
Vd_corr_smooth[Vd_corr_smooth == -999.] = np.nan

datetimeRadar = pd.to_datetime(MeanDopVelDataset_res['time'].values)

# setting time interval for plotting
date_start = datetime(2020,1,20,6,21,20)
date_end = datetime(2020,1,20,6,27,0)
#date_start = datetime(2020,2,7,16,10,0)
#date_end = datetime(2020,2,7,16,35,0)
#date_start = datetime(2020,2,10,2,30,0)
#date_end = datetime(2020,2,10,2,43,0)


time_local = datetimeRadar-timedelta(hours=4)
date_start_local = date_start-timedelta(hours=4)
date_end_local = date_end-timedelta(hours=4)



#datetimeTable_local = pd.to_datetime(datetimeTable)-timedelta(hours=4)
ST_time_local = pd.to_datetime(ST_flag.time.values)-timedelta(hours=4)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# settings for mean Doppler velocity
h_max_2 = 3000.
h_max = 2200.

mincm = -5.
maxcm = 5.
step = 0.1
thrs = 0.
colorsLower = ["#4553c2", "#2b19d9", "#d0d2f0", "#42455e", "#66abf9"]# grigio: 8c8fab
colorsUpper = ["#fd2c3b", "#59413f", "#fdc7cc", "#8f323c", "#e66c4c", "#ae8788"] #MDV
cmap, ticks, norm, bounds =  f_defineDoubleColorPalette(colorsLower, colorsUpper, mincm, maxcm, step, thrs)
cbarstr = 'Vd [$ms^{-1}$]'
# plot quicklook of filtered and corrected mdv for checking
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
matplotlib.rc('xtick', labelsize=24)  # sets dimension of ticks in the plots
matplotlib.rc('ytick', labelsize=24)  # sets dimension of ticks in the plots
fig, axs = plt.subplots(4, 2, figsize=(25,18), constrained_layout=True)#  

# build colorbar
mesh = axs[0,0].pcolormesh(time_local, rangeRadar, Vd.T, vmin=mincm, vmax=maxcm, cmap=cmap, rasterized=True)

mesh = axs[1,0].pcolormesh(time_local, rangeRadar, Vd_corr.T, vmin=mincm, vmax=maxcm, cmap=cmap, rasterized=True)

mesh = axs[2,0].pcolormesh(time_local, rangeRadar, Vd_corr_smooth.T, vmin=mincm, vmax=maxcm, cmap=cmap, rasterized=True)

#axs[3].scatter(datetimeTable_local, flagTable)
axs[3,0].scatter(ST_time_local, ST_flag.flag_table_working.values, color='red')
axs[3,0].set_xlabel('Local time (UTC-4h) [hh:mm]', fontsize=fontSizeX)

axs[0,0].set_ylabel('Height [m]', fontsize=fontSizeX)
axs[0,0].set_ylim(100., h_max)
axs[1,0].set_ylabel('Height [m]', fontsize=fontSizeX)
axs[1,0].set_ylim(100., h_max)
axs[2,0].set_ylabel('Height [m]', fontsize=fontSizeX)
axs[2,0].set_ylim(100., h_max)
#axs[3].set_ylabel('Status [1=off,0=on]', fontsize=fontSizeX)
#axs[3].yaxis.set_major_locator(ticker.MultipleLocator(1.0))
axs[3,0].set_yticks([0, 1, 1.5])
axs[3,0].set_yticklabels(['table ON', 'table OFF', ''])
#axs[3].yaxis.set_major_formatter(ticker.FixedFormatter(['table on', 'table off']))
#axs[3].set_ylim(0, 1)

for ax, l in zip(axs[:,0].flatten(), ['(a) Original', '(b) Corrected', '(c) Corrected and smoothed', '(d) Stabilization platform status']):
    ax.text(-0.05, 1.05, l,  fontweight='black', fontsize=26, transform=ax.transAxes)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(date_start_local, date_end_local)
    ax.spines["bottom"].set_linewidth(2)
    ax.spines["left"].set_linewidth(2)
    ax.tick_params(which='minor', length=7, width=3)
    ax.tick_params(which='major', length=7, width=3)

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))


# panels on the right

# reading variables to plot
Vd_2 = MeanDopVelDataset_2['vm'].values
Vd_2[Vd_2 == -999.] = np.nan
Vd_corr_2 = MeanDopVelDataset_2['vm_corrected'].values
Vd_corr_2[Vd_corr_2 == -999.] = np.nan
Vd_corr_smooth_2 = MeanDopVelDataset_2['vm_corrected_smoothed'].values
Vd_corr_smooth_2[Vd_corr_smooth_2 == -999.] = np.nan
datetimeRadar_2 = pd.to_datetime(MeanDopVelDataset_2['time'].values)
time_local_2 = datetimeRadar_2-timedelta(hours=4)
date_start_2 = datetime(2020,2,10,2,30,0)
date_end_2 = datetime(2020,2,10,2,36,0)
date_start_local_2 = date_start_2-timedelta(hours=4)
date_end_local_2 = date_end_2-timedelta(hours=4)
rangeRadar_2 = MeanDopVelDataset_2['range'].values


# build colorbar
mesh = axs[0,1].pcolormesh(time_local_2, rangeRadar, Vd_2.T, vmin=mincm, vmax=maxcm, cmap=cmap, rasterized=True)

mesh = axs[1,1].pcolormesh(time_local_2, rangeRadar, Vd_corr_2.T, vmin=mincm, vmax=maxcm, cmap=cmap, rasterized=True)

mesh = axs[2,1].pcolormesh(time_local_2, rangeRadar, Vd_corr_smooth_2.T, vmin=mincm, vmax=maxcm, cmap=cmap, rasterized=True)

#axs[3].scatter(datetimeTable_local, flagTable)
axs[3,1].scatter(ST_time_local, ST_flag.flag_table_working.values, color='red')
axs[3,1].set_xlabel('Local time (UTC-4h) [hh:mm]', fontsize=fontSizeX)

axs[0,1].set_ylabel('Height [m]', fontsize=fontSizeX)
axs[0,1].set_ylim(100., h_max_2)
axs[1,1].set_ylabel('Height [m]', fontsize=fontSizeX)
axs[1,1].set_ylim(100., h_max_2)
axs[2,1].set_ylabel('Height [m]', fontsize=fontSizeX)
axs[2,1].set_ylim(100., h_max_2)
#axs[3].set_ylabel('Status [1=off,0=on]', fontsize=fontSizeX)
#axs[3].yaxis.set_major_locator(ticker.MultipleLocator(1.0))
axs[3,1].set_yticks([0, 1, 1.5])
axs[3,1].set_yticklabels(['table ON', 'table OFF', ''])
#axs[3].yaxis.set_major_formatter(ticker.FixedFormatter(['table on', 'table off']))
#axs[3].set_ylim(0, 1)

cbar = fig.colorbar(mesh, ax=axs[:,1], location='right', aspect=20, use_gridspec=grid)
cbar.set_label(label='Mean Doppler velocity [$ms^{-1}$]', size=26)

for ax, l in zip(axs[:,1].flatten(), ['(e) Original', '(f) Corrected', '(g) Corrected and smoothed', '(h) Stabilization platform status']):
    ax.text(-0.05, 1.05, l, fontweight='black', fontsize=26, transform=ax.transAxes)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(date_start_local_2, date_end_local_2)
    ax.spines["bottom"].set_linewidth(2)
    ax.spines["left"].set_linewidth(2)
    ax.tick_params(which='minor', length=7, width=3)
    ax.tick_params(which='major', length=7, width=3)

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    
fig.savefig(PathFigHour+date+'_'+hh+'_figure_paper.png')
