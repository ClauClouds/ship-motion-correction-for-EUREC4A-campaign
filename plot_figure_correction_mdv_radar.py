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
from matplotlib import rcParams
import matplotlib


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
Path(pathOutData).mkdir(parents=True, exist_ok=True)


# generating array of days for the dataset
Eurec4aDays  = pd.date_range(datetime(2020,1,19),datetime(2020,2,19),freq='d')
NdaysEurec4a = len(Eurec4aDays)
radar_name = 'msm'
#%%

indDay = 1
dayEu = '2020-01-20'


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
# reading variables to plot


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
# settings for mean Doppler velocity
mincm = -3.
maxcm = 3.
step = 0.1
thrs = 0.
colorsLower = ["#4553c2", "#2b19d9", "#d0d2f0", "#42455e", "#66abf9"]# grigio: 8c8fab
colorsUpper = ["#fd2c3b", "#59413f", "#fdc7cc", "#8f323c", "#e66c4c", "#ae8788"] #MDV
cmap, ticks, norm, bounds =  f_defineDoubleColorPalette(colorsLower, colorsUpper, mincm, maxcm, step, thrs)
cbarstr = 'Vd [$ms^{-1}$]'
# plot quicklook of filtered and corrected mdv for checking
labelsizeaxes   = 14
fontSizeTitle   = 16
fontSizeX       = 16
fontSizeY       = 16
cbarAspect      = 10
fontSizeCbar    = 16
rcParams['font.sans-serif'] = ['Tahoma']
matplotlib.rcParams['savefig.dpi'] = 100
plt.rcParams.update({'font.size':14})
grid            = True
fig, axs = plt.subplots(3, 1, figsize=(14,14), sharey=True, constrained_layout=True)

# build colorbar
mesh = axs[0].pcolormesh(datetimeRadar, rangeRadar,  Vd.T, vmin=mincm, vmax=maxcm, cmap=cmap, rasterized=True)
#axs[0].set_title('Original', loc='left')
axs[0].spines["top"].set_visible(False)
axs[0].spines["right"].set_visible(False)
axs[0].get_xaxis().tick_bottom()
axs[0].get_yaxis().tick_left()
axs[0].set_xlim(datetimeRadar[0], datetimeRadar[-1])

mesh = axs[1].pcolormesh(datetimeRadar, rangeRadar, Vd_corr.T, vmin=mincm, vmax=maxcm, cmap=cmap, rasterized=True)
#[a.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M')) for a in axs.flatten()]
axs[1].spines["top"].set_visible(False)
axs[1].spines["right"].set_visible(False)
axs[1].get_xaxis().tick_bottom()
axs[1].get_yaxis().tick_left()
#axs[1].set_title('Corrected', fontsize=fontSizeX, loc='left')
axs[1].set_xlim(datetimeRadar[0], datetimeRadar[-1])

axs[0].set_ylabel('Height   [m]', fontsize=fontSizeX)
axs[1].set_ylabel('Height   [m]', fontsize=fontSizeX)
axs[2].set_ylabel('Height   [m]', fontsize=fontSizeX)

axs[0].set_ylim(100., 2500.)
axs[1].set_ylim(100., 2500.)

axs[2].set_ylim(100., 2500.)

axs[2].set_xlabel('time [hh:mm]', fontsize=fontSizeX)
#axs[0].set_xlabel('time [hh:mm]', fontsize=fontSizeX)
#axs[1].set_xlabel('time [mm:ss]', fontsize=fontSizeX)
#axs[1].set_xlabel('time [mm:ss]', fontsize=fontSizeX)
[a.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M')) for a in axs[:].flatten()]

mesh = axs[2].pcolormesh(datetimeRadar, rangeRadar, Vd_corr_smooth.T, vmin=mincm, vmax=maxcm, cmap=cmap, rasterized=True)
#axs[2].set_title('Corrected and smoothed', fontsize=fontSizeX, loc='left')
axs[2].spines["top"].set_visible(False)
axs[2].spines["right"].set_visible(False)
axs[2].get_xaxis().tick_bottom()
axs[2].get_yaxis().tick_left()
axs[2].set_xlim(timeStart, timeEnd)

cbar = fig.colorbar(mesh, ax=axs[:], location='right', aspect=60, use_gridspec=grid)
cbar.set_label(label='Mean Doppler velocity [$ms^{-1}$]', size=20)
for ax, l in zip(axs.flatten(), ['(a) Original', '(b) Corrected', '(c) Corrected and smoothed']):
    ax.text(-0.05, 1.05, l,  fontweight='black', fontsize=20, transform=ax.transAxes)
fig.savefig(PathFigHour+date+'_'+hh+'_figure_paper.png')
fig.savefig(PathFigHour+date+'_'+hh+'_figure_paper.pdf')