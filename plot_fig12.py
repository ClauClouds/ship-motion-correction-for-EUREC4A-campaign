#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 20 16:32:27 2021

@author: claudia
"""

# importing necessary libraries
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import netCDF4 as nc4
from netCDF4 import Dataset
import netCDF4
import glob
import pandas as pd
from datetime import datetime
from datetime import timedelta
#import atmos
import xarray as xr
from functions_essd import f_calculateMomentsCol
from functions_essd import f_readAndMergeRadarDataDay_DopplerCorrection
from functions_essd import f_readAndMergeRadarDataDay
from functions_essd import generate_preprocess
from scipy.interpolate import CubicSpline
import custom_color_palette as ccp
from matplotlib import rcParams
import matplotlib
# importing necessary libraries
from matplotlib import rcParams
import matplotlib
import os.path
import pandas as pd
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
from functions_essd import f_closest
import matplotlib.ticker as ticker

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



path_data = '/home/cacquist/mrr_paper_plot/'
#mrr_file = '20200128_MRR_PRO_msm_eurec4a.nc'
#w_band_file = '20200128_wband_radar_msm_eurec4a_intake.nc'
mrr_file = '20200212_MRR_PRO_msm_eurec4a.nc'
w_band_file = '20200212_wband_radar_msm_eurec4a_intake.nc'
path_lcl = path_data
lcl_file = 'LCL_dataset.nc'
Path_out = path_data+'/plots/'


w_band = xr.open_dataset(path_data+w_band_file)


# selecting time intervals for the case study
time_start = datetime(2020,2,12,15,50,0,0)
time_end = datetime(2020,2,12,16,5,0,0)
#time_start = datetime(2020,1,28,9,45,0)
#time_end = datetime(2020,1,28,11,15,0)

# slicing the data for plotting the selected time interval
w_band_cs = w_band.sel(time=slice(time_start, time_end))

# removing gaps in time from w band radar data
datetimeRadar = pd.to_datetime(w_band_cs['time'].values)
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
w_band_cs = w_band_cs.reindex({"time":new_time_arr}, method=None)
print('resampling on new axis for time, done. ')


# reading stable table status
# reading ST flag file
ST_flag_file = '/home/cacquist/mrr_paper_plot/stabilization_platform_status_eurec4a.nc'
ST_flag = xr.open_dataset(ST_flag_file)


# generate stable table flag on wband radar time resolution
ST_flag_hour = ST_flag.sel(time=slice(time_start, time_end))
ST_interp_Wband = ST_flag_hour.interp(time=w_band_cs.time.values)




# reading variables to plot
Vd = w_band_cs['mean_doppler_velocity'].values
Vd[Vd == -999.] = np.nan

rangeRadar = w_band_cs['height'].values
timeLocal = pd.to_datetime(w_band_cs['time'].values)

LWP = w_band_cs['liquid_water_path'].values
LWP[LWP > 1000] = np.nan


Ze = w_band_cs['radar_reflectivity'].values
Ze[Ze == -999.]    = np.nan

Sw = w_band_cs['spectral_width'].values
Sw[Sw == -999.]    = np.nan

Sk = w_band_cs['skewness'].values
Sk[Sk == -999.]    = np.nan


#reading and slicing lcl data
lcl_data = xr.open_dataset(path_lcl+lcl_file)
lcl_cs = lcl_data.sel(time=slice(time_start, time_end))


# reading and slicing MRR data
mrr = xr.open_dataset(path_data+mrr_file)
mrr_cs = mrr.sel(time=slice(time_start, time_end))


#interpolating data on radar time stamps
mrr_interp = mrr_cs.interp(time=w_band_cs.time.values)
lcl_interp = lcl_cs.interp(time=w_band_cs.time.values)

lcl = lcl_interp.lcl.values
lcl_time = lcl_interp.time.values

# reading MRR variables
RR = mrr_interp.rain_rate.values
Ze_mrr = mrr_interp.Ze.values
range_mrr = mrr_interp.height.values
time_mrr = mrr_interp.time.values
fall_speed_mrr = -mrr_interp.fall_speed.values


labelsizeaxes   = 26
fontSizeTitle   = 26
fontSizeX       = 26
fontSizeY       = 26
cbarAspect      = 26
fontSizeCbar    = 26

# setting y range limits for the plots
ymin_w = 100.
ymax_w = 2200.
ymin_mrr = 50.
ymax_mrr = 1200.


# settings for w band radar reflectivity
mincm_ze = -40.
maxcm_ze = 30.
step_ze = 0.1
colors_ze =['#4f8c9d', '#1c4c5e', '#8ae1f9', '#8f0f1b', '#e0bfb4', '#754643', '#ef7e58', '#ff1c5d']
cmap_ze, ticks_ze, norm_ze, bounds_ze =  f_defineSingleColorPalette(colors_ze, mincm_ze, maxcm_ze, step_ze)

# settings for mean Doppler velocity
mincm_vd = -6.
maxcm_vd = 2.
step_vd = 0.1
thrs_vd = 0.
colorsLower_vd = ["#4553c2", "#2b19d9", "#d0d2f0", "#42455e", "#66abf9"]# grigio: 8c8fab
colorsUpper_vd = ["#fd2c3b", "#59413f", "#fdc7cc", "#8f323c", "#e66c4c", "#ae8788"] #MDV
cmap_vd, ticks_vd, norm_vd, bounds_vd =  f_defineDoubleColorPalette(colorsLower_vd, colorsUpper_vd, mincm_vd, maxcm_vd, step_vd, thrs_vd)
cbarstr = 'Vd [$ms^{-1}$]'

# settings for Spectral width
mincm_sw = 0.
maxcm_sw = 0.8
step_sw = 0.01
#colors = ["#72e5ef", "#460942", "#4dc172", "#cd71b5", "#274c56", "#91ec17", "#b00bd9", "#abc177"]
#colors_sw = [ "#e6bcbe", "#51443f", "#f17a74", "#e81659", "#cdd2df", "#124ac1"]
colors_sw =["#0cc0aa", "#4e2da6", "#74de58", "#bf11af", "#50942f", "#2a5fa0", "#e9b4f5", "#0b522e", "#95bbef", "#8a2f6b"]
cmap_sw, ticks_sw, norm_sw, bounds_sw =  f_defineSingleColorPalette(colors_sw, mincm_sw, maxcm_sw, step_sw)


# settings for skewness
mincm_sk = -2.
maxcm_sk = 2.
step_sk = 0.1
thrs_sk = 0.
colorsLower_sk = ["#4553c2", "#2b19d9", "#d0d2f0", "#42455e", "#66abf9"]# grigio: 8c8fab
colorsUpper_sk = ["#fd2c3b", "#59413f", "#fdc7cc", "#8f323c", "#e66c4c", "#ae8788"] #MDV
cmap_sk, ticks_sk, norm_sk, bounds_sk =  f_defineDoubleColorPalette(colorsLower_sk, colorsUpper_sk, mincm_sk, maxcm_sk, step_sk, thrs_sk)
cbarstr = 'Sk []'



#settings for MRR-pro reflectivity
mincm_ze_mrr = -10.
maxcm_ze_mrr = 40.
step_ze_mrr = 1.
#colors_ze_mrr =['#4f8c9d', '#1c4c5e', '#8ae1f9', '#8f0f1b', '#e0bfb4', '#754643', '#ef7e58', '#ff1c5d']
colors_ze_mrr = ["#56ebd3", "#9e4302", "#2af464", "#d6061a", "#1fa198"]
cmap_ze_mrr, ticks_ze_mrr, norm_ze_mrr, bounds_ze_mrr =  f_defineSingleColorPalette(colors_ze_mrr, mincm_ze_mrr, maxcm_ze_mrr, step_ze_mrr)


# settings for rain fall speed
mincm_vd_mrr = -8.
maxcm_vd_mrr = 0.
step_vd_mrr = 0.1
thrs_vd_mrr = 0.
#colorsLower_vd_mrr = ["#4553c2", "#2b19d9", "#d0d2f0", "#42455e", "#66abf9"]# grigio: 8c8fab
colorsLower_vd_mrr = ["#4553c2", "#2b19d9", "#d0d2f0", "#42455e", "#66abf9"]# grigio: 8c8fab

cmap_vd_mrr, ticks_vd_mrr, norm_vd_mrr, bounds_vd_mrr =  f_defineSingleColorPalette(colorsLower_vd_mrr, mincm_vd_mrr, maxcm_vd_mrr, step_vd_mrr)
#f_defineDoubleColorPalette(colorsLower_vd_mrr, colorsUpper_vd_mrr, mincm_vd_mrr, maxcm_vd_mrr, step_vd_mrr, thrs_vd_mrr)


# settings for rain rate
mincm_rr = 0.
maxcm_rr = 1.5
step_rr = 0.01
colors_rr =['#4f8c9d', '#1c4c5e', '#8ae1f9', '#8f0f1b', '#e0bfb4', '#754643', '#ef7e58', '#ff1c5d']
cmap_rr, ticks_rr, norm_rr, bounds_rr =  f_defineSingleColorPalette(colors_rr, mincm_rr, maxcm_rr, step_rr)



color_arr = np.repeat('black', len(ST_interp_Wband.time.values))
flag = ST_interp_Wband['flag_table_working'].values
color_arr[np.where(flag == 1)] = 'red'



# composite figure for Wband and MRR data together
fig, axs = plt.subplots(4, 2, figsize=(24,20), sharex=True, constrained_layout=True)#
grid = True
matplotlib.rc('xtick', labelsize=36)  # sets dimension of ticks in the plots
matplotlib.rc('ytick', labelsize=36) # sets dimension of ticks in the plots
  # sets dimension of ticks in the plots


# plotting W-band radar variables and color bars
mesh = axs[0,0].pcolormesh(timeLocal, rangeRadar, Ze.T, vmin=maxcm_ze, vmax=mincm_ze, cmap=cmap_ze, rasterized=True)
cbar = fig.colorbar(mesh, ax=axs[0,0], location='right', aspect=10, use_gridspec=grid)
cbar.set_label(label='Ze [dBZ]',  size=26)

mesh = axs[1,0].pcolormesh(timeLocal, rangeRadar, Vd.T, vmin=mincm_vd, vmax=maxcm_vd, cmap=cmap_vd, rasterized=True)
cbar = fig.colorbar(mesh, ax=axs[1,0], location='right', aspect=10, use_gridspec=grid)
cbar.set_label(label='Vd [ms$^{-1}$]',  size=26)

mesh = axs[2,0].pcolormesh(timeLocal, rangeRadar, Sw.T, vmin=mincm_sw, vmax=maxcm_sw, cmap=cmap_sw, rasterized=True)
cbar = fig.colorbar(mesh, ax=axs[2,0], location='right', aspect=10, use_gridspec=grid)
cbar.set_label(label='Spectral width [ms$^{-1}$]',  size=26)

mesh = axs[3,0].pcolormesh(timeLocal, rangeRadar, -Sk.T, vmin=mincm_sk, vmax=maxcm_sk, cmap=cmap_sk, rasterized=True)
cbar = fig.colorbar(mesh, ax=axs[3,0], location='right', aspect=10, use_gridspec=grid)
cbar.set_label(label='Skewness []',  size=26)

axs[3,0].set_xlabel('Time UTC [hh:mm]', fontsize=fontSizeX)

for ax, l in zip(axs[:,0].flatten(), ['(a) Reflectivity - W-band',  '(b) Mean Doppler velocity - W-band ', '(c) Spectral width - W-band', '(d) Skewness - W-band']):
    ax.text(-0.05, 1.05, l,  fontweight='black', fontsize=26, transform=ax.transAxes)
    ax.set_xlim(time_start, time_end)
    ax.set_ylabel('Height [m]', fontsize=fontSizeX)
    ax.set_ylim(100., 2200.)
    ax.plot(lcl_time, lcl, color='black', linestyle='dotted', linewidth=3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(3)
    ax.spines["left"].set_linewidth(3)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(n=3))
    ax.tick_params(which='minor', length=7, width=3)
    ax.tick_params(which='major', length=7, width=3)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(n=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.tick_params(axis='both', labelsize=26)


mesh = axs[0,1].pcolormesh(time_mrr, range_mrr,  Ze_mrr.T, vmin=mincm_ze_mrr, vmax=maxcm_ze_mrr, cmap=cmap_ze_mrr, rasterized=True)
cbar = fig.colorbar(mesh, ax=axs[0,1], location='right', aspect=10, use_gridspec=grid)
cbar.set_label(label='Ze [dBZ]',  size=fontSizeX)

mesh = axs[1,1].pcolormesh(time_mrr, range_mrr, fall_speed_mrr.T, vmin=-8., vmax=0., cmap=cmap_vd_mrr, rasterized=True)
cbar = fig.colorbar(mesh, ax=axs[1,1], location='right', aspect=10, use_gridspec=grid)
cbar.set_label(label='Vd [ms$^{-1}$]',  size=fontSizeX)

mesh = axs[2,1].pcolormesh(time_mrr, range_mrr, RR.T, vmin=mincm_rr, vmax=maxcm_rr, cmap='viridis', rasterized=True)
cbar = fig.colorbar(mesh, ax=axs[2,1], location='right', aspect=10, use_gridspec=grid)
cbar.set_label(label='RR [mmh$^{-1}$]',  size=fontSizeX)

mesh = axs[3,1].scatter(timeLocal, LWP, c=color_arr, vmin=0., vmax=1000.)
#axs[2,1].text(0.1, 0.5, 'Begin text', horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes)

axs[3,1].plot([datetime(2020,2,12,16,1,45)], [800.], 'o', color='black', markersize=10)
axs[3,1].plot([datetime(2020,2,12,16,1,45)], [700.], 'o', color='red', markersize=10)

axs[3,1].text(0.83, 0.8, 'stab. ON', verticalalignment='center', transform=axs[3,1].transAxes, fontsize=20) #horizontalalignment='center',
axs[3,1].text(0.83, 0.72, 'stab. OFF',verticalalignment='center', transform=axs[3,1].transAxes, fontsize=20) # horizontalalignment='center',

count=0

for ax, l in zip(axs[:,1].flatten(), ['(e) Reflectivity - MRR-PRO',  '(f) Mean Doppler velocity - MRR-PRO', '(g) Rain rate - MRR-PRO', '(h) Liquid water path - W-band']):
    ax.text(-0.05, 1.05, l,  fontweight='black', fontsize=26, transform=ax.transAxes)
    ax.set_xlim(time_start, time_end)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(3)
    ax.spines["left"].set_linewidth(3)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(n=3))
    ax.tick_params(which='minor', length=7, width=3)
    ax.tick_params(which='major', length=7, width=3)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(n=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.tick_params(axis='both', labelsize=26)

    if (count <= 2):
        ax.set_ylabel('Height [m]', fontsize=fontSizeX)
        ax.set_ylim(100., 1200.)
        ax.plot(lcl_time, lcl, color='black', linestyle='dotted', linewidth=3)
    else:
        ax.set_ylabel('LWP [gm$^{-2}$]', fontsize=fontSizeX)
        ax.set_ylim(0., 1000.)
        ax.set_xlabel('Time UTC [hh:mm]', fontsize=fontSizeX)

    count=count+1

fig.savefig(Path_out+'Fig12.png')

