#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 11 Nov 18:02

@author: claudia
@ reproduce appendix image of interference patterns for MRR

"""


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


mrr_file_original = '/home/cacquist/mrr_paper_plot/20200213_010000-FirstStep-processed.nc'
mrr_file_final = '/home/cacquist/mrr_paper_plot/20200213_MRR_PRO_msm_eurec4a.nc'
mrr_spec_file = '/home/cacquist/mrr_paper_plot/13022020_01_preprocessedClau_4Albert.nc'
Path_out = '/home/cacquist/mrr_paper_plot/plots/'
mrr_metek = '/home/cacquist/mrr_paper_plot/20200213_010000.nc'



dict_plot = {'path':"/home/cacquist/mrr_paper_plot/plots/",
             "varname":'mrr_process', 
         "instr":"MRR_PRO"}

# reading variables to plot from the corrected file
mrr_final              = xr.open_dataset(mrr_file_final)
time                   = mrr_final['time'].values
units_time             = 'seconds since 1970-01-01 00:00:00'
datetimeM              = pd.to_datetime(time, unit ='s', origin='unix')
mrr_slice_hour         = mrr_final.sel(time=slice(datetime(2020,2,13,1,0,0), datetime(2020,2,13,1,59,59)))
height                 = mrr_slice_hour['height'].values
Ze                     = mrr_slice_hour['Ze'].values
W                      = mrr_slice_hour['fall_speed'].values
time_hour              = mrr_slice_hour['time'].values
datetime_hour          = pd.to_datetime(time_hour, unit ='s', origin='unix')


# setting time limits for the plot
timeStartDay = datetime(2020,2,13,1,32,0,0)
timeEndDay = datetime(2020,2,13,1,35,0,0)



# reading original data for Ze and W original 
origData = xr.open_dataset(mrr_file_original)
W_orig   = origData['W'].values
Ze_orig = origData['Zea'].values
time_orig = origData['time'].values
datetime_orig = pd.to_datetime(time_orig, unit ='s', origin='unix')
height = origData['Height'].values

# reading spec file for spectra and noise
spec_data = xr.open_dataset(mrr_spec_file)
spec_metek = xr.open_dataset(mrr_metek)
spec = spec_metek['spectrum_raw'].values
time_spec = spec_data['time'].values
datetime_spec = pd.to_datetime(time_spec, unit ='s', origin='unix')
height_spec = spec_metek['range'].values


# defining doppler
doppler = np.arange(0,-12,-0.1875)


# selecting time_data and time_noise
time_data = datetime(2020,2,13,1,32,0,0)
time_noise = datetime(2020,2,13,1,50,0,0)
from functions_essd import f_closest

ind_time_data = f_closest(datetime_spec, time_data)
ind_time_noise = f_closest(datetime_spec, time_noise)

spec_data = spec[ind_time_data,:,:]
spec_noise = spec[ind_time_noise,:,:]



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


grid = True
fig, axs = plt.subplots(3, 2, figsize=(14,9), constrained_layout=True)
[a.get_yaxis().tick_left() for a in axs[:,:].flatten()]
[a.get_xaxis().tick_bottom() for a in axs[:,:].flatten()]

mesh = axs[0,0].pcolormesh(datetime_orig, height, -W_orig.T, vmin=-8., vmax=0., cmap='jet', rasterized=True)
cbar = fig.colorbar(mesh, ax=axs[0,0], location='right', aspect=10, use_gridspec=grid)
cbar.set_label(label='Vd [ms$^{-1}$]',  size=fontSizeX)
axs[0,0].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M")) 
axs[0,0].axvline(x=pd.to_datetime(time_data), color='black',linewidth=4, linestyle=':')
axs[0,0].axvline(x=pd.to_datetime(time_noise), color='black',linewidth=4, linestyle=':')
axs[0,0].set_xlabel('Time UTC [hh:mm]', fontsize=16)

mesh = axs[0,1].pcolormesh(datetime_orig, height, Ze_orig.T, vmin=-10., vmax=40., cmap=cmap_ze_mrr, rasterized=True)
cbar = fig.colorbar(mesh, ax=axs[0,1], location='right', aspect=10, use_gridspec=grid)
cbar.set_label(label='Ze [dBZ]',  size=fontSizeX)
axs[0,1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M")) 
axs[0,1].axvline(x=pd.to_datetime(time_data), color='black',linewidth=4, linestyle=':')
axs[0,1].axvline(x=pd.to_datetime(time_noise), color='black',linewidth=4, linestyle=':')
axs[0,1].set_xlabel('Time UTC [hh:mm]', fontsize=16)

mesh = axs[1,0].pcolormesh(doppler, height_spec, spec_data, vmin=0., vmax=30., cmap='jet', rasterized=True)
cbar = fig.colorbar(mesh, ax=axs[1,0], location='right', aspect=10, use_gridspec=grid)
cbar.set_label(label='Power [dB]',  size=fontSizeX)
axs[1,0].set_xlabel('Doppler velocity [ms$^{-1}$]', fontsize=16)


mesh = axs[1,1].pcolormesh(doppler, height_spec, spec_noise, vmin=0., vmax=30., cmap='jet', rasterized=True)
cbar = fig.colorbar(mesh, ax=axs[1,1], location='right', aspect=10, use_gridspec=grid)
cbar.set_label(label='Power [dB]',  size=fontSizeX)
axs[1,1].set_xlabel('Doppler velocity [ms$^{-1}$]', fontsize=16)

color_arr = ['blue', 'violet', 'red', 'orange', 'green']
i = 0
delta_y = 0.5
for ind_height in [24,48,72,96,120]:
	axs[2,0].plot(doppler, spec[ind_time_data, ind_height]+delta_y*i, color=color_arr[i], label = str(height_spec[ind_height])+' m')
	axs[2,1].plot(doppler, spec[ind_time_noise, ind_height]+delta_y*i, color=color_arr[i], label = str(height_spec[ind_height])+' m')
	i = i +1 
axs[2,0].set_ylim(0.,35.)
axs[2,1].set_ylim(0.,35.)
axs[2,0].set_xlabel('Doppler velocity [ms$^{-1}$]', fontsize=16)
axs[2,1].set_xlabel('Doppler velocity [ms$^{-1}$]', fontsize=16)
axs[2,0].legend(frameon=False, fontsize=14)
axs[2,1].legend(frameon=False, fontsize=14)

for ax, l in zip(axs[:,:].flatten(), ['(a) Mean Doppler velocity - MRR ', \
	'(b) Reflectivity - W-band', \
	'(c) Doppler spectra for signal ', \
	'(e) Doppler spectra for noise', \
	'(f) Doppler spectra for signal at selected heights', '(d) Doppler spectra for noise at selected heights']):
    ax.text(-0.05, 1.05, l,  fontweight='black', fontsize=14, transform=ax.transAxes)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(3)
    ax.spines["left"].set_linewidth(3)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(n=3))
    ax.tick_params(which='minor', length=7, width=3)
    ax.tick_params(which='major', length=7, width=3)
    #ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(n=3))
    ax.tick_params(axis='both', labelsize=16)



fig.savefig(Path_out+'FigA0.png')