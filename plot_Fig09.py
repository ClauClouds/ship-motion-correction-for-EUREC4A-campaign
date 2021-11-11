# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 10:42:54 2020
@ date; 22 march 2021
@ date; 28 april 2021
@author: cacquist
@goal: this script has the goal of correcting for ship motions the MRR data collected on the ship and remove the
interference noise the data show.
To achieve this goal, the script needs to have two different input files, i.e. the MRR original data and the data
postprocessed by Albert. This double input is needed because the algorithm from Albert provides Doppler spectra pretty much filtered
from noise, but not entirely. The w matrix instead is not filtered, and for this reason, data from the original file are needed.
The filtering algorithm developed in fact, provides a detailed mask for filtering noise that works only when reading the original spectra as
input. To calculate the ship motion correction, W field filtered from nthe interference pattern is needed. For this reason,
the script works as follows:
    - it first applies the filtering mask on the original data. From them, it filters the mean Doppler velocity.
    - from the filtered mean doppler velocity, it applies the algorithm for ship motion correction and obtains the doppler shift corresponding to each pixel.
    - It applies the doppler shift to the Doppler spectra obtained from Albert.
    - it produces a quicklook containing: original mdv, filtered mdv, mdv /ship time matching, doppler spectra shifted vs original.
    - it saves the shifted doppler spectra and the filtering mask applied in a ncdf file to be sent to albert.
@goal: produce daily and hourly quicklooks of MRR data

"""

# importing necessary libraries
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import rcParams
import matplotlib
import netCDF4 as nc4
from netCDF4 import Dataset
import glob
import os.path
import pandas as pd
import numpy as np
import xarray as xr
import scipy.integrate as integrate
from datetime import datetime
from datetime import timedelta
from pathlib import Path
from scipy.signal import chirp, find_peaks, peak_widths, peak_prominences
import custom_color_palette as ccp

################################## functions definitions ####################################
def f_defineSingleColorPalette(colors, minVal, maxVal, step):
    """
    author: Claudia Acquistapace
    date : 13 Jan 2020
    goal: define color palette and connected variables associated to a given range for a variable
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

    Intervals = ccp.range(minVal,maxVal,step)

    # defines a sublist with characteristics of colors that will be used to create a custom palette
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

 # read radar height array for interpolation of model data from one single radar file
 
date_selected = '20200213'
dd = '13'
mm = '02'
yy = '2020'
hh = '01'

mrr_file_original = '/home/cacquist/mrr_paper_plot/20200213_010000-FirstStep-processed.nc'
mrr_file_final = '/home/cacquist/mrr_paper_plot/20200213_MRR_PRO_msm_eurec4a.nc'

dict_plot = {'path':"/home/cacquist/mrr_paper_plot/plots/",
             "varname":'mrr_process', 
         "instr":"MRR_PRO"}

mrr_final              = xr.open_dataset(mrr_file_final)
time                   = mrr_final['time'].values
units_time             = 'seconds since 1970-01-01 00:00:00'
datetimeM              = pd.to_datetime(time, unit ='s', origin='unix')
mrr_slice_hour = mrr_final.sel(time=slice(datetime(2020,2,13,1,0,0), datetime(2020,2,13,1,59,59)))


# extracting only the file for the interested hour
height                 = mrr_slice_hour['height'].values
Ze                     = mrr_slice_hour['Ze'].values
W                      = mrr_slice_hour['fall_speed'].values
time_hour = mrr_slice_hour['time'].values
datetime_hour              = pd.to_datetime(time_hour, unit ='s', origin='unix')



print('file data read for : '+hh)
print('----------------------')

# setting time limits for the plot
timeStartDay = datetime(2020,2,13,1,32,0,0)
timeEndDay = datetime(2020,2,13,1,35,0,0)



# reading original data for paper plots
origData = xr.open_dataset(mrr_file_original)
Ze_orig  = origData['Zea'].values
W_orig   = origData['W'].values
time_orig = origData['time'].values
datetime_orig = pd.to_datetime(time_orig, unit ='s', origin='unix')


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
fig, axs = plt.subplots(2, 2, figsize=(14,9), sharey=True, constrained_layout=True)
[a.get_yaxis().tick_left() for a in axs[:,:].flatten()]
[a.get_xaxis().tick_bottom() for a in axs[:,:].flatten()]

colorsLower = ["#4553c2", "#2b19d9", "#d0d2f0", "#42455e", "#66abf9"]# grigio: 8c8fab
colorsUpper = ["#fd2c3b", "#59413f", "#fdc7cc", "#8f323c", "#e66c4c", "#ae8788"] #MDV
colors_ze_mrr = ["#56ebd3", "#9e4302", "#2af464", "#d6061a", "#1fa198"]

cmap, ticks, norm, bounds = f_defineSingleColorPalette(colors_ze_mrr, -20, 40., 1.)
mesh = axs[0,0].pcolormesh(datetime_orig, height, Ze_orig.T, vmin=-20., vmax=40., cmap=cmap, rasterized=True)
mesh = axs[0,1].pcolormesh(datetime_hour, height, Ze.T, vmin=-20., vmax=40., cmap=cmap, rasterized=True)
axs[0,0].axvline(x=pd.to_datetime(datetime(2020,2,13,1,32,0)), color='black',linewidth=4, linestyle=':')
axs[0,0].axvline(x=pd.to_datetime(datetime(2020,2,13,1,35,0)), color='black', linewidth=4, linestyle=':')
axs[0,1].axvline(x=pd.to_datetime(datetime(2020,2,13,1,32,0)), color='black',linewidth=4, linestyle=':')
axs[0,1].axvline(x=pd.to_datetime(datetime(2020,2,13,1,35,0)), color='black', linewidth=4, linestyle=':')


cbar = fig.colorbar(mesh, ax=axs[0,:], label='Reflectivity [dBz]', location='right', aspect=20, use_gridspec=grid)
cbar.set_label(label='Reflectivity [dBz]', size=fontSizeCbar)
[a.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M')) for a in axs[0,:].flatten()]


cmap, ticks, norm, bounds =  f_defineDoubleColorPalette(colorsLower, colorsUpper, -10., 2., 0.1, 0.)
mesh = axs[1,0].pcolormesh(datetime_orig, height, -W_orig.T, vmin=-10., vmax=2., cmap=cmap, rasterized=True)
axs[1,0].set_xlim(timeStartDay, timeEndDay)
mesh = axs[1,1].pcolormesh(datetime_hour, height, -W.T, vmin=-10, vmax=2., cmap=cmap, rasterized=True)
axs[1,1].set_xlim(timeStartDay, timeEndDay)

[a.xaxis.set_major_formatter(mdates.DateFormatter('%M:%S')) for a in axs[1,:].flatten()]
[a.set_ylim(50., 1200.) for a in axs[:,:].flatten()]
[a.set_xlabel("Time (UTC) [hh:mm]", fontsize=fontSizeX) for a in  axs[0,:].flatten()]
[a.set_xlabel("Time (UTC) [mm:ss]", fontsize=fontSizeX) for a in  axs[1,:].flatten()]

cbar = fig.colorbar(mesh, ax=axs[1,:], label='Doppler velocity [ms$^{-1}$]', location='right', aspect=20, use_gridspec=grid)
cbar.set_label(label='Doppler velocity [$ms^{-1}$]', size=fontSizeCbar)


for ax, l in zip(axs.flatten(), ['(a) Original', '(b) Filtered', '(c) Original', '(d) Filtered and ship motions corrected']):
    ax.text(0, 1.02, l,  fontweight='black', transform=ax.transAxes)


fig.savefig('{path}Fig09.png'.format(**dict_plot))#pathFig+date+'_'+hour+'quicklook_MRR.png')
fig.savefig('{path}Fig09.pdf'.format(**dict_plot))