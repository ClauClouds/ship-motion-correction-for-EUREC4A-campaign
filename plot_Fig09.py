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
 


radarDatatest          = xr.open_dataset(radarFileName)
height                 = radarDatatest['Height'].values
mask                   = radarDatatest['mask'].values
RR                     = radarDatatest['RR'].values
Ze                     = radarDatatest['Ze'].values
LWC                    = radarDatatest['LWC'].values
W                      = radarDatatest['W'].values
time                   = radarDatatest['time'].values
units_time             = 'seconds since 1970-01-01 00:00:00'

datetimeM              = pd.to_datetime(time, unit ='s', origin='unix')

print('file data read for : '+date_hour)
print('----------------------')
# filtering data based on the mask
Ze[mask == 0]  = np.nan
RR[mask == 0]  = np.nan
LWC[mask == 0] = np.nan
W[mask == 0]   = np.nan


# reading corresponding metek file
MetekFile       = pathFolderTree+'/mrr/'+yy+'/'+mm+'/'+dd+'/'+yy+mm+dd+'_'+hh+'0000.nc'
print('file metek: ', MetekFile)

# reading original data for paper plots
origData = xr.open_dataset(MetekFile)
Ze_orig  = origData['Zea'].values
W_orig   = origData['VEL'].values


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
cmap, ticks, norm, bounds =  f_defineDoubleColorPalette(colorsLower, colorsUpper, -10., 2., 0.1, 0.)
mesh = axs[1,0].pcolormesh(datetimeM, height, -W_orig.T, vmin=-10., vmax=2., cmap=cmap, rasterized=True)
axs[1,0].set_xlim(timeStartDay, timeEndDay)
mesh = axs[1,1].pcolormesh(datetimeM, height, -W.T, vmin=-10, vmax=2., cmap=cmap, rasterized=True)
axs[1,1].set_xlim(timeStartDay, timeEndDay)
[a.xaxis.set_major_formatter(mdates.DateFormatter('%M:%S')) for a in axs[1,:].flatten()]
[a.set_ylim(50., 1200.) for a in axs[:,:].flatten()]
cbar = fig.colorbar(mesh, ax=axs[1,:], label='Doppler velocity [$ms^{-1}$]', location='right', aspect=20, use_gridspec=grid)
cbar.set_label(label='Doppler velocity [$ms^{-1}$]', size=fontSizeCbar)
for ax, l in zip(axs.flatten(), ['(a) Original', '(b) Filtered', '(c) Original', '(d) Filtered and ship motions corrected']):
    ax.text(0, 1.02, l,  fontweight='black', transform=ax.transAxes)
fig.savefig('{path}{date}_{hour}_quicklook_MRR.png'.format(**dict_plot))#pathFig+date+'_'+hour+'quicklook_MRR.png')
fig.savefig('{path}{date}_{hour}_quicklook_MRR.pdf'.format(**dict_plot))