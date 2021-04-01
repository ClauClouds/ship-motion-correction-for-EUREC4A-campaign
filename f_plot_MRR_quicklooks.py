#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 10:42:54 2020
@ date; 22 march 2021
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

#import atmos

from functions_essd import f_closest
from functions_essd import f_calculateMomentsCol
from functions_essd import f_readAndMergeRadarDataDay_DopplerCorrection
from functions_essd import f_readAndMergeRadarDataDay
from functions_essd import generate_preprocess
from functions_essd import generate_preprocess_zen
from scipy.interpolate import CubicSpline
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




# reading file list and its path
# establishing paths for data input and output
pathFolderTree  = '/Volumes/Extreme SSD/ship_motion_correction_merian/'
pathNcData      = pathFolderTree+'/mrr/corrected_2ways/'
pathFig         = pathFolderTree+'/mrr/plots/mrr/'

DataList        = np.sort(glob.glob(pathNcData+'*.nc'))

print('filelist found: ', DataList)
# loop on files from Albert
for indHour,file in enumerate(DataList):
    date_hour       = file[len(pathNcData):len(pathNcData)+8]

    yy              = date_hour[4:8]
    mm              = date_hour[2:4]
    dd              = date_hour[0:2]
    hh              = file[len(pathNcData)+9:len(pathNcData)+11]

    print(yy,mm,dd,hh)
    # setting dates strings
    date            = dd+mm+yy      #'04022020'
    dateRadar       = yy[0:2]+mm+dd #'200204'
    dateReverse     = yy+mm+dd      #'20200204'

    #def dict for data plotting
    radarFileName   = DataList[indHour]
    print('file :', radarFileName)

    # reading corresponding metek file
    #MetekFile       = pathFolderTree+'/mrr/'+yy+'/'+mm+'/'+dd+'/'+yy+mm+dd+'_'+hh+'0000.nc'

    # read radar height array for interpolation of model data from one single radar file
    radarDatatest          = xr.open_dataset(radarFileName)
    height                 = radarDatatest['Height'].values
    mask                   = radarDatatest['mask'].values
    RR                     = radarDatatest['RR'].values
    Ze                     = radarDatatest['Zea'].values
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


    # set here the variable you want to plot
    varStringArr = ['Ze', 'RR', 'LWC', 'W', ]


    for ivar,varSel in enumerate(varStringArr):

        print('producing quicklook of '+varStringArr[ivar])
        varString = varStringArr[ivar]
        # settings for reflectivity
        if varString == 'Ze':
            mincm = -20.
            maxcm = 40.
            step = 1.
            colors =['#4f8c9d', '#1c4c5e', '#8ae1f9', '#8f0f1b', '#e0bfb4', '#754643', '#ef7e58', '#ff1c5d']
            cmap, ticks, norm, bounds =  f_defineSingleColorPalette(colors, mincm, maxcm, step)
            var = Ze
            cbarstr = 'Ze [dBz]'
            strTitle = 'Attenuated reflectivity : '+dd+'.'+mm+'.'+yy
            dict_plot = {'path':pathFig, 'date':date, 'hour':str(hh), 'varname':varSel}

            # settings for mean Doppler velocity
        elif varString == 'W':
                mincm = -10.
                maxcm = 2.
                step = 0.1
                thrs = 0.
                colorsLower = ["#4553c2", "#2b19d9", "#d0d2f0", "#42455e", "#66abf9"]# grigio: 8c8fab
                colorsUpper = ["#fd2c3b", "#59413f", "#fdc7cc", "#8f323c", "#e66c4c", "#ae8788"] #MDV
                cmap, ticks, norm, bounds =  f_defineDoubleColorPalette(colorsLower, colorsUpper, mincm, maxcm, step, thrs)
                var = -W
                cbarstr = 'Vd [$ms^{-1}$]'
                strTitle = 'Fall speed : '+dd+'.'+mm+'.'+yy
                dict_plot = {'path':pathFig, 'date':date, 'hour':str(hh), 'varname':varSel}

            # settings for Spectral width
        elif varString == 'RR':
                mincm = 0.
                maxcm = 0.8
                step = 0.01
                #colors = ["#72e5ef", "#460942", "#4dc172", "#cd71b5", "#274c56", "#91ec17", "#b00bd9", "#abc177"]
                colors = [ "#e6bcbe", "#51443f", "#f17a74", "#e81659", "#cdd2df", "#124ac1"]
                cmap, ticks, norm, bounds =  f_defineSingleColorPalette(colors, mincm, maxcm, step)
                var = RR
                cbarstr = 'RR [$mmh^{-1}$]'
                strTitle = 'Rain rate : '+dd+'.'+mm+'.'+yy
                dict_plot = {'path':pathFig, 'date':date, 'hour':str(hh), 'varname':varSel}

            # settings for skewness
        elif varString == 'LWC':
                mincm = 0.
                maxcm = 1.
                step = 0.01
                thrs = 0.
                colors = [ "#e6bcbe", "#51443f", "#f17a74", "#e81659", "#cdd2df", "#124ac1"]
                cmap, ticks, norm, bounds =  f_defineSingleColorPalette(colors, mincm, maxcm, step)
                var = LWC
                cbarstr = '$LWC$ [$gm^{-3}$]'
                strTitle = 'Liquid water content : '+dd+'.'+mm+'.'+yy
                dict_plot = {'path':pathFig, 'date':date, 'hour':str(hh), 'varname':varSel}

        timeStartDay = datetime(2020,2,13,1,32,0)#datetimeM[0]#
        timeEndDay   = datetime(2020,2,13,1,35,0)#datetimeM[-1]#
        hmin         = 0.
        hmax         = 1200.
        ystring       = cbarstr
        labelsizeaxes = 16
        fontSizeTitle = 16
        fontSizeX     = 16
        fontSizeY     = 16
        cbarAspect    = 10
        fontSizeCbar  = 16
        yyPlot        = datetimeM[0].year
        mmPlot        = datetimeM[0].month
        ddPlot        = datetimeM[0].day
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12,6))
        rcParams['font.sans-serif'] = ['Tahoma']
        matplotlib.rcParams['savefig.dpi'] = 100
        plt.gcf().subplots_adjust(bottom=0.15)
        fig.tight_layout()
        ax = plt.subplot(1,1,1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        matplotlib.rc('xtick', labelsize=labelsizeaxes)  # sets dimension of ticks in the plots
        matplotlib.rc('ytick', labelsize=labelsizeaxes)  # sets dimension of ticks in the plots
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax.xaxis.set_minor_formatter(mdates.DateFormatter("%H:%M"))
        ax.xaxis_date()
        ax.axvline(x=pd.to_datetime(datetime(yyPlot,mmPlot,ddPlot,6,30,0,0)), color='black',linewidth=4, linestyle=':')
        ax.axvline(x=pd.to_datetime(datetime(yyPlot,mmPlot,ddPlot,19,15,0,0)), color='black', linewidth=4, linestyle=':')

        cax = ax.pcolormesh(datetimeM, height, var.transpose(), vmin=mincm, vmax=maxcm, cmap=cmap)
        ax.set_ylim(hmin,hmax)                                               # limits of the y-axesn  cmap=plt.cm.get_cmap("viridis", 256)
        ax.set_xlim(timeStartDay, timeEndDay)                                 # limits of the x-axes
        ax.set_title(strTitle, fontsize=fontSizeTitle, loc='left')
        ax.set_xlabel("Time (UTC) [hh:mm]", fontsize=fontSizeX)
        ax.set_ylabel("Height [$m$]", fontsize=fontSizeY)
        cbar = fig.colorbar(cax, orientation='vertical', aspect=cbarAspect)
        cbar.set_label(label=cbarstr, size=fontSizeCbar)
        cbar.ax.tick_params(labelsize=labelsizeaxes)
        # Turn on the frame for the twin axis, but then hide all
        # but the bottom spine

        fig.tight_layout()
        fig.savefig('{path}{date}_{hour}_{varname}_quicklook_MRR.png'.format(**dict_plot), bbox_inches='tight')
        #plt.savefig(pathFig+yy+mm+dd+'_'+radar_name+'_'+varString+'_quicklooks.png', format='png', bbox_inches='tight')




# plot quicklook of filtered and corrected mdv for checking
labelsizeaxes   = 14
fontSizeTitle   = 16
fontSizeX       = 16
fontSizeY       = 16
cbarAspect      = 10
fontSizeCbar    = 16
rcParams['font.sans-serif'] = ['Tahoma']
matplotlib.rcParams['savefig.dpi'] = 100
plt.rcParams.update({'font.size':10})
grid = True
fig, axs = plt.subplots(2, 2, figsize=(14,9), sharey=True, constrained_layout=True)

# build colorbar
mesh = axs[0,0].pcolormesh(datetimeM, height, Ze.T, vmin=-10, vmax=40, cmap='jet', rasterized=True)
axs[0,0].set_title('Original')
axs[0,0].spines["top"].set_visible(False)
axs[0,0].spines["right"].set_visible(False)
axs[0,0].get_xaxis().tick_bottom()
axs[0,0].get_yaxis().tick_left()
axs[0,0].set_xlim(datetimeM[0], datetimeM[-1])


mesh = axs[0,1].pcolormesh(datetimeM, height, RR.T, vmin=0., vmax=2.,  cmap='jet', rasterized=True)
#[a.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M')) for a in axs.flatten()]
axs[0,1].spines["top"].set_visible(False)
axs[0,1].spines["right"].set_visible(False)
axs[0,1].get_xaxis().tick_bottom()
axs[0,1].get_yaxis().tick_left()
axs[0,1].set_title('Filtered', fontsize=fontSizeX)
axs[0,1].set_xlim(datetimeM[0], datetimeM[-1])

axs[0,0].set_ylabel('Height   [m]', fontsize=fontSizeX)
axs[0,0].set_xlabel('time [hh:mm]', fontsize=fontSizeX)
axs[0,1].set_xlabel('time [hh:mm]', fontsize=fontSizeX)
axs[1,0].set_xlabel('time [mm:ss]', fontsize=fontSizeX)
axs[1,1].set_xlabel('time [mm:ss]', fontsize=fontSizeX)
[a.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M')) for a in axs[0,:].flatten()]

mesh = axs[1,0].pcolormesh(datetimeM, height, LWC.T, vmin=0., vmax=0.5, cmap='jet', rasterized=True)
axs[1,0].set_title('Original')
axs[1,0].spines["top"].set_visible(False)
axs[1,0].spines["right"].set_visible(False)
axs[1,0].get_xaxis().tick_bottom()
axs[1,0].get_yaxis().tick_left()
axs[1,0].set_xlim(datetimeM[0], datetimeM[-1])

mesh = axs[1,1].pcolormesh(datetimeM, height, W.T, vmin=-10, vmax=20, cmap='jet', rasterized=True)
axs[1,1].set_title('ship motions corrected')
axs[1,1].spines["top"].set_visible(False)
axs[1,1].spines["right"].set_visible(False)
axs[1,1].get_xaxis().tick_bottom()
axs[1,1].get_yaxis().tick_left()
axs[1,1].set_xlim(datetimeM[0], datetimeM[-1])
[a.xaxis.set_major_formatter(mdates.DateFormatter('%M:%S')) for a in axs[1,:].flatten()]
fig.colorbar(mesh, ax=axs[:,-1], label='Doppler velocity [$ms^{-1}$]', location='right', aspect=60, use_gridspec=grid)
for ax, l in zip(axs.flatten(), ['(a)', '(b)', '(c)', '(d)']):
    ax.text(0, 1.02, l,  fontweight='black', transform=ax.transAxes)
fig.savefig('{path}{date}_{hour}_quicklook_MRR.png'.format(**dict_plot))#pathFig+date+'_'+hour+'quicklook_MRR.png')
fig.savefig('{path}{date}_{hour}_quicklook_MRR.pdf'.format(**dict_plot))
#fig.savefig(pathFig+date+'_'+hour+'quicklook_MRR.png')
#fig.savefig(pathFig+date+'_'+hour+'quicklook_MRR.pdf')
