#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 10:42:54 2020
@ date; 28 april 2021
@author: cacquist
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
pathNcData      = '/Volumes/Extreme SSD/ship_motion_correction_merian/mrr/second_step/'
pathFig         = '/Volumes/Extreme SSD/ship_motion_correction_merian/mrr/plots/'
pathFigHour     = '/Volumes/Extreme SSD/ship_motion_correction_merian/mrr/plots/hourly_plots/'
DataList        = np.sort(glob.glob(pathNcData+'*eurec4a.nc'))

print('files found: ', DataList)


for indDay,file in enumerate(DataList):
    
    date_hour       = file[len(pathNcData):len(pathNcData)+8]

    yy              = date_hour[0:4]
    mm              = date_hour[4:6]
    dd              = date_hour[6:8]

    print(yy,mm,dd)
    
    # setting dates strings
    date            = dd+mm+yy      #'04022020'
    dateRadar       = yy[0:2]+mm+dd #'200204'
    dateReverse     = yy+mm+dd      #'20200204'

    radarFileName   = DataList[indDay]
    print('file :', radarFileName)


    # read MRR data
    radarData          = xr.open_dataset(radarFileName)
    

    # set here the variable you want to plot
    varStringArr = ['Zea', 'RR', 'LWC', 'W', ]


    for ivar,varSel in enumerate(varStringArr):

        print('producing quicklook of '+varStringArr[ivar])
        varString = varStringArr[ivar]
        # settings for reflectivity
        if varString == 'Zea':
            mincm = -20.
            maxcm = 40.
            step = 1.
            colors =['#4f8c9d', '#1c4c5e', '#8ae1f9', '#8f0f1b', '#e0bfb4', '#754643', '#ef7e58', '#ff1c5d']
            cmap, ticks, norm, bounds =  f_defineSingleColorPalette(colors, mincm, maxcm, step)
            var = radarData.Zea.values
            cbarstr = 'Ze [dBz]'
            strTitle = 'MRR - Equivalent reflectivity attenuated: '+dd+'.'+mm+'.'+yy
            dict_plot = {'path':pathFig, 'date':date, 'varname':varSel}

            # settings for mean Doppler velocity
        elif varString == 'W':
            mincm = -10.
            maxcm = 2.
            step = 0.1
            thrs = 0.
            colorsLower = ["#4553c2", "#2b19d9", "#d0d2f0", "#42455e", "#66abf9"]# grigio: 8c8fab
            colorsUpper = ["#fd2c3b", "#59413f", "#fdc7cc", "#8f323c", "#e66c4c", "#ae8788"] #MDV
            cmap, ticks, norm, bounds =  f_defineDoubleColorPalette(colorsLower, colorsUpper, mincm, maxcm, step, thrs)
            var = -radarData.fall_speed.values
            cbarstr = 'Fall speed [$ms^{-1}$]'
            strTitle = 'MRR - Fall speed : '+dd+'.'+mm+'.'+yy
            dict_plot = {'path':pathFig, 'date':date, 'varname':varSel}

            # settings for Spectral width
        elif varString == 'RR':
            mincm = 0.
            maxcm = 0.8
            step = 0.01
            #colors = ["#72e5ef", "#460942", "#4dc172", "#cd71b5", "#274c56", "#91ec17", "#b00bd9", "#abc177"]
            colors = [ "#e6bcbe", "#51443f", "#f17a74", "#e81659", "#cdd2df", "#124ac1"]
            cmap, ticks, norm, bounds =  f_defineSingleColorPalette(colors, mincm, maxcm, step)
            var = radarData.rain_rate.values
            cbarstr = 'Rainfall rate [$mmh^{-1}$]'
            strTitle = 'MRR - Rainfall rate : '+dd+'.'+mm+'.'+yy
            dict_plot = {'path':pathFig, 'date':date, 'varname':varSel}

            # settings for skewness
        elif varString == 'LWC':
            mincm = 0.
            maxcm = 1.
            step = 0.01
            thrs = 0.
            colors = [ "#e6bcbe", "#51443f", "#f17a74", "#e81659", "#cdd2df", "#124ac1"]
            cmap, ticks, norm, bounds =  f_defineSingleColorPalette(colors, mincm, maxcm, step)
            var = radarData.liquid_water_content.values
            cbarstr = '$LWC$ [$gm^{-3}$]'
            strTitle = 'MRR - Liquid water content : '+dd+'.'+mm+'.'+yy
            dict_plot = {'path':pathFig, 'date':date,  'varname':varSel}

        timeStartDay = radarData.time.values[0]#datetime(2020,2,13,1,32,0)#datetimeM[0]#
        timeEndDay   = radarData.time.values[-1]#datetime(2020,2,13,1,35,0)#datetimeM[-1]#
        hmin         = 0.
        hmax         = 1200.
        ystring       = cbarstr
        labelsizeaxes = 16
        fontSizeTitle = 16
        fontSizeX     = 16
        fontSizeY     = 16
        cbarAspect    = 10
        fontSizeCbar  = 16
        yyPlot        = pd.to_datetime(timeStartDay).year
        mmPlot        = pd.to_datetime(timeStartDay).month
        ddPlot        = pd.to_datetime(timeStartDay).day
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

        cax = ax.pcolormesh(pd.to_datetime(radarData.time.values), radarData.height.values, var.transpose(), vmin=mincm, vmax=maxcm, cmap=cmap)
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
        fig.savefig('{path}{date}_{varname}_quicklook_MRR.png'.format(**dict_plot), bbox_inches='tight')
        #plt.savefig(pathFig+yy+mm+dd+'_'+radar_name+'_'+varString+'_quicklooks.png', format='png', bbox_inches='tight')
        
    

    # producing now hourly plots
    hourArr     = pd.date_range(pd.to_datetime(radarData.time.values[0]),pd.to_datetime(radarData.time.values[-1]),freq='h')

    # loop on the hours of the day
    for indHour in range(len(hourArr)-1):
        
        # establishing the string associated to the hour
        hour = str(hourArr[indHour].hour)
        if len(hour) == 1:
            hour = '0'+hour
        
        # slicing the dataset for the corresponding time interval
        if indHour != 23:
            slicedData = radarData.sel(time=slice(hourArr[indHour], hourArr[indHour+1]))
        else:
            slicedData = radarData.sel(time=slice(hourArr[indHour], pd.to_datetime(datetime(yyPlot,mmPlot,ddPlot,23,59,59))))
        
    
        # set here the variable you want to plot
        varStringArr = ['Zea', 'RR', 'LWC', 'W', ]
    
    
        for ivar,varSel in enumerate(varStringArr):
    
            print('producing quicklook of '+varStringArr[ivar])
            varString = varStringArr[ivar]
            # settings for reflectivity
            if varString == 'Zea':
                mincm = -20.
                maxcm = 40.
                step = 1.
                colors =['#4f8c9d', '#1c4c5e', '#8ae1f9', '#8f0f1b', '#e0bfb4', '#754643', '#ef7e58', '#ff1c5d']
                cmap, ticks, norm, bounds =  f_defineSingleColorPalette(colors, mincm, maxcm, step)
                var = slicedData.Zea.values
                cbarstr = 'Ze [dBz]'
                strTitle = 'MRR - Equivalent reflectivity attenuated: '+dd+'.'+mm+'.'+yy
                dict_plot = {'path':pathFigHour, 'date':date, 'hour':hour, 'varname':varSel}
    
                # settings for mean Doppler velocity
            elif varString == 'W':
                mincm = -10.
                maxcm = 2.
                step = 0.1
                thrs = 0.
                colorsLower = ["#4553c2", "#2b19d9", "#d0d2f0", "#42455e", "#66abf9"]# grigio: 8c8fab
                colorsUpper = ["#fd2c3b", "#59413f", "#fdc7cc", "#8f323c", "#e66c4c", "#ae8788"] #MDV
                cmap, ticks, norm, bounds =  f_defineDoubleColorPalette(colorsLower, colorsUpper, mincm, maxcm, step, thrs)
                var = -slicedData.fall_speed.values
                cbarstr = 'Fall speed [$ms^{-1}$]'
                strTitle = 'MRR - Fall speed : '+dd+'.'+mm+'.'+yy
                dict_plot = {'path':pathFigHour, 'date':date,  'hour':hour, 'varname':varSel}
    
                # settings for Spectral width
            elif varString == 'RR':
                mincm = 0.
                maxcm = 0.8
                step = 0.01
                #colors = ["#72e5ef", "#460942", "#4dc172", "#cd71b5", "#274c56", "#91ec17", "#b00bd9", "#abc177"]
                colors = [ "#e6bcbe", "#51443f", "#f17a74", "#e81659", "#cdd2df", "#124ac1"]
                cmap, ticks, norm, bounds =  f_defineSingleColorPalette(colors, mincm, maxcm, step)
                var = slicedData.rain_rate.values
                cbarstr = 'Rainfall rate [$mmh^{-1}$]'
                strTitle = 'MRR - Rainfall rate : '+dd+'.'+mm+'.'+yy
                dict_plot = {'path':pathFigHour, 'date':date,  'hour':hour, 'varname':varSel}
    
                # settings for skewness
            elif varString == 'LWC':
                mincm = 0.
                maxcm = 1.
                step = 0.01
                thrs = 0.
                colors = [ "#e6bcbe", "#51443f", "#f17a74", "#e81659", "#cdd2df", "#124ac1"]
                cmap, ticks, norm, bounds =  f_defineSingleColorPalette(colors, mincm, maxcm, step)
                var = slicedData.liquid_water_content.values
                cbarstr = '$LWC$ [$gm^{-3}$]'
                strTitle = 'MRR - Liquid water content : '+dd+'.'+mm+'.'+yy
                dict_plot = {'path':pathFigHour, 'date':date,   'hour':hour, 'varname':varSel}
    
            timeStartDay = slicedData.time.values[0]#datetime(2020,2,13,1,32,0)#datetimeM[0]#
            timeEndDay   = slicedData.time.values[-1]#datetime(2020,2,13,1,35,0)#datetimeM[-1]#
            hmin         = 0.
            hmax         = 1200.
            ystring       = cbarstr
            labelsizeaxes = 16
            fontSizeTitle = 16
            fontSizeX     = 16
            fontSizeY     = 16
            cbarAspect    = 10
            fontSizeCbar  = 16
            yyPlot        = pd.to_datetime(timeStartDay).year
            mmPlot        = pd.to_datetime(timeStartDay).month
            ddPlot        = pd.to_datetime(timeStartDay).day
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
    
            cax = ax.pcolormesh(pd.to_datetime(slicedData.time.values), slicedData.height.values, var.transpose(), vmin=mincm, vmax=maxcm, cmap=cmap)
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
#%%
