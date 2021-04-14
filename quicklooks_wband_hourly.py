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




# paths to the different data files and output directories for plots
pathFolderTree  = '/Volumes/Extreme SSD/ship_motion_correction_merian/'
pathFig         = pathFolderTree+'/plots/quicklooks_hourly/'
pathNcDataAnc   = pathFolderTree+'/ncdf_ancillary/'
Path(pathFig).mkdir(parents=True, exist_ok=True)

# generating array of days for the dataset
Eurec4aDays  = pd.date_range(datetime(2020,1,19),datetime(2020,2,19),freq='d')
NdaysEurec4a = len(Eurec4aDays)
radar_name = 'msm'
#%%
for indDay, dayEu in enumerate(Eurec4aDays):



    # extracting strings for yy, dd, mm
    yy              = str(dayEu)[0:4]
    mm              = str(dayEu)[5:7]
    dd              = str(dayEu)[8:10]

    # setting dates strings
    date            = dd+mm+yy      #'04022020'
    dateRadar       = yy[0:2]+mm+dd #'200204'
    dateReverse     = yy+mm+dd      #'20200204'

    # creating output path for figures
    Path(pathFig+'/'+dateReverse+'/').mkdir(parents=True, exist_ok=True)

    pathRadar          = pathFolderTree+'/corrected_data/'+yy+'/'+mm+'/'+dd+'/'
    radarFileList      = np.sort(glob.glob(pathRadar+date+'_*msm94_msm_ZEN_corrected.nc'))

    '''
    Diciamo che dato X dataarray con coordinate tempo, altezza. 
    la funzione restituisce un dataarray di Nan, con coordinate tempo fissa di '3s' (ma che magari fa scopa con la periodicita' di X) 
                                                                                     e coordinata altezza uguale a quella di X. Chiameremo questa variabile VUOTO
    '''
    for iHour, filename in enumerate(radarFileList):

        MeanDopVelDataset  = xr.open_dataset(filename)
        
        # read hour from the selected files
        n = len(pathRadar)
        hh                 = filename[n+9:n+11] #19012020_00msm94_msm_ZEN_corrected
        print('plotting hh ', hh)
        Vd                 = MeanDopVelDataset['vm_corrected_smoothed'].values
        datetimeRadar      = pd.to_datetime(MeanDopVelDataset['time'].values)
        rangeRadar         = MeanDopVelDataset['range'].values
        Vd[Vd == -999.]    = np.nan
        Ze                 = MeanDopVelDataset['ze']
        #Ze[Ze == -999.]    = np.nan
        ZeLog              = 10.*np.log10(Ze)
        Sw                 = MeanDopVelDataset['sw'].values
        #Sw[Sw == -999.]    = np.nan
        Sk                 = MeanDopVelDataset['skew'].values
        #Sk[Sk == -999.]    = np.nan
        timeLocal          = datetimeRadar-timedelta(hours=4)
        
        timegrid = pd.date_range(timeLocal[0],timeLocal[-1],freq='3s')
        a = np.zeros((len(timegrid),len(rangeRadar)))
        a.fill(np.nan)
        Ze_resampled = xr.DataArray(a, coords=[timegrid, rangeRadar], dims=["time", "range"])
        Vd_resampled = xr.DataArray(a, coords=[timegrid, rangeRadar], dims=["time", "range"])
        Sw_resampled = xr.DataArray(a, coords=[timegrid, rangeRadar], dims=["time", "range"])
        Sk_resampled = xr.DataArray(a, coords=[timegrid, rangeRadar], dims=["time", "range"])
        
        for indTime, timeSel in enumerate(timeLocal):
            indFound = np.where(timegrid == timeSel)[0][0]
            print(indFound)
        
        strasuka
        
        # set here the variable you want to plot
        varStringArr = ['Ze', 'Vd', 'Sw', 'Sk' ]


        for ivar in range(len(varStringArr)):

            print('producing quicklook of '+varStringArr[ivar])
            varString = varStringArr[ivar]
            # settings for reflectivity
            if varString == 'Ze':
                mincm = -40.
                maxcm = 26.
                step = 1.
                colors =['#4f8c9d', '#1c4c5e', '#8ae1f9', '#8f0f1b', '#e0bfb4', '#754643', '#ef7e58', '#ff1c5d']
                cmap, ticks, norm, bounds =  f_defineSingleColorPalette(colors, mincm, maxcm, step)
                cbarstr = 'Ze [dBz]'
                strTitle = 'Reflectivity : '+dd+'.'+mm+'.'+yy
                ze_df = pd.DataFrame(ZeLog, index=timeLocal, columns=rangeRadar)
                #ze_res_df = ze_df.asfreq('3s')
                ze_res_df = ze_df.data.resample('3s').asfreq()
                var = ze_res_df.values
                timeResampled =  pd.to_datetime(ze_res_df.index)
                
            # settings for mean Doppler velocity
            elif varString == 'Vd':
                mincm = -4.
                maxcm = 4.
                step = 0.1
                thrs = 0.
                colorsLower = ["#4553c2", "#2b19d9", "#d0d2f0", "#42455e", "#66abf9"]# grigio: 8c8fab
                colorsUpper = ["#fd2c3b", "#59413f", "#fdc7cc", "#8f323c", "#e66c4c", "#ae8788"] #MDV
                cmap, ticks, norm, bounds =  f_defineDoubleColorPalette(colorsLower, colorsUpper, mincm, maxcm, step, thrs)
                vd_df = pd.DataFrame(Vd, index=timeLocal, columns=rangeRadar)
                cbarstr = 'Vd [$ms^{-1}$]'
                strTitle = 'Mean Doppler velocity : '+dd+'.'+mm+'.'+yy
                timegrid = pd.date_range(timeLocal[0],timeLocal[-1],freq='3s')
                vd_res_df = vd_df.reindex(timegrid)
                var = vd_res_df.values
                timeResampled =  pd.to_datetime(vd_res_df.index)

            # settings for Spectral width
            elif varString == 'Sw':
                mincm = 0.
                maxcm = 0.8
                step = 0.01
                #colors = ["#72e5ef", "#460942", "#4dc172", "#cd71b5", "#274c56", "#91ec17", "#b00bd9", "#abc177"]
                colors = [ "#e6bcbe", "#51443f", "#f17a74", "#e81659", "#cdd2df", "#124ac1"]
                cmap, ticks, norm, bounds =  f_defineSingleColorPalette(colors, mincm, maxcm, step)
                sw_df = pd.DataFrame(Sw, index=timeLocal, columns=rangeRadar)
                cbarstr = 'Sw [$ms^{-1}$]'
                strTitle = 'Spectral width : '+dd+'.'+mm+'.'+yy
                timegrid = pd.date_range(timeLocal[0],timeLocal[-1],freq='3s')
                sw_res_df = sw_df.reindex(timegrid)
                var = sw_res_df.values
                timeResampled =  pd.to_datetime(sw_res_df.index)
                
            # settings for skewness
            elif varString == 'Sk':
                mincm = -2.
                maxcm = 2.
                step = 0.1
                thrs = 0.
                colorsLower = ["#082742", "#c5d5f0", "#3693f2", "#555f6e"] # grigio: 8c8fab
                colorsUpper = ["#5f3e3f", "#ddc0bd", "#fa2e55", "#ee7b85", "#851e39", "#e518a9"]
                cmap, ticks, norm, bounds =  f_defineDoubleColorPalette(colorsLower, colorsUpper, mincm, maxcm, step, thrs)
                sk_df = pd.DataFrame(Sk, index=timeLocal, columns=rangeRadar)
                cbarstr = '$Sk$ []'
                strTitle = 'Skewness : '+dd+'.'+mm+'.'+yy
                timegrid = pd.date_range(timeLocal[0],timeLocal[-1],freq='3s')
                sk_res_df = sk_df.reindex(timegrid)
                var = sk_res_df.values
                timeResampled =  pd.to_datetime(sk_res_df.index)
            
#%%
timeStartDay = timeLocal[0]
timeEndDay   = timeLocal[-1]
hmin         = 0.
hmax=2500.
ystring = cbarstr
labelsizeaxes   = 16
fontSizeTitle = 16
fontSizeX = 16
fontSizeY = 16
cbarAspect = 10
fontSizeCbar = 16
yyPlot = datetimeRadar[0].year
mmPlot = datetimeRadar[0].month
ddPlot = datetimeRadar[0].day
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

cax = ax.pcolormesh(timeResampled, rangeRadar, var.transpose(), vmin=mincm, vmax=maxcm, cmap=cmap)
ax.set_ylim(hmin,hmax)                                               # limits of the y-axesn  cmap=plt.cm.get_cmap("viridis", 256)
ax.set_xlim(timeStartDay, timeEndDay)                                 # limits of the x-axes
ax.set_title(strTitle, fontsize=fontSizeTitle, loc='left')
ax.set_xlabel("Local Time (UTC - 4h) [hh:mm]", fontsize=fontSizeX)
ax.set_ylabel("Height [$m$]", fontsize=fontSizeY)
cbar = fig.colorbar(cax, orientation='vertical', aspect=cbarAspect)
cbar.set_label(label=cbarstr, size=fontSizeCbar)
cbar.ax.tick_params(labelsize=labelsizeaxes)
# Turn on the frame for the twin axis, but then hide all
# but the bottom spine

fig.tight_layout()
plt.savefig(pathFig+'/'+dateReverse+'/'+yy+mm+dd+'_'+hh+'_'+radar_name+'_'+varString+'_quicklooks.png', format='png', bbox_inches='tight')

#%%
        # -------------------------------------------------------------------
        # --- NEXT PLOT -----------------------------------------------------
        # -------------------------------------------------------------------
        data10 = MeanDopVelDataset['lwp']
        data10 = np.ma.masked_invalid(data10)
        data11 = MeanDopVelDataset['rr']
        data11 = np.ma.masked_invalid(data11)
        data12 = MeanDopVelDataset['ta']-273.15 # to C
        data12 = np.ma.masked_invalid(data12)
        data13 = MeanDopVelDataset['rh']
        data13 = np.ma.masked_invalid(data13)
        data14 = MeanDopVelDataset['pa']
        data14 = np.ma.masked_invalid(data14)
        fs = 14.
        time_dd = yy+'-'+mm+'-'+dd
        t_plot = timeLocal
        data11 = MeanDopVelDataset['tb']
        data11 = np.ma.masked_invalid(data11)
        time1 = timeStartDay
        time2 =  timeEndDay
        fig, axes = plt.subplots(figsize=[12.0, 15.0], nrows=5, sharex=True)
        # ---------------------------------------------------------------------
        axes[0].plot(t_plot, data10, linestyle='None', marker='o', markersize=3., mec='b', mfc='b')
        axes[0].set_ylabel('LWP [g m$^{-2}$]', fontsize=fs)
        axes[0].text(0.01, 0.86, time_dd+" - Liquid Water Path "+radar_name+',(RPG processing)',
                     transform=axes[0].transAxes, ha="left", fontsize=fs+1)
        axes[0].set_xlim(time1, time2)  # limets of the x-axes
        axes[0].set_ylim(-1., np.nanmax(data10)+10.)  # limets of the x-axes)  # limets of the x-axes
        axes[0].grid(True, which="both")
        # ----------------------------------
        axes[1].plot(t_plot, data11, linestyle='None', marker='o', markersize=3., mec='m', mfc='m')
        axes[1].set_ylabel('RR [mm h$^{-1}$]', fontsize=fs)
        axes[1].text(0.01, 0.86, time_dd+" - Met.-station rain rate ",
                     transform=axes[1].transAxes, ha="left", fontsize=fs+1)
        axes[1].set_xlim(time1, time2)  # limets of the x-axes
        axes[1].set_ylim(-0.25, np.nanmax(data11)+1.)  # limets of the x-axes
        axes[1].grid(True, which="both")
        # ---------------------------------------------------------------------
        axes[2].plot(t_plot, data12, linestyle='None', marker='o', markersize=3., mec='r', mfc='r')
        axes[2].set_ylabel('T [$^{o}$C]', fontsize=fs)
        axes[2].text(0.01, 0.86, time_dd+" - Met.-station air temperature ",
                     transform=axes[2].transAxes, ha="left", fontsize=fs+1)
        axes[2].set_xlim(time1, time2)  # limets of the x-axes
        axes[2].set_ylim(np.nanmin(data12)-1, np.nanmax(data12)+1)  # limets of the x-axes
        axes[2].grid(True, which="both")
        # ---------------------------------------------------------------------
        axes[3].axhline(y=70., color='gray', linestyle='-', linewidth = 5, alpha=0.4)
        axes[3].plot(t_plot, data13, linestyle='None', marker='o', markersize=3., mec='c', mfc='c')
        axes[3].set_ylabel('rel. humidity [%]', fontsize=fs)
        axes[3].text(0.01, 0.86, time_dd+" - Met.-station relative humidity ",
                     transform=axes[3].transAxes, ha="left", fontsize=fs+1)
        axes[3].set_xlim(time1, time2)  # limets of the x-axes
        axes[3].set_ylim(0.0, 100.)  # limets of the x-axes
        axes[3].grid(True, which="both")
        # ---------------------------------------------------------------------
        axes[4].plot(t_plot, data14, linestyle='None', marker='o', markersize=3., mec='b', mfc='b')
        axes[4].set_ylabel('pressure [HPa]', fontsize=fs)
        axes[4].set_xlabel('local time (UTC -4)', fontsize=fs)
        axes[4].text(0.01, 0.86, time_dd+" - Met.-station atmospheric pressure  ",
                     transform=axes[4].transAxes, ha="left", fontsize=fs+1)
        axes[4].set_xlim(time1, time2)  # limets of the x-axes
        #axes[4].set_ylim(-0.1, np.nanmax(data14)+0.25)  # limets of the x-axes
        axes[4].grid(True, which="both")
        print('met data overview quicklook produced')

        # -----------------------------------------------------------------------
        plt.savefig(pathFig+'/'+dateReverse+'/'+yy+mm+dd+'_'+hh+'_'+radar_name+'_metstation.png')
        print('------------ quicklook done ------------')
