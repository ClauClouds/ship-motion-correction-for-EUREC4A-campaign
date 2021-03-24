#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 10:50:04 2020

@author  : Claudia Acquistapace
@date    : 20.10.2020
@goal    : function to read wind profiles from ICON LEM for the times of the gaps, resample them on the radar range resolution and 
store them in a matrix having dimension (time, height). For the times in which the table is working, the wind profiles of speed and 
direction are all nan, while for the times in which the table is stuck, it resamples and stores the profiles from ICON-LEM

input:  
    timeship (time array for the entire campaign)
    range height (radar range height)
    wind speed profiles for the gap (model data)
    wind direction profiles for the gap (model data)
output: 
    windspeed matrix (dimTime, dimHeightRadar)
    wind direction matrix (dimTime, dimHeightRadar)

        
"""
import numpy as np
import glob
import xarray as xr
from datetime import datetime
import pandas as pd


# generating array of days for the dataset
Eurec4aDays  = pd.date_range(datetime(2020,1,19),datetime(2020,2,19),freq='d')
NdaysEurec4a = len(Eurec4aDays)

for indDayEu in range(16, NdaysEurec4a):
    indDayEu = 6
    # excluding day without data
    #if Eurec4aDays[indDayEu] != datetime(2020,1,22):
    
    # select a date
    dayEu           = Eurec4aDays[indDayEu]
    
    # extracting strings for yy, dd, mm
    yy              = str(dayEu)[0:4]
    mm              = str(dayEu)[5:7]
    dd              = str(dayEu)[8:10]
    print('processing date :'+yy+'-'+mm+'-'+dd)
    
    # reading radar path and single radar file to extract the height
    print('* reading radar data for the height')
    pathRadar       = '/Volumes/Extreme SSD/ship_motion_correction_merian/w_band_radar_data/EUREC4Aprocessed_20201102/'+yy+'/'+mm+'/'+dd+'/'
    radarFileList   = np.sort(glob.glob(pathRadar+'*ZEN_compact_v2.nc'))
    radarDayDataset = xr.open_dataset(radarFileList[0]) #f_readAndMergeRadarDataDay(radarFileList)
    
    # reading radar range gates [m]
    rangeRadar      = radarDayDataset['range'].values
    rangeRadarTop   = rangeRadar[-1]
    
    # generating time array with resolution of 1s for the selected day
    timeDayStart    = datetime(int(yy), int(mm), int(dd),0,0,0)
    timeDayEnd      = datetime(int(yy), int(mm), int(dd),23,59,59)
    datetimeShipDay = pd.date_range(start=timeDayStart, end=timeDayEnd, freq='1S')
    
    # setting dimensions of the output matrices: matrices for the whole day,
    # nan when table works, with data for time intervals where table is stuck
    dimTime         = len(datetimeShipDay)
    dimHeight       = len(rangeRadar)
    
    # defining xr dataArray output of wind speed, wind direction, lats, lons of icon profiles for the day
    uDay   = np.zeros((dimTime,dimHeight))
    uDay.fill(np.nan)
    vDay     = np.zeros((dimTime,dimHeight))
    vDay.fill(np.nan)
    latsDay         = np.zeros((dimTime))
    latsDay.fill(np.nan)
    lonsDay         = np.zeros((dimTime))
    lonsDay.fill(np.nan)
    
    
    # setting path and filelist for wind data from model
    pathwindIcon    = '/Volumes/Extreme SSD/ship_motion_correction_merian/wind_data_iconlem/'
    
    # selecting all gaps files for the day
    windFileListDay = np.sort(glob.glob(pathwindIcon+'*_'+yy+'-'+mm+'-'+dd+' *'))
    
    # setting the number of files to be read
    NwindFiles      = len(windFileListDay)
    
    # loop on number of wind files for one day
    for indFileGap in range(NwindFiles):
    
        print('processing file gap '+str(indFileGap)+' of total '+str(NwindFiles))
        # setting filename
        windFile        = windFileListDay[indFileGap]
        
        # reading wind data
        windData        = xr.open_dataset(windFile)
        datetimeGap     = windData['time'].values
        startGap        = datetimeGap[0]
        endGap          = datetimeGap[-1]
        datetimeShipGap = datetimeShipDay[(datetimeShipDay >=pd.to_datetime(startGap)) * (datetimeShipDay < pd.to_datetime(endGap))]

        # removing duplicates if they exist
        if (len(set(datetimeGap)) != len(datetimeGap)):
            print('* removing duplicates in time array from wind data')
            windData = windData.sel(time=~windData.indexes['time'].duplicated())
        
        # interpolating wind data on the time array of the gap selected from the day
        windData        = windData.interp(time=datetimeShipGap)
                
        # assigining heightIcon variable as coordinate in place of height (ordinal index of height) in windData dataset
        iconHeight      = windData['heightIcon'].values[0,:]
        windData        = windData.drop('height')
        windData        = windData.assign_coords({'height':iconHeight})
    
        # interpolating icon data on radar range gates
        windDataRadar   = windData.interp(height=rangeRadar)
    
        # copying wind speed and wind direction on the output matrices
        uDay[(datetimeShipDay >=startGap) * (datetimeShipDay < endGap),:]  = windDataRadar['u'].values
        vDay[(datetimeShipDay >=startGap) * (datetimeShipDay < endGap),:]  = windDataRadar['v'].values
        latsDay[(datetimeShipDay >=startGap) * (datetimeShipDay < endGap)] = windDataRadar['lats'].values
        lonsDay[(datetimeShipDay >=startGap) * (datetimeShipDay < endGap)] = windDataRadar['lons'].values
        
        
    # saving output matrices in dataset for days and storing files
    dims            = ['time','height']
    dimsT           = ['time']
    coordsT         = {"time":datetimeShipDay}
    coords          = {"time":datetimeShipDay,"height":rangeRadar}
    UDataArray      = xr.DataArray(dims=dims, coords=coords, data=uDay,
                        attrs={'long_name':'horizontal absolute wind speed',
                               'units':'m s-1'})
    VDataArray  = xr.DataArray(dims=dims, coords=coords, data=vDay,
                        attrs={'long_name':'wind direction wr to the north, from ICON-LEM',
                               'units':'degrees'})
    latitude        = xr.DataArray(dims=dimsT, coords=coordsT, data=latsDay,
                        attrs={'long_name':'latitude of the icon profiles selected',
                               'units':'degrees'})
    longitude       = xr.DataArray(dims=dimsT, coords=coordsT, data=lonsDay,
                        attrs={'long_name':'longitude of the icon profiles selected',
                               'units':'degrees'})
    
    # Put everything in a nice Dataset
    variables = {'u'    :UDataArray,
                 'v'    :VDataArray,
                 'lats' :latitude,
                 'lons' :longitude}
    global_attributes = {'created_by':'Claudia Acquistapace',
                         'created_on':str(datetime.now()),
                         'comment':'absolute wind speed and direction extracted from ICON LEM 1.25 km'}
    dataset = xr.Dataset(data_vars=variables,
                         coords=coords,
                         attrs=global_attributes)
    
    #add variable to xarray dataset and save to ncdf
    dataset.to_netcdf('/Volumes/Extreme SSD/ship_motion_correction_merian/wind_data_iconlem/'+yy+mm+dd+'windSpeed_direction_ICON.nc')
#%%
# plot wind profile to check if interpolation went well

from matplotlib import rcParams
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.dates as mdates

plt.rcParams.update({'figure.autolayout': True})
#plot of arthus backscatter signal and radar reflectivity
timeMin = timeStartGap # setts start time of the plot = day at 00:00:00 UTC
timeMax = timeStopGap   # setts end time of the plot = day+1 at 00:00:00 UTC
Ncols = 1
Nrows = 2
Nplots = 2
fontSizeTitle = 12
fontSizeX = 12
fontSizeY = 12
fontSizeCbar = 12
labelsizeaxes = 12
cbarAspect = 10
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Tahoma']
t = windDataRadar['time'].values
z = windDataRadar['height'].values
t_plot = t# time stamp convertion
#cmap = plt.cm.get_cmap('viridis', len(datetimeKite)) 

fig, ax = plt.subplots(nrows=Nrows, ncols=Ncols, figsize=(14, 10))
plt.gcf().subplots_adjust(bottom=0.15)
ymax = 2000.
ymin = 100.#107.
ax = plt.subplot(Nrows, Ncols, 1)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()
matplotlib.rc('xtick', labelsize=labelsizeaxes)  # sets dimension of ticks in the plots
matplotlib.rc('ytick', labelsize=labelsizeaxes)  # sets dimension of ticks in the plots
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
ax.xaxis.set_minor_formatter(mdates.DateFormatter("%H:%M"))
ax.xaxis_date()
cm = ax.pcolormesh(t_plot, z, windDataRadar['HwindSpeed'].values.T, vmin=0., vmax=15., cmap='inferno')
ax.set_ylim(ymin,ymax)                                               # limits of the y-axesn  cmap=plt.cm.get_cmap("viridis", 256)
ax.set_xlim(timeMin, timeMax)                                 # limits of the x-axes
ax.set_title('horizontal wind speed [m/s]', fontsize=fontSizeTitle, loc='left')
ax.set_xlabel("time [hh:mm]", fontsize=fontSizeX)
ax.set_ylabel("height [m]", fontsize=fontSizeY)
cbar = fig.colorbar(cm, orientation='vertical', aspect=cbarAspect)
cbar.set_label(label="H wind speed [m/s] ",size=fontSizeCbar)
cbar.ax.tick_params(labelsize=labelsizeaxes)

#fig.subplots_adjust(wspace=0.7)
#divider = make_axes_locatable(ax)
#ax_cbar = fig.add_axes([0.2, 1.05, 0.5, 0.05])

#Add an axes at position rect [left, bottom, width, height] 
#where all quantities are in fractions of figure width and height


# plot of wind or
ax = plt.subplot(Nrows, Ncols, 2)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.get_yaxis().tick_left()
matplotlib.rc('xtick', labelsize=labelsizeaxes)  # sets dimension of ticks in the plots
matplotlib.rc('ytick', labelsize=labelsizeaxes)  # sets dimension of ticks in the plots
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
ax.xaxis.set_minor_formatter(mdates.DateFormatter("%H:%M"))
ax.xaxis_date()
cax = ax.pcolormesh(IconData['time'].values, IconData['heightIcon'][0,:].values, IconData['HwindSpeed'].values.T, vmin=0., vmax=15., cmap='inferno')
ax.set_ylim(ymin,ymax)                                               # limits of the y-axesn  cmap=plt.cm.get_cmap("viridis", 256)
ax.set_xlim(timeMin, timeMax)                                 # limits of the x-axes
ax.set_title('wind Speed ICON [m/s]', fontsize=fontSizeTitle, loc='left')
ax.set_xlabel("time [hh:mm]", fontsize=fontSizeX)
ax.set_ylabel("height [m]", fontsize=fontSizeY)
cbar = fig.colorbar(cax, orientation='vertical', aspect=cbarAspect)
cbar.set_label(label="wind Speed ICON [m/s]",size=fontSizeCbar)
cbar.ax.tick_params(labelsize=labelsizeaxes)
