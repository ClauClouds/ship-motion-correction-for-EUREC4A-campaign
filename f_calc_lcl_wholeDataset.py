#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 11:21:40 2021


'seconds since 1970',\
'SYS.STR.PosLat',\
'SYS.STR.PosLon',\
'Weatherstation.PEUMA.Absolute_wind_direction',\
'Weatherstation.PEUMA.Absolute_wind_speed',\
'Weatherstation.PEUMA.Absolute_wind_speed_bf',\
'Weatherstation.PEUMA.Air_pressure2',\
'Weatherstation.PEUMA.Air_pressure',\
'Weatherstation.PEUMA.Air_temperature',\
'Weatherstation.PEUMA.Humidity',\
'Weatherstation.PEUMA.Relative_wind_direction',\
'Weatherstation.PEUMA.Relative_wind_speed',\
'Weatherstation.PEUMA.Relative_wind_speed_bf',\
'Weatherstation.PEUMA.Water_temperature',\
'Global_radiation.SMSMN.GS',\
'Global_radiation.SMSMN.IR',\
'Global_radiation.SMSMN.PA',\
'Global_radiation.SMSMN.TI',\
'Rainmeter.RAINX.Sensor1',\
'Rainmeter.RAINX.Sensor2',\

@author: cacquist
"""
import os.path
import pandas as pd
import netCDF4 as nc4
from netCDF4 import Dataset
import xarray as xr
import numpy as np
from functions_essd import f_interpShipData
from functions_essd import lcl
from functions_essd import f_readShipDataset
from datetime import datetime
from datetime import timedelta
# reading ship data for the entire campaign
# paths to the different data files and output directories for plots
pathFolderTree  = '/Volumes/Extreme SSD/ship_motion_correction_merian/'
pathFig         = pathFolderTree+'/plots_essd/'
pathNcDataAnc   = pathFolderTree+'/ncdf_ancillary/'
ShipData        = pathFolderTree+'/ship_data/new/ship_data_all.dat'   
 
print('* reading ship data')
if os.path.isfile(pathFolderTree+'/ship_data/new/LCL_dataset.nc') * os.path.isfile(pathFolderTree+'/ship_data/new/wind_dataset.nc'):
    print('ship data already processed, opening ncdf')
    LCL_dataset = xr.open_dataset(pathFolderTree+'/ship_data/new/LCL_dataset.nc')
    wind_dataset = xr.open_dataset(pathFolderTree+'/ship_data/new/wind_dataset.nc')
else:
    print('file LCL_dataset.nc or wind_dataset.nc not found: producing them')
    dataset       = pd.read_csv(ShipData, skiprows=[1,2], usecols=['seconds since 1970',\
                                                       'SYS.STR.PosLat',\
                                                        'SYS.STR.PosLon',\
                                                        'Weatherstation.PEUMA.Absolute_wind_direction',\
                                                        'Weatherstation.PEUMA.Absolute_wind_speed',\
                                                        'Weatherstation.PEUMA.Air_pressure2',\
                                                        'Weatherstation.PEUMA.Air_pressure',\
                                                        'Weatherstation.PEUMA.Air_temperature',\
                                                        'Weatherstation.PEUMA.Humidity',\
                                                        'Weatherstation.PEUMA.Relative_wind_direction',\
                                                        'Weatherstation.PEUMA.Relative_wind_speed',\
                                                        'Weatherstation.PEUMA.Water_temperature',\
                                                        'Rainmeter.RAINX.Sensor1',\
                                                        'Rainmeter.RAINX.Sensor2'], low_memory=False)
    # converting pandas to xarray
    xrDataset     = dataset.to_xarray()   
    # renaming xarray variables
    ShipData      = xrDataset.rename({'seconds since 1970':'time',\
          'SYS.STR.PosLat':'lat',\
          'SYS.STR.PosLon':'lon',\
          'Weatherstation.PEUMA.Relative_wind_direction':'relWindDir',\
          'Weatherstation.PEUMA.Relative_wind_speed':'relWindSpeed',\
          'Weatherstation.PEUMA.Air_pressure':'pressure1', \
          'Weatherstation.PEUMA.Air_pressure2':'pressure2', \
          'Weatherstation.PEUMA.Air_temperature': 'T',\
          'Weatherstation.PEUMA.Humidity':'RH', \
          'Weatherstation.PEUMA.Water_temperature':'SST',\
          'Rainmeter.RAINX.Sensor1':'rain1',\
          'Rainmeter.RAINX.Sensor2':'rain2', \
          'Weatherstation.PEUMA.Absolute_wind_direction':'wind_direction', \
          'Weatherstation.PEUMA.Absolute_wind_speed':'wind_speed'})
                                

    print('* converting time array to datetime')
    # adding time in datetime format to ncdf file
    timeShipArr               = ShipData['time'].values
    unitsShipArr              = 'seconds since 1970-01-01 00:00:00'
    datetimeShip              = nc4.num2date(timeShipArr, unitsShipArr, only_use_cftime_datetimes=False)
    ShipData = ShipData.assign_coords({'datetime':datetimeShip})

    print('* removing nan values and calculating  lcl ')
    P = ShipData['pressure1'].values
    T = ShipData['T'].values
    RH = ShipData['RH'].values
    SST = ShipData['SST'].values
    lat = ShipData['lat'].values
    lon = ShipData['lon'].values
    Hwind = ShipData['wind_speed'].values
    windDir =  ShipData['wind_direction'].values
    
    
    # selecting only valid values and converting in the right units to derive LCL
    i_valid = np.where((P != -999.) *(T != -999.) *(RH != -999.) * (lat != -999.) * (lon != -999.))
    datetimeLCL = datetimeShip[i_valid]
    lat = lat[i_valid]
    lon = lon[i_valid] 
    P = P[i_valid] * 100. #pa
    T = T[i_valid]+ 273.15
    RH = RH[i_valid]*0.01
    SST = SST[i_valid]
 
    # vectorization of the function to calculate lcl
    vecDist = np.vectorize(lcl)
    # calculation of lcl for the whole dataset
    lclDataset = vecDist(P[:], T[:], RH[:])
    print('lcl calculated')

    
    # additional filtering for wind data
    i_valid_wind = np.where((Hwind != -999.) * (windDir != -999.))
    Hwind = Hwind[i_valid_wind]
    windDir = windDir[i_valid_wind]   
    timeWind = datetimeShip[i_valid_wind] 

    ## save all data of wind in a nice xarray dataset
    dims             = ['time']
    coords           = {"time":datetimeLCL}
   
    dimsWiind             = ['time']
    coordsWind           = {"time":timeWind}
    HwindArray         = xr.DataArray(dims=dimsWiind, coords=coordsWind, data=Hwind,
     attrs={'long_name':'Horizontal wind speed ',
    'units':'ms-1'})
    HwindDirArray         = xr.DataArray(dims=dimsWiind, coords=coordsWind, data=windDir,
     attrs={'long_name':'Horizontal wind direction ',
    'units':'degrees'}) 
    variables = {'Hwind_speed':HwindArray, 
                 'Hwind_dir':HwindDirArray}
    global_attributes = {'created_by':'Claudia Acquistapace',
                         'created_on':str(datetime.now()),
                         'comment':'basic thermodynamic variables from RV Merian'}
    wind_dataset      = xr.Dataset(data_vars = variables,
                                  coords = coords,
                                  attrs = global_attributes)
    wind_dataset.to_netcdf(pathFolderTree+'/ship_data/new/wind_dataset.nc')    
    
    # save all other data in ncdf
    lclArray         = xr.DataArray(dims=dims, coords=coords, data=lclDataset,
     attrs={'long_name':'LCL height calculated using algorithm Version 1.0 released by David Romps on September 12, 2017',
    'units':'m'})
    pressArray       = xr.DataArray(dims=dims, coords=coords, data=P,
     attrs={'long_name':'Air pressure',
    'units':'Pa'})
    tempArray        = xr.DataArray(dims=dims, coords=coords, data=T,
     attrs={'long_name':'Air temperature',
    'units':'K'})
    RelHumArray        = xr.DataArray(dims=dims, coords=coords, data=RH,
     attrs={'long_name':'Relative humidity',
    'units':'RH'})
    SSTArray        = xr.DataArray(dims=dims, coords=coords, data=SST,
     attrs={'long_name':'Sea surface temperature',
    'units':'K'})
    latArray        = xr.DataArray(dims=dims, coords=coords, data=lat,
     attrs={'long_name':'latitude',
    'units':'degrees'})
    lonArray        = xr.DataArray(dims=dims, coords=coords, data=lon,
     attrs={'long_name':'longitude',
    'units':'degrees'})    
    variables         = {'lcl':lclArray,
                         'P':pressArray, 
                         'T':tempArray, 
                         'RH':RelHumArray,
                         'SST':SSTArray, 
                         'lat':latArray,
                         'lon':lonArray}
    global_attributes = {'created_by':'Claudia Acquistapace',
                         'created_on':str(datetime.now()),
                         'comment':'basic thermodynamic variables from RV Merian'}
    LCL_dataset      = xr.Dataset(data_vars = variables,
                                  coords = coords,
                                  attrs = global_attributes)
    LCL_dataset.to_netcdf(pathFolderTree+'/ship_data/new/LCL_dataset.nc')
    
    

#%%

# defining array pf seconds of the day
lcltime = pd.to_datetime(LCL_dataset.time[:].values).hour*3600+\
          pd.to_datetime(LCL_dataset.time[:].values).minute*60+\
              pd.to_datetime(LCL_dataset.time[:].values).second
              
# defining data array for seconds of the day         
dims             = ['time']
coords           = {"time":pd.to_datetime(LCL_dataset.time[:].values)}
secondsoftheday = xr.DataArray(dims=dims, coords=coords, data=lcltime)

#assigning secondsoftheday as new coordinate variable
LCL_dataset = LCL_dataset.assign_coords({'secondsoftheday':secondsoftheday}) 

#group by the new coordinate and calculate mean of each variable
lcl_mean_day = LCL_dataset.lcl.groupby('secondsoftheday').mean()
lcl_std_day = LCL_dataset.lcl.groupby('secondsoftheday').std()
P_mean_day  = LCL_dataset.P.groupby('secondsoftheday').mean()
P_std_day   = LCL_dataset.P.groupby('secondsoftheday').std()
T_mean_day  = LCL_dataset.T.groupby('secondsoftheday').mean()
T_std_day   = LCL_dataset.T.groupby('secondsoftheday').std()
RH_mean_day  = LCL_dataset.RH.groupby('secondsoftheday').mean()
RH_std_day   = LCL_dataset.RH.groupby('secondsoftheday').std()
SST_mean_day  = LCL_dataset.SST.groupby('secondsoftheday').mean()
SST_std_day   = LCL_dataset.SST.groupby('secondsoftheday').std()

# saving all variables in ncdf
lclArray         = xr.DataArray(dims=dims, coords=coords, data=lcl_mean_day,
 attrs={'long_name':'LCL mean over campaign days',
'units':'m'})
lclstdArray         = xr.DataArray(dims=dims, coords=coords, data=lcl_std_day,
 attrs={'long_name':'LCL std over campaign days',
'units':'m'})
pressArray       = xr.DataArray(dims=dims, coords=coords, data=P_mean_day,
 attrs={'long_name':'Air pressure mean over campaign days',
'units':'Pa'})
PstdArray         = xr.DataArray(dims=dims, coords=coords, data=P_std_day,
 attrs={'long_name':'P std over campaign days',
'units':'Pa'})
tempArray        = xr.DataArray(dims=dims, coords=coords, data=T_mean_day,
 attrs={'long_name':'Air temperature mean over campaign days',
'units':'K'})
tempstdArray         = xr.DataArray(dims=dims, coords=coords, data=T_std_day,
 attrs={'long_name':'T std over campaign days',
'units':'K'})
RelHumArray        = xr.DataArray(dims=dims, coords=coords, data=RH_mean_day,
 attrs={'long_name':'Relative humidity mean over campaign days',
'units':'%'})
RHstdArray         = xr.DataArray(dims=dims, coords=coords, data=RH_std_day,
 attrs={'long_name':'RH std over campaign days',
'units':'%'})
SSTArray        = xr.DataArray(dims=dims, coords=coords, data=SST_mean_day,
 attrs={'long_name':'Sea surface temperature mean over campaign days',
'units':'degrees'})  
SSTstdArray         = xr.DataArray(dims=dims, coords=coords, data=SST_std_day,
 attrs={'long_name':'SST std over campaign days',
'units':'degrees'})
variables         = {'lcl':lclArray,
                     'P':pressArray, 
                     'T':tempArray, 
                     'RH':RelHumArray,
                     'SST':SSTArray, 
                     'lcl_std':lclstdArray,
                     'P_std':PstdArray, 
                     'T_std':tempstdArray, 
                     'RH_std':RHstdArray,
                     'SST_std':SSTstdArray}
global_attributes = {'created_by':'Claudia Acquistapace',
                     'created_on':str(datetime.now()),
                     'comment':'basic thermodynamic variables from RV Merian'}
LCL_dataset      = xr.Dataset(data_vars = variables,
                              coords = coords,
                              attrs = global_attributes)
LCL_dataset.to_netcdf(pathFolderTree+'/ship_data/new/thermodyn_var_dailymean_dataset.nc')


#%%

# defining array pf seconds of the day
wind_time = pd.to_datetime(wind_dataset.time[:].values).hour*3600+\
          pd.to_datetime(wind_dataset.time[:].values).minute*60+\
              pd.to_datetime(wind_dataset.time[:].values).second
              
# defining data array for seconds of the day         
dims             = ['time']
coords           = {"time":pd.to_datetime(wind_dataset.time[:].values)}
secondsoftheday = xr.DataArray(dims=dims, coords=coords, data=wind_time)

#assigning secondsoftheday as new coordinate variable
wind_dataset = wind_dataset.assign_coords({'secondsoftheday':secondsoftheday}) 

hWindSpeed_mean_day = wind_dataset.Hwind_speed.groupby('secondsoftheday').mean()
hWindSpeed_std_day = wind_dataset.Hwind_speed.groupby('secondsoftheday').std()
hWindDir_mean_day = wind_dataset.Hwind_dir.groupby('secondsoftheday').mean()
hWindDir_std_day = wind_dataset.Hwind_dir.groupby('secondsoftheday').std()
#%%
datetimeDay = np.arange(datetime(2020,1,1,0,0,0), datetime(2020,1,2,0,0,0), timedelta(seconds=1)).astype(datetime)
# store variables in a xarray dataset    
dims             = ['time']
coords           = {"time":datetimeDay}

hWindSpeedArray         = xr.DataArray(dims=dims, coords=coords, data=hWindSpeed_mean_day,
 attrs={'long_name':'horizontal wind speed mean over campaign days',
'units':'ms-1'})
hWindSpeedstdArray         = xr.DataArray(dims=dims, coords=coords, data=hWindSpeed_std_day,
 attrs={'long_name':'horizontal wind speed std over campaign days',
'units':'ms-1'})
hWindDirArray         = xr.DataArray(dims=dims, coords=coords, data=hWindDir_mean_day,
 attrs={'long_name':'horizontal wind direction mean over campaign days',
'units':'ms-1'})
hWindDirstdArray         = xr.DataArray(dims=dims, coords=coords, data=hWindDir_std_day,
 attrs={'long_name':'horizontal wind direction std over campaign days',
'units':'ms-1'})
variables         = {'Hwind_speed':hWindSpeedArray,
                     'Hwind_speed_std':hWindSpeedstdArray, 
                     'Hwind_dir':hWindDirArray, 
                     'Hwind_dir_std':hWindDirstdArray}
global_attributes = {'created_by':'Claudia Acquistapace',
                     'created_on':str(datetime.now()),
                     'comment':'basic thermodynamic variables from RV Merian'}
Hwind_dataset      = xr.Dataset(data_vars = variables,
                              coords = coords,
                              attrs = global_attributes)
Hwind_dataset.to_netcdf(pathFolderTree+'/ship_data/new/Hwind_dailymean_dataset.nc')


#%%
pathFolderTree = '/Volumes/Extreme SSD/ship_motion_correction_merian/'
diurnalCycle_dataset = xr.open_dataset(pathFolderTree+'/ship_data/new/thermodyn_var_dailymean_dataset.nc')
# plot diurnal cicles of lcl
Hwind_dataset =  xr.open_dataset(pathFolderTree+'/ship_data/new/Hwind_dailymean_dataset.nc')


import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import rcParams
import matplotlib

    
# calculate distribution of LCL values as a function of the day 
# conversion of time from UTC to local time
timeLocal = pd.to_datetime(diurnalCycle_dataset['time'].values)-timedelta(hours=4)
timeStartDay = timeLocal[0]
timeEndDay = timeLocal[-1]
labelsizeaxes   = 12
fontSizeTitle = 12
fontSizeX = 12
fontSizeY = 12
cbarAspect = 10
fontSizeCbar = 12


fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10,6))
rcParams['font.sans-serif'] = ['Tahoma']
matplotlib.rcParams['savefig.dpi'] = 100
plt.gcf().subplots_adjust(bottom=0.15)
fig.tight_layout()
ax = plt.subplot(2,1,1)
ax.spines["top"].set_visible(False)  
ax.spines["right"].set_visible(False)  
ax.get_xaxis().tick_bottom()  
ax.get_yaxis().tick_left() 
matplotlib.rc('xtick', labelsize=labelsizeaxes)  # sets dimension of ticks in the plots
matplotlib.rc('ytick', labelsize=labelsizeaxes)  # sets dimension of ticks in the plots
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M")) # set the label format
ax.xaxis.set_minor_formatter(mdates.DateFormatter(" "))     # empty prints no label
major_ticks = np.arange(timeStartDay, timeEndDay, 7200000000, dtype='datetime64')
minor_ticks = np.arange(timeStartDay, timeEndDay, 900000000, dtype='datetime64')
ax.tick_params(which = 'major', direction = 'out')
ax.set_xticks(minor_ticks, minor=True)
ax.set_xticks(major_ticks)
ax.grid(which='minor', alpha=0.2)
ax.grid(which='major', alpha=0.5)
ax.xaxis_date()
ax.plot(timeLocal, Hwind_dataset['Hwind_speed'].values, color='red')
ax.fill_between(timeLocal, Hwind_dataset['Hwind_speed'].values-Hwind_dataset['Hwind_speed_std'].values, \
                Hwind_dataset['Hwind_speed'].values+Hwind_dataset['Hwind_speed_std'].values, \
                    alpha=0.2, color='red')
ax.axvline(x=pd.to_datetime(datetime(2020,1,1,6,30,0,0)), color='black',linewidth=2, linestyle=':')
ax.axvline(x=pd.to_datetime(datetime(2020,1,1,19,15,0,0)), color='black', linewidth=2, linestyle=':')
ax.set_ylim(5.,15.)                                               # limits of the y-axesn  cmap=plt.cm.get_cmap("viridis", 256)
ax.set_xlim(timeStartDay, timeEndDay)                                 # limits of the x-axes
ax.set_title('diurnal cycle of horizontal wind speed at the ocean surface ', fontsize=fontSizeTitle, loc='left')
ax.set_xlabel("local time (UTC -4) [hh:mm]", fontsize=fontSizeX)
ax.set_ylabel('wind speed [$ms-1$]', fontsize=fontSizeY)

ax1 = plt.subplot(2,1,2)
ax1.spines["top"].set_visible(False)  
ax1.spines["right"].set_visible(False)  
ax1.get_xaxis().tick_bottom()  
ax1.get_yaxis().tick_left() 
matplotlib.rc('xtick', labelsize=labelsizeaxes)  # sets dimension of ticks in the plots
matplotlib.rc('ytick', labelsize=labelsizeaxes)  # sets dimension of ticks in the plots
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M")) # set the label format
ax1.xaxis.set_minor_formatter(mdates.DateFormatter(" "))     # empty prints no label
ax1.tick_params(which = 'major', direction = 'out')
ax1.set_xticks(minor_ticks, minor=True)
ax1.set_xticks(major_ticks)
ax1.grid(which='minor', alpha=0.2)
ax1.grid(which='major', alpha=0.5)
ax1.xaxis_date()
ax1.plot(timeLocal, Hwind_dataset['Hwind_dir'].values, color='blue')
ax1.fill_between(timeLocal, Hwind_dataset['Hwind_dir'].values-Hwind_dataset['Hwind_dir_std'].values, \
                Hwind_dataset['Hwind_dir'].values+Hwind_dataset['Hwind_dir_std'].values, \
                    alpha=0.2, color='blue')
ax1.axvline(x=pd.to_datetime(datetime(2020,1,1,6,30,0,0)), color='black',linewidth=2, linestyle=':')
ax1.axvline(x=pd.to_datetime(datetime(2020,1,1,19,15,0,0)), color='black', linewidth=2, linestyle=':')
#(x=.5, ymin=0.25, ymax=0.75)
ax1.set_ylim(0.,200.)                                               # limits of the y-axesn  cmap=plt.cm.get_cmap("viridis", 256)
ax1.set_xlim(timeStartDay, timeEndDay)                                 # limits of the x-axes
ax1.set_title('diurnal cycle of horizontal wind direction at the surface', fontsize=fontSizeTitle, loc='left')
ax1.set_xlabel("local time (UTC -4) [hh:mm]", fontsize=fontSizeX)
ax1.set_ylabel('wind direction [degrees]', fontsize=fontSizeY)
fig.tight_layout()
fig.savefig(pathFig+'wind_diurnalCycle_timeSerie.png', format='png')


#%%
fig, ax = plt.subplots(nrows=5, ncols=1, figsize=(8,14))
rcParams['font.sans-serif'] = ['Tahoma']
matplotlib.rcParams['savefig.dpi'] = 100
plt.gcf().subplots_adjust(bottom=0.15)
fig.tight_layout()
ax = plt.subplot(5,1,1)
ax.spines["top"].set_visible(False)  
ax.spines["right"].set_visible(False)  
ax.get_xaxis().tick_bottom()  
ax.get_yaxis().tick_left() 
matplotlib.rc('xtick', labelsize=labelsizeaxes)  # sets dimension of ticks in the plots
matplotlib.rc('ytick', labelsize=labelsizeaxes)  # sets dimension of ticks in the plots
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M")) # set the label format
ax.xaxis.set_minor_formatter(mdates.DateFormatter(" "))     # empty prints no label
major_ticks = np.arange(timeStartDay, timeEndDay, 7200000000, dtype='datetime64')
minor_ticks = np.arange(timeStartDay, timeEndDay, 900000000, dtype='datetime64')
ax.tick_params(which = 'major', direction = 'out')
ax.set_xticks(minor_ticks, minor=True)
ax.set_xticks(major_ticks)
ax.grid(which='minor', alpha=0.2)
ax.grid(which='major', alpha=0.5)
ax.xaxis_date()
ax.plot(timeLocal, diurnalCycle_dataset['lcl'].values, color='red')
ax.fill_between(timeLocal, diurnalCycle_dataset['lcl'].values-diurnalCycle_dataset['lcl_std'].values, \
                diurnalCycle_dataset['lcl'].values+diurnalCycle_dataset['lcl_std'].values, \
                    alpha=0.2, color='red')
ax.axvline(x=pd.to_datetime(datetime(2020,1,1,6,30,0,0)), color='black',linewidth=2, linestyle=':')
ax.axvline(x=pd.to_datetime(datetime(2020,1,1,19,15,0,0)), color='black', linewidth=2, linestyle=':')
ax.set_ylim(400.,900.)                                               # limits of the y-axesn  cmap=plt.cm.get_cmap("viridis", 256)
ax.set_xlim(timeStartDay, timeEndDay)                                 # limits of the x-axes
ax.set_title('diurnal cycle of lifting condensation level (LCL)', fontsize=fontSizeTitle, loc='left')
ax.set_xlabel("local time (UTC -4) [hh:mm]", fontsize=fontSizeX)
ax.set_ylabel('height [m]', fontsize=fontSizeY)

ax1 = plt.subplot(5,1,2)
ax1.spines["top"].set_visible(False)  
ax1.spines["right"].set_visible(False)  
ax1.get_xaxis().tick_bottom()  
ax1.get_yaxis().tick_left() 
matplotlib.rc('xtick', labelsize=labelsizeaxes)  # sets dimension of ticks in the plots
matplotlib.rc('ytick', labelsize=labelsizeaxes)  # sets dimension of ticks in the plots
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M")) # set the label format
ax1.xaxis.set_minor_formatter(mdates.DateFormatter(" "))     # empty prints no label
ax1.tick_params(which = 'major', direction = 'out')
ax1.set_xticks(minor_ticks, minor=True)
ax1.set_xticks(major_ticks)
ax1.grid(which='minor', alpha=0.2)
ax1.grid(which='major', alpha=0.5)
ax1.xaxis_date()
ax1.plot(timeLocal, diurnalCycle_dataset['SST'].values, color='blue')
ax1.fill_between(timeLocal, diurnalCycle_dataset['SST'].values-diurnalCycle_dataset['SST_std'].values, \
                diurnalCycle_dataset['SST'].values+diurnalCycle_dataset['SST_std'].values, \
                    alpha=0.2, color='blue')
ax1.axvline(x=pd.to_datetime(datetime(2020,1,1,6,30,0,0)), color='black',linewidth=2, linestyle=':')
ax1.axvline(x=pd.to_datetime(datetime(2020,1,1,19,15,0,0)), color='black', linewidth=2, linestyle=':')
#(x=.5, ymin=0.25, ymax=0.75)
ax1.set_ylim(26.8,27.6)                                               # limits of the y-axesn  cmap=plt.cm.get_cmap("viridis", 256)
ax1.set_xlim(timeStartDay, timeEndDay)                                 # limits of the x-axes
ax1.set_title('diurnal cycle of sea surface temperature (SST)', fontsize=fontSizeTitle, loc='left')
ax1.set_xlabel("local time (UTC -4) [hh:mm]", fontsize=fontSizeX)
ax1.set_ylabel('SST [degrees C]', fontsize=fontSizeY)

ax2 = plt.subplot(5,1,3)
ax2.spines["top"].set_visible(False)  
ax2.spines["right"].set_visible(False)  
ax2.get_xaxis().tick_bottom()  
ax2.get_yaxis().tick_left() 
matplotlib.rc('xtick', labelsize=labelsizeaxes)  # sets dimension of ticks in the plots
matplotlib.rc('ytick', labelsize=labelsizeaxes)  # sets dimension of ticks in the plots
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M")) # set the label format
ax2.xaxis.set_minor_formatter(mdates.DateFormatter(" "))     # empty prints no label
ax2.tick_params(which = 'major', direction = 'out')
ax2.set_xticks(minor_ticks, minor=True)
ax2.set_xticks(major_ticks)
ax2.grid(which='minor', alpha=0.2)
ax2.grid(which='major', alpha=0.5)
ax2.xaxis_date()
ax2.plot(timeLocal, diurnalCycle_dataset['RH'].values*100., color='green')
ax2.fill_between(timeLocal, (diurnalCycle_dataset['RH'].values-diurnalCycle_dataset['RH_std'].values)*100, \
                (diurnalCycle_dataset['RH'].values+diurnalCycle_dataset['RH_std'].values)*100., \
                    alpha=0.2, color='green')
ax2.axvline(x=pd.to_datetime(datetime(2020,1,1,6,30,0,0)), color='black',linewidth=2, linestyle=':')
ax2.axvline(x=pd.to_datetime(datetime(2020,1,1,19,15,0,0)), color='black', linewidth=2, linestyle=':')
ax2.set_ylim(65.,80.)                                               # limits of the y-axesn  cmap=plt.cm.get_cmap("viridis", 256)
ax2.set_xlim(timeStartDay, timeEndDay)                                 # limits of the x-axes
ax2.set_title('diurnal cycle of relative humidity (RH) ', fontsize=fontSizeTitle, loc='left')
ax2.set_xlabel("local time (UTC -4) [hh:mm]", fontsize=fontSizeX)
ax2.set_ylabel('RH [%]', fontsize=fontSizeY)

ax3 = plt.subplot(5,1,4)
ax3.spines["top"].set_visible(False)  
ax3.spines["right"].set_visible(False)  
ax3.get_xaxis().tick_bottom()  
ax3.get_yaxis().tick_left() 
matplotlib.rc('xtick', labelsize=labelsizeaxes)  # sets dimension of ticks in the plots
matplotlib.rc('ytick', labelsize=labelsizeaxes)  # sets dimension of ticks in the plots
ax3.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M")) # set the label format
ax3.xaxis.set_minor_formatter(mdates.DateFormatter(" "))     # empty prints no label
ax3.tick_params(which = 'major', direction = 'out')
ax3.set_xticks(minor_ticks, minor=True)
ax3.set_xticks(major_ticks)
ax3.grid(which='minor', alpha=0.2)
ax3.grid(which='major', alpha=0.5)
ax3.xaxis_date()
ax3.plot(timeLocal, diurnalCycle_dataset['T'].values-273.15, color='orange')
ax3.fill_between(timeLocal, (diurnalCycle_dataset['T'].values-diurnalCycle_dataset['T_std'].values)-273.15, \
                (diurnalCycle_dataset['T'].values+diurnalCycle_dataset['T_std'].values)-273.15, \
                    alpha=0.2, color='orange')
ax3.axvline(x=pd.to_datetime(datetime(2020,1,1,6,30,0,0)), color='black',linewidth=2, linestyle=':')
ax3.axvline(x=pd.to_datetime(datetime(2020,1,1,19,15,0,0)), color='black', linewidth=2, linestyle=':')
ax3.set_ylim(25.5,28.)                                               # limits of the y-axesn  cmap=plt.cm.get_cmap("viridis", 256)
ax3.set_xlim(timeStartDay, timeEndDay)                                 # limits of the x-axes
ax3.set_title('diurnal cycle of air temperature (T) ', fontsize=fontSizeTitle, loc='left')
ax3.set_xlabel("local time (UTC -4) [hh:mm]", fontsize=fontSizeX)
ax3.set_ylabel('T [degrees C]', fontsize=fontSizeY)


ax4 = plt.subplot(5,1,5)
ax4.spines["top"].set_visible(False)  
ax4.spines["right"].set_visible(False)  
ax4.get_xaxis().tick_bottom()  
ax4.get_yaxis().tick_left() 
matplotlib.rc('xtick', labelsize=labelsizeaxes)  # sets dimension of ticks in the plots
matplotlib.rc('ytick', labelsize=labelsizeaxes)  # sets dimension of ticks in the plots
ax4.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M")) # set the label format
ax4.xaxis.set_minor_formatter(mdates.DateFormatter(" "))     # empty prints no label
ax4.tick_params(which = 'major', direction = 'out')
ax4.set_xticks(minor_ticks, minor=True)
ax4.set_xticks(major_ticks)
ax4.grid(which='minor', alpha=0.2)
ax4.grid(which='major', alpha=0.5)
ax4.xaxis_date()
ax4.plot(timeLocal, diurnalCycle_dataset['P'].values/100., color='purple')
ax4.fill_between(timeLocal, (diurnalCycle_dataset['P'].values-diurnalCycle_dataset['P_std'].values)/100., \
                (diurnalCycle_dataset['P'].values+diurnalCycle_dataset['P_std'].values)/100., \
                    alpha=0.2, color='purple')
ax4.axvline(x=pd.to_datetime(datetime(2020,1,1,6,30,0,0)), color='black',linewidth=2, linestyle=':')
ax4.axvline(x=pd.to_datetime(datetime(2020,1,1,19,15,0,0)), color='black', linewidth=2, linestyle=':')
ax4.set_ylim(1010.,1018.)                                               # limits of the y-axesn  cmap=plt.cm.get_cmap("viridis", 256)
ax4.set_xlim(timeStartDay, timeEndDay)                                 # limits of the x-axes
ax4.set_title('diurnal cycle of air pressure (P) ', fontsize=fontSizeTitle, loc='left')
ax4.set_xlabel("local time (UTC -4) [hh:mm]", fontsize=fontSizeX)
ax4.set_ylabel('P [hPa]', fontsize=fontSizeY)
fig.tight_layout()

fig.savefig(pathFig+'_diurnalCycle_timeSerie.png', format='png')

#%%



timeSlots = [0, 4, 8, 12, 16, 20]
ind_00_06 = np.where(hourlcl < 6)
ind_06_12 = np.where((hourlcl >= 6) * (hourlcl < 12))
ind_12_18 = np.where((hourlcl >= 12) * (hourlcl < 18))
ind_18_24 = np.where((hourlcl >= 18))

lcl_00_06 = LCL_dataset['lcl'].values[ind_00_06]
lcl_06_12 = LCL_dataset['lcl'].values[ind_06_12]
lcl_12_18 = LCL_dataset['lcl'].values[ind_12_18]
lcl_18_24 = LCL_dataset['lcl'].values[ind_18_24]

sst_00_06 = LCL_dataset['SST'].values[ind_00_06]
sst_06_12 = LCL_dataset['SST'].values[ind_06_12]
sst_12_18 = LCL_dataset['SST'].values[ind_12_18]
sst_18_24 = LCL_dataset['SST'].values[ind_18_24]


datetimeLCL_00_06 = LCL_dataset['time'].values[ind_00_06]
datetimeLCL_06_12 = LCL_dataset['time'].values[ind_06_12]
datetimeLCL_12_18 = LCL_dataset['time'].values[ind_12_18]
datetimeLCL_18_24 = LCL_dataset['time'].values[ind_18_24]



# scatter plot of SST vs LCL
colors = datetimeLCL_00_06
