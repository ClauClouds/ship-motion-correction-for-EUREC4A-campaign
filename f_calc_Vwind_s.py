#!/usr/bin/env python3
# -*- coding: utf-8 -*-


'''
author: Claudia Acquistapace
date  : 14/12/2020
goal : calculate profiles of wind zonal and meridional speeds from ICON LEM 1.25 Km model output
'''

import pandas as pd
import netCDF4 as nc4
import numpy as np
import xarray as xr
from datetime import datetime
import matplotlib.pyplot as plt

def f_shiftTimeDataset(dataset):
    '''
    function to shift time variable of the dataset to the central value of the time interval
    of the time step
    input: 
        dataset
    output:
        dataset with the time coordinate shifted added to the coordinates and the variables now referring to the shifted time array
    '''
    # reading time array
    time   = dataset['time'].values
    # calculating deltaT using consecutive time stamps
    deltaT = time[2]-time[1]
    print('delta T for the selected dataset: ', deltaT)
    # defining additional coordinate to the dataset
    dataset.coords['time_shifted'] = dataset['time']+0.5*deltaT
    # exchanging coordinates in the dataset
    datasetNew = dataset.swap_dims({'time':'time_shifted'})
    return(datasetNew)
def f_calcWindSpeedDir(u,v):
    """
    function to calculate wind speed and direction given the wind zonal and meridional components
    using the formula presented in the ECMWF documentation page
    https://confluence.ecmwf.int/pages/viewpage.action?pageId=133262398
    The convention for wind direction increasing clockwise with:
    - wind blowing from north: 0 degrees
    - wind blowing from east: 90 degrees
    - wind blowing from south: 180 degrees
    - wind blowing from west: 360 degrees

    Parameters
    ----------
    u : TYPE float
        DESCRIPTION. zonal wind component (positive to the right)
    v : TYPE float
        DESCRIPTION. meridional wind component (positive towards the north)
 
    Returns
    -------
    speed : float wind speed [ms-1]
    direction : float wind direction [degrees]
    

    """
    import numpy as np

    speed          = np.sqrt(u**2 + v**2)
    alpha          = 180 + np.arctan2(v,u) *180/np.pi       
    return(speed, alpha)

# paths to the different data files and output directories for plots
pathFolderTree  = '/Volumes/Extreme SSD/ship_motion_correction_merian/'
pathFig         = pathFolderTree+'/plots/27112020/'
pathNcDataAnc   = pathFolderTree+'/ncdf_ancillary/'
shipFile        = pathFolderTree+'/ship_data/new/shipData_all2.nc'


# open ship data 
shipData        = xr.open_dataset(shipFile)
#shipDataHour    = shipData.sel(time=slice(datetime(int(yy), int(mm),int(dd), int(hour),0,0),\
#                                          datetime(int(yy), int(mm),int(dd), int(hour)+1,0,0)))

# shifting time stamp for ship data
shipDataC   = f_shiftTimeDataset(shipData)
    
# calculation of heave velocity (heave rate in [ms-1])
heave       = shipDataC['heave'].values
timeShip    = pd.to_datetime(shipDataC['time_shifted'].values)

# calculating rotational velocity in [ms-1] 
roll        = shipDataC['roll'].values
pitch       = shipDataC['pitch'].values
yaw         = shipDataC['yaw'].values

#%%
# generating array of days for the dataset
Eurec4aDays          = pd.date_range(datetime(2020,1,19),datetime(2020,2,19),freq='d')
listDaysNonProcessed = [datetime(2020,1,22), datetime(2020,1,25), datetime(2020,2,2), datetime(2020,2,3)]

for ind in range(len(listDaysNonProcessed)):
    Eurec4aDays = Eurec4aDays.drop(listDaysNonProcessed[ind])
NdaysEurec4a    = len(Eurec4aDays)

    #%%   
for indDayEu in range(NdaysEurec4a):
        
    # select a date
    dayEu           = Eurec4aDays[indDayEu]
    
    # extracting strings for yy, dd, mm
    yy              = str(dayEu)[0:4]
    mm              = str(dayEu)[5:7]
    dd              = str(dayEu)[8:10]
    print('processing date :'+yy+'-'+mm+'-'+dd)

    # read fileList containing matrices per day from ICON 1.25 km dataset
    windFile     = '/Volumes/Extreme SSD/ship_motion_correction_merian/wind_data_iconlem/'+yy+mm+dd+'windSpeed_direction_ICON.nc'


    # read wind data from the selected file
    windData     = xr.open_dataset(windFile)
    u            = windData['u'].values
    v            = windData['v'].values
    timeWind     = windData['time'].values
    
    absWindSpeed, absWindDir = f_calcWindSpeedDir(u,v)
    
    # selecting ship yaw values for the selected day and building dataArray for interpolation
    yawDay       = yaw[(timeShip >= timeWind[0]) * (timeShip < timeWind[-1])]
    timeDay      = timeShip[(timeShip >= timeWind[0]) * (timeShip < timeWind[-1])]
    dims         = ['time']
    coordsT      = {"time":timeDay}
    yawDataArray = xr.DataArray(dims=dims, coords=coordsT, data=yawDay,
                            attrs={'long_name':'yaw time serie for the selected day from ship dataset',
                                   'units':'degrees'})
    
    
    # interpolation of yaw on the time grid of the wind
    yaw_interpolated = yawDataArray.interp(time=timeWind)
    
    # reading dimensions from wind matrices
    dimHeight        = absWindSpeed.shape[1]
    dimTime          = absWindSpeed.shape[0]
    
    # creating a matrix by replicating yaw time series for every height, to perform vectorial calculations
    yawMatrix        = np.repeat(yaw_interpolated.values, dimHeight, axis=0).reshape(dimTime,dimHeight)
    
    # converting yaw and wind direction in radians
    yaw_rad          = np.deg2rad(yawMatrix)
    windDir_rad      = np.deg2rad(absWindDir)
    
     
    # calculation of wind speed components in ship ref system
    Vs_x             = - np.cos(windDir_rad-yaw_rad) * absWindSpeed
    Vs_y             = np.sin(windDir_rad-yaw_rad) * absWindSpeed

    
    # saving variables of wind_s in a xarray dataset
    dims2             = ['time','height']
    coords2           = {"time":timeWind, "height":windData['height'].values}
    
    v_wind_s_x        = xr.DataArray(dims=dims2, coords=coords2, data=Vs_x,
                             attrs={'long_name':'zonal wind speed profile in ship reference system from ICON-LEM 1.25 km',
                                    'units':'m s-1'})
    v_wind_s_y        = xr.DataArray(dims=dims2, coords=coords2, data=Vs_y,
                             attrs={'long_name':'meridional wind speed in ship reference system from ICON-LEM 1.25 km',
                                    'units':'m s-1'})
    variables         = {'vs_x':v_wind_s_x,
                         'vs_y':v_wind_s_y}
    global_attributes = {'created_by':'Claudia Acquistapace',
                         'created_on':str(datetime.now()),
                         'comment':'wind direction and speed in the ship reference system for RV Merian'}
    dataset           = xr.Dataset(data_vars = variables,
                                       coords = coords2,
                                       attrs = global_attributes)


    # saving global dataset for the campaign in ncdf
    dataset.to_netcdf('/Volumes/Extreme SSD/ship_motion_correction_merian/ncdf_ancillary/'+yy+'-'+mm+'-'+dd+'_wind_s_dataset.nc')
        
