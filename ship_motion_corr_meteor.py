  #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
 date  : Fri Nov 20 14:09:09 2020
 author: Claudia Acquistapace
 goal: run ship motion correction code on Meteor data 

"""
# importing necessary libraries
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import netCDF4 as nc4
from netCDF4 import Dataset
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import rcParams
import matplotlib
import glob
import pandas as pd
from datetime import datetime
from datetime import timedelta
#import atmos
import xarray as xr
from scipy.interpolate import CubicSpline

# generating array of days for the entire campaign
Eurec4aDays  = pd.date_range(datetime(2020,1,19),datetime(2020,2,19),freq='d')
NdaysEurec4a = len(Eurec4aDays)

#######################################################################################
                        # PARAMETERS TO BE SET BY USERS *

# paths to the different data files and output directories for plots
pathFolderTree  = '/Volumes/Extreme SSD/ship_motion_correction_merian/'
pathFig         = pathFolderTree+'/plots/meteor/'
pathNcDataAnc   = pathFolderTree+'/ncdf_ancillary/'

### instrument position coordinates [+5.15m; +5.40m;−15.60m]
r_FMCW          = [-11. , 4.07, -15.8] # [m]

# select a date
dayEu           = Eurec4aDays[-4]

# extracting strings for yy, dd, mm
yy              = str(dayEu)[0:4]
mm              = str(dayEu)[5:7]
dd              = str(dayEu)[8:10]
date            = yy+mm+dd

# selecting hour to be processed
hour            = '16'

# selecting time resolution of ship data to be used ( in Hertz)
Hrz             = 10

# selecting height for comparison of power spectra of fft transform of w time series
selHeight        = 1600. # height in the chirp 1

#######################################################################################

# definitions of functions necessary for the processing 
def f_closest(array,value):
    '''
    # closest function
    #---------------------------------------------------------------------------------
    # date :  16.10.2017
    # author: Claudia Acquistapace
    # goal: return the index of the element of the input array that in closest to the value provided to the function
    '''
    import numpy as np
    idx = (np.abs(array-value)).argmin()
    return idx  
def f_readShipDataset(shipDataName):

  '''
  Created on Mer Sep 30 16:06:20 2020

  @author: cacquist
  @goal: read the ship data file and extract data to an xarray dataset


  inputs:
    shipDataName (ship path+filename string)
    
  Output: ShipData xarray dataset containing 
  time [sec since 1970-01-01]
  lat 
  lon
  heading
  heave
  pitch
  roll
  absWindDir
  absWindSpeed
  relWindDir
  relWindSpeed
  heading2
  '''
  import pandas as pd
  import xarray as xr

  dataset       = pd.read_csv(shipDataName, skiprows=[1,2], usecols=['seconds since 1970','SYS.STR.PosLat','SYS.STR.PosLon',\
                                                  'Seapath.INHDT.Heading','Seapath.PSXN.Heave','Seapath.PSXN.Pitch',\
                                                  'Seapath.PSXN.Roll','Weatherstation.PEUMA.Absolute_wind_direction',\
                                                  'Weatherstation.PEUMA.Absolute_wind_speed','Weatherstation.PEUMA.Relative_wind_direction',\
                                                  'Weatherstation.PEUMA.Relative_wind_speed', 'Seapath.PSXN.Heading'], low_memory=False)
  xrDataset     = dataset.to_xarray()   
      
  ShipData      = xrDataset.rename({'seconds since 1970':'time',\
                    'SYS.STR.PosLat':'lat',\
                    'SYS.STR.PosLon':'lon',\
                    'Seapath.INHDT.Heading':'heading_INHDT',\
                    'Seapath.PSXN.Heave':'heave',\
                    'Seapath.PSXN.Pitch':'pitch',\
                    'Seapath.PSXN.Roll':'roll',\
                    'Weatherstation.PEUMA.Absolute_wind_direction':'absWindDir',\
                    'Weatherstation.PEUMA.Absolute_wind_speed':'AbsWindSpeed',\
                    'Weatherstation.PEUMA.Relative_wind_direction':'relWindDir',\
                    'Weatherstation.PEUMA.Relative_wind_speed':'relWindSpeed',\
                    'Seapath.PSXN.Heading':'heading_PSXN'})
  return(ShipData)
def f_findMdvTimeSerie(values, datetime, rangeHeight, NtimeStampsRun, pathFig, chirp):
    '''
    author: Claudia Acquistapace
    date: 25 november 2020
    goal : identify, given a mean doppler velocity matrix, a sequence of lenght 
    NtimeStampsRun, of values in the matrix at a given height 
    that contains the minimum possible amount of nan values in it. 

    Parameters
    ----------
    INPUT:
    values : TYPE ndarray(time, height)
        DESCRIPTION : matrix of values for which it is necessary to find a serie of non nan values of given lenght
    datetime : datetime
        DESCRIPTION: time array associated with the matrix of values
    rangeHeight : ndarray
        DESCRIPTION: height array associated with the matrix of values
    NtimeStampsRun : scalar
        DESCRIPTION: length of the sequence of values that are read for scanning the matrix
        (expressed in interval of the time resolution)
    pathFig : string
        DESCRIPTION: string for the output path of the selected plot
    chirp: string
        DESCRIPTION: string indicating the chirp of the radar data processed
    OUTPUT:
    valuesTimeSerie: type(ndarray) - time serie of the lenght prescribed by NtimeStampsRun corresponding 
    to the minimum amount of nan values found in the serie 
    -------
    
    '''
    import pandas as pd
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib import rcParams
    import matplotlib
    
    # extracting date from timestamp format 
    date = '20'+pd.to_datetime(datetimeRadar[0]).strftime(format="%y%m%d-%H")
    
    #  concept: scan the matrix using running mean for every height, and check the number of nans in the selected serie. 
    nanAmountMatrix = np.zeros((len(datetime)-NtimeStampsRun, len(rangeHeight)))
    nanAmountMatrix.fill(np.nan)
    for indtime in range(len(datetime)-NtimeStampsRun):
        mdvChunk = values[indtime:indtime+NtimeStampsRun, :]
        df = pd.DataFrame(mdvChunk, index = datetime[indtime:indtime+NtimeStampsRun], columns=rangeHeight)
        
        # count number of nans in each height
        nanAmountMatrix[indtime,:] = df.isnull().sum(axis=0).values


    # find indeces where nanAmount is minimal
    ntuples      = np.where(nanAmountMatrix == np.nanmin(nanAmountMatrix))
    i_time_sel   = ntuples[0][0]
    i_height_sel = ntuples[1][0]
        
    # extract corresponding time Serie of mean Doppler velocity values for the chirp
    valuesTimeSerie = values[i_time_sel:i_time_sel+NtimeStampsRun, i_height_sel]
    timeSerie       = datetime[i_time_sel:i_time_sel+NtimeStampsRun]
    heightSerie     = np.repeat(rangeHeight[i_height_sel], NtimeStampsRun)


    ###### adding test for columns ########
    valuesColumn = values[i_time_sel:i_time_sel+NtimeStampsRun, :]
    valuesColumnMean = np.nanmean(valuesColumn, axis=1)
    
    
    # plotting quicklooks of the values map and the picked time serie interval
    labelsizeaxes   = 12
    fontSizeTitle = 12
    fontSizeX = 12
    fontSizeY = 12
    cbarAspect = 10
    fontSizeCbar = 12
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12,6))
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
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.xaxis.set_minor_formatter(mdates.DateFormatter("%H:%M"))
    ax.xaxis_date()
    cax = ax.pcolormesh(datetime[:-NtimeStampsRun], rangeHeight, nanAmountMatrix.transpose(), vmin=0., vmax=200., cmap='viridis')
    #ax.scatter(timeSerie, heightSerie, s=nanAmountSerie, c='orange', marker='o')
    ax.plot(timeSerie, heightSerie, color='orange', linewidth=7.0)
    ax.set_ylim(rangeHeight[0],rangeHeight[-1]+200.)                                               # limits of the y-axesn  cmap=plt.cm.get_cmap("viridis", 256)
    ax.set_xlim(datetime[0], datetime[-200])                                 # limits of the x-axes
    ax.set_title('time-height plot for the day : '+date, fontsize=fontSizeTitle, loc='left')
    ax.set_xlabel("time [hh:mm]", fontsize=fontSizeX)
    ax.set_ylabel("height [m]", fontsize=fontSizeY)
    cbar = fig.colorbar(cax, orientation='vertical', aspect=cbarAspect)
    cbar.set_label(label='Nan Amount [%]', size=fontSizeCbar)
    cbar.ax.tick_params(labelsize=labelsizeaxes)
    
    
    ax = plt.subplot(2,1,2)
    ax.spines["top"].set_visible(False)  
    ax.spines["right"].set_visible(False)  
    ax.get_xaxis().tick_bottom()  
    ax.get_yaxis().tick_left() 
    matplotlib.rc('xtick', labelsize=labelsizeaxes)  # sets dimension of ticks in the plots
    matplotlib.rc('ytick', labelsize=labelsizeaxes)  # sets dimension of ticks in the plots
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.xaxis.set_minor_formatter(mdates.DateFormatter("%H:%M"))
    ax.xaxis_date()
    ax.plot(timeSerie,valuesTimeSerie, color='black', label='selected values at '+str(heightSerie[0]))
    ax.legend(frameon=False)
    ax.set_xlim(timeSerie[0], timeSerie[-1])                                 # limits of the x-axes
    fig.tight_layout()
    fig.savefig(pathFig+date+'_'+chirp+'_quicklooks_mdvSelectedSerie.png', format='png')
 
    return(valuesTimeSerie,timeSerie, valuesColumnMean)
def plot_2Dmaps(time,height,y,ystring,ymin, ymax, hmin, hmax, timeStartDay, timeEndDay, colormapName, date, yVarName, pathFig): 
    """
    author: Claudia Acquistapace
    date: 25 november 2020
    goal : plot 2d map of the input variable
    input:
        time: time coordinate in datetime format
        height: height coordinate
        y: numpyarray 2d of the corresponding variable to be mapped
        ystring: string with name and units of the variable to be plotted
        ymin: min value to be plotted for the variable
        ymax: max value to be plotted for the variable
        timeStartDay: datetime start for the xaxis
        timeEndDay: datetime end for the xaxis
        colormapName: string indicating the chosen color map
        yVarName: name of the variable to be used for filename output
        pathFig: path where to store the plot
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib import rcParams
    import matplotlib
    
    
    labelsizeaxes   = 12
    fontSizeTitle = 12
    fontSizeX = 12
    fontSizeY = 12
    cbarAspect = 10
    fontSizeCbar = 12
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
    cax = ax.pcolormesh(time, height, y.transpose(), vmin=ymin, vmax=ymax, cmap=colormapName)
    ax.set_ylim(hmin,hmax)                                               # limits of the y-axesn  cmap=plt.cm.get_cmap("viridis", 256)
    ax.set_xlim(timeStartDay, timeEndDay)                                 # limits of the x-axes
    ax.set_title('time-height plot for the day : '+date, fontsize=fontSizeTitle, loc='left')
    ax.set_xlabel("time [hh:mm]", fontsize=fontSizeX)
    ax.set_ylabel("height [m]", fontsize=fontSizeY)
    cbar = fig.colorbar(cax, orientation='vertical', aspect=cbarAspect)
    cbar.set_label(label=ystring, size=fontSizeCbar)
    cbar.ax.tick_params(labelsize=labelsizeaxes)
    fig.tight_layout()
    fig.savefig(pathFig+date+'_'+yVarName+'_2dmaps.png', format='png')
def f_shiftTimeDataset(dataset):
    '''
    author: Claudia Acquistapace
    date: 25 november 2020
    goal : function to shift time variable of the dataset to the central value of the time interval
    of the time step
    input: 
        dataset: xarray dataset
    output:
        dataset: xarray dataset with the time coordinate shifted added to the coordinates and the variables now referring to the shifted time array
    '''
    # reading time array
    time   = dataset['time'].values
    # calculating deltaT using consecutive time stamps
    deltaT = time[2]-time[1]
    # print('delta T for the selected dataset: ', deltaT)
    # defining additional coordinate to the dataset
    dataset.coords['time_shifted'] = dataset['time']+0.5*deltaT
    # exchanging coordinates in the dataset
    datasetNew = dataset.swap_dims({'time':'time_shifted'})
    return(datasetNew)
def f_calcRMatrix(rollShipArr,pitchShipArr,yawShipArr,NtimeShip):
    '''
    author: Claudia Acquistapace
    date : 27/10/2020
    goal: function to calculate R matrix given roll, pitch, yaw
    input:
        rollShipArr: roll array in degrees
        pitchShipArr: pitch array in degrees
        yawShipArr: yaw array in degrees
        NtimeShip: dimension of time array for the definition of R_inv as [3,3,dimTime]
    output: 
        R[3,3,Dimtime]: array of rotational matrices, one for each time stamp
    '''            
    # calculation of the rotational matrix for each time stamp of the ship data for the day
    cosTheta = np.cos(np.deg2rad(rollShipArr))
    senTheta = np.sin(np.deg2rad(rollShipArr))
    cosPhi   = np.cos(np.deg2rad(pitchShipArr))
    senPhi   = np.sin(np.deg2rad(pitchShipArr))
    cosPsi   = np.cos(np.deg2rad(yawShipArr))
    senPsi   = np.sin(np.deg2rad(yawShipArr))
    
    R = np.zeros([3, 3, NtimeShip])
    A = np.zeros([3, 3, NtimeShip])
    B = np.zeros([3, 3, NtimeShip])
    C = np.zeros([3, 3, NtimeShip])
    R.fill(np.nan)
    A.fill(0.)
    B.fill(0.)
    C.fill(0.)
    
    # indexing for the matrices
    #[0,0]  [0,1]  [0,2]
    #[1,0]  [1,1]  [1,2]
    #[2,0]  [2,1]  [2,2]
    A[0,0,:] = 1
    A[1,1,:] = cosTheta
    A[1,2,:] = -senTheta
    A[2,1,:] = senTheta
    A[2,2,:] = cosTheta
    
    B[0,0,:] = cosPhi
    B[1,1,:] = 1
    B[2,2,:] = cosPhi
    B[0,2,:] = senPhi
    B[2,0,:] = -senPhi
    
    C[0,0,:] = cosPsi
    C[0,1,:] = -senPsi
    C[2,2,:] = 1
    C[1,0,:] = senPsi
    C[1,1,:] = cosPsi
        
    # calculation of the rotation matrix
    A = np.moveaxis(A, 2, 0)
    B = np.moveaxis(B, 2, 0)
    C = np.moveaxis(C, 2, 0)
    R = np.matmul(C, np.matmul(B, A))
    R = np.moveaxis(R, 0, 2)
    return(R)   
def plot_timeSeries(x,y, ystring,ymin, ymax, timeStartDay, timeEndDay, date, yVarName, pathFig): 
    """
    author: Claudia Acquistapace
    date: 25 november 2020
    goal : function to plot time series of any variable
    input:
        x: time coordinate in datetime format
        y: numpyarray of the corresponding variable
        ystrring: string with name and units of the variable to be plotted
        ymin: min value to be plotted for the variable
        ymax: max value to be plotted for the variable
        timeStartDay: datetime start for the xaxis
        timeEndDay: datetime end for the xaxis
        date: string with date for the selected time serie
        yVarName: name of the variable to be used for filename output
        pathFig: path where to store the plot
    output: saves a plot in pathFig+date+'_'+yVarName+'_timeSerie.png'
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib import rcParams
    import matplotlib
    
    labelsizeaxes   = 12
    fontSizeTitle = 12
    fontSizeX = 12
    fontSizeY = 12
    cbarAspect = 10
    fontSizeCbar = 12

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
    #ax.scatter(x, y, color = "m", marker = "o", s=1)
    ax.plot(x, y)
    ax.xaxis_date()
    ax.set_ylim(ymin,ymax)                                               # limits of the y-axesn  cmap=plt.cm.get_cmap("viridis", 256)
    ax.set_xlim(timeStartDay, timeEndDay)                                 # limits of the x-axes
    ax.set_title('time serie for the day : '+date, fontsize=fontSizeTitle, loc='left')
    ax.set_xlabel("time [hh:mm]", fontsize=fontSizeX)
    ax.set_ylabel(ystring, fontsize=fontSizeY)
    fig.tight_layout()
    fig.savefig(pathFig+date+'_'+yVarName+'_timeSerie.png', format='png')
def read_seapath(date, path=pathFolderTree+'/ship_meteor_data/', **kwargs):
    """
    author: Johannes Roettenbacher
    goal: Read in Seapath measurements from ship from .dat files to a pandas.DataFrame
    Args:
        date (datetime.datetime): object with date of current file
        path (str): path to seapath files
        **kwargs for read_csv

    Returns:
        seapath (DataFrame): DataFrame with Seapath measurements

    """
    # Seapath attitude and heave data 1Hz
    #start = time.time()
    # unpack kwargs
    nrows = kwargs['nrows'] if 'nrows' in kwargs else None
    skiprows = kwargs['skiprows'] if 'skiprows' in kwargs else (1, 2)
    file = date+'_DSHIP_seapath_10Hz.dat'
    # set encoding and separator, skip the rows with the unit and type of measurement
    seapath = pd.read_csv(path+file, encoding='windows-1252', sep="\t", skiprows=skiprows,
                          index_col='date time', nrows=nrows)
    # transform index to datetime
    seapath.index = pd.to_datetime(seapath.index, infer_datetime_format=True)
    seapath.index.name = 'datetime'
    seapath.columns = ['Heading [°]', 'Heave [m]', 'Pitch [°]', 'Roll [°]']  # rename columns
    #logger.info(f"Done reading in Seapath data in {time.time() - start:.2f} seconds")
    return seapath
def f_calculateExactRadarTime(millisec, chirpIntegrations, datetimeRadar):
    '''
    date   : 23/11/2020
    author : Claudia Acquistapace
    contact: cacquist@uni-koeln.de
    goal   : function to calculate the exact radar time composing the millisecond part, 
             the chirp integration time and summing everything up to get for each time 
             step, the exact central time stamp of the time step. 
             Exact time of the radar is
             t_radar_i = t_radar + sampleTms/1000 - sum(j=i,N) chirpIntegrations[j] + chirpIntegrations[i]/2
             because the time recorded is the final time of the time interval

    input: 
        millisec: numpy array containing milliseconds to be added, expressed in seconds
        chirpIntegrations: numpy array of integration time of each chirp, expressed in nanoseconds
        datetimeRadar: datetime array of the radar time stamps
    output:
        datetimeChirp matrix type (datetime64[ns]) with dimensions (3,len(datetimeRadar))
        each row corresponds to the new calculated chirp datetime array as indicated below:
        timeChirp1 = datetime[0,:]
        timeChirp2 = datetime[1,:] 
        timeChirp3 = datetime[2,:] 
    '''
        
    # converting time in seconds since 1970
    timeRadar          = datetimeRadar[:].astype('datetime64[s]').astype('int')  # time in seconds since 1970
    
    # calculating chirp starting time
    timeChirp          = np.zeros((3,len(datetimeRadar)))
    datetimeChirp      = np.empty((3,len(datetimeRadar)), dtype='datetime64[ns]')
    
    # calculating time stamps of each chirp
    for i_chirp in range(len(chirpIntegrations)):
        timeChirp[i_chirp,:] = timeRadar +  millisec * 10.**(-3) -  np.sum(chirpIntegrations[i_chirp:]) + chirpIntegrations[i_chirp]/2
    
    for ind_chirp in range(len(chirpIntegrations)):
        for ind_time in range(len(timeRadar)):
            datetimeChirp[ind_chirp, ind_time] = pd.to_datetime(datetime.utcfromtimestamp(timeChirp[ind_chirp, ind_time]), unit='ns')
            
    return(datetimeChirp)
def f_calcTimeShift(w_radar_meanCol, DeltaTimeShift, w_ship_chirp, timeSerieRadar, pathFig, chirp, date, hour):
    '''             
    author: Claudia Acquistapace, Jan. H. Schween
    date:   25/11/2020
    goal:   calculate and estimation of the time lag between the radar time stamps and the ship time stamp
    
    NOTE: adding or subtracting the obtained time shift depends on what you did
    during the calculation of the covariances: if you added/subtracted time _shift 
    to t_radar you have to do the same for the 'exact time' 
    Here is the time shift anaylysis as plot:
    <ww> is short for <w'_ship*w'_radar> i.e. covariance between vertical speeds from 
    ship movements and radar its maximum gives an estimate for optimal agreement in 
    vertical velocities of ship and radar
    <Delta w^2> is short for <(w[i]-w[i-1])^2> where w = w_rad - 2*w_ship - this 
    is a measure for the stripeness. Its minimum gives an 
    estimate how to get the smoothest w data

    Parameters
    ----------
    INPUT: 
    w_radar_meanCol : ndarray
        DESCRIPTION: mdv time serie of NtimeStampsRun elements selected from the mdv matrix of the radar ( average over height)
    DeltaTimeShift : ndarray
        DESCRIPTION: array of time shifts to be used to calculate shifted series and derive covariances
    w_ship_chirp : TYPE: ndarray
        DESCRIPTION: time serie of vertical velocities of the ship in the selected time interval
    timeSerieRadar : datetime array
        DESCRIPTION.time array corresponding to the w_radar serie of values
    pathFig : TYPE string
        DESCRIPTION. output path for quicklooks
    chirp : TYPE string
        DESCRIPTION. string indicating the chirp processed
    date : TYPE string
        DESCRIPTION. date 
    hour : TYPE string
        DESCRIPTION. hour
    OUTPUT:
    timeShift_chirp: TYPE ndarray of dimension equal to the number of chirps
        DESCRIPTION: time lag for each chirp in seconds   
    -------

    '''
    from scipy.interpolate import CubicSpline

    labelsizeaxes   = 12
    fontSizeTitle = 12
    fontSizeX = 12
    fontSizeY = 12
    rcParams['font.sans-serif'] = ['Tahoma']
    matplotlib.rcParams['savefig.dpi'] = 100
    plt.gcf().subplots_adjust(bottom=0.15)   
    
    # calculating variation for w_radar
    w_prime_radar = w_radar_meanCol - np.nanmean(w_radar_meanCol)
    
    print(np.shape(w_prime_radar))
    # calculating covariance between w-ship and w_radar where w_ship is ahifted for each deltaT given gy DeltaTimeShift
    cov_ww      = np.zeros(len(DeltaTimeShift))
    deltaW_ship = np.zeros(len(DeltaTimeShift))
    
    for i in range(len(DeltaTimeShift)):
        
        # calculate w_ship interpolating it on the new time array (timeShip+deltatimeShift(i))
        T_corr = pd.to_datetime(timeSerieRadar) + timedelta(seconds=DeltaTimeShift[i])

        # interpolating w_ship on the shifted time serie
        cs_ship = CubicSpline(timeSerieRadar, w_ship_chirp)
        w_ship_shifted = cs_ship(pd.to_datetime(T_corr))
     
        #calculating w_prime_ship with the new interpolated serie
        w_ship_prime = w_ship_shifted - np.nanmean(w_ship_shifted)
        
        # calculating covariance of the prime series
        cov_ww[i] = np.nanmean(w_ship_prime*w_prime_radar)
        
        # calculating sharpness deltaW_ship
        w_corrected = w_radar_meanCol - w_ship_shifted
        delta_w = (np.ediff1d(w_corrected))**2
        deltaW_ship[i] = np.nanmean(delta_w)
    

    #calculating max of covariance and min of deltaW_ship
    minDeltaW = np.nanmin(deltaW_ship)
    indMin    = np.where(deltaW_ship == minDeltaW)
    maxCov_w  = np.nanmax(cov_ww)
    indMax    = np.where(cov_ww == maxCov_w)
    print('time shift found for the chirp '+str(DeltaTimeShift[indMin][0]))
        
    # calculating time shift for radar data
    timeShift_chirp = DeltaTimeShift[indMin][0]
    
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12,6))
    fig.tight_layout()
    ax = plt.subplot(1,1,1)
    ax.spines["top"].set_visible(False)  
    ax.spines["right"].set_visible(False)  
    ax.get_xaxis().tick_bottom()  
    ax.get_yaxis().tick_left() 
    matplotlib.rc('xtick', labelsize=labelsizeaxes)  # sets dimension of ticks in the plots
    matplotlib.rc('ytick', labelsize=labelsizeaxes)  # sets dimension of ticks in the plots
    ax.plot(DeltaTimeShift, cov_ww, color='red', linestyle=':', label='cov_ww')
    ax.axvline(x=DeltaTimeShift[indMax], color='red', linestyle=':', label='max cov_w')
    ax.plot(DeltaTimeShift, deltaW_ship, color='red', label='Deltaw^2')
    ax.axvline(x=DeltaTimeShift[indMin], color='red', label='min Deltaw^2')
    ax.legend(frameon=False)
    #ax.xaxis_date()
    ax.set_ylim(-0.1,2.)                                               # limits of the y-axesn  cmap=plt.cm.get_cmap("viridis", 256)
    ax.set_xlim(DeltaTmin,DeltaTmax)                                 # limits of the x-axes
    ax.set_title('covariance and sharpiness for chirp '+chirp+': '+date+' '+hour+':'+str(int(hour)+1)+', time lag found : '+str(DeltaTimeShift[indMin]), fontsize=fontSizeTitle, loc='left')
    ax.set_xlabel("time Shift [seconds]", fontsize=fontSizeX)
    ax.set_ylabel('w [m s-1]', fontsize=fontSizeY)
    fig.tight_layout()
    fig.savefig(pathFig+date+'_'+hour+'_'+chirp+'_timeShiftQuicklook.png', format='png')
    
    return(timeShift_chirp)       
def f_calcFftSpectra(vel, time):
        '''
        author: Claudia Acquistapace
        goal  :function to calculate fft spectra of a velocity time series
        date  : 07.12.2020
        Parameters
        ----------
        vel : TYPE ndarray [m/s]
            DESCRIPTION: time serie of velocity
        time : TYPE datetime
            DESCRIPTION. time array corresponding to the velocity

        Returns
        -------
        w_pow power spectra obtained with fft transform of the velocity time serie
        freq  corresponding frequencies

        '''
        import numpy as np
        w_fft = np.fft.fft(vel)
        N     = len(w_fft)
        T_len = (time[-1] - time[0]).total_seconds()
        w_pow = (abs(w_fft))**2
        w_pow = w_pow[1:int(N/2)+1]
        freq  = np.arange(int(N/2))  * 1/T_len
        return(w_pow, freq)
def nan_helper(y):
    """ 
    author : Claudia Acquistapace
    date   : 21/12/2020
    source : this code was found on the web at the following link 
            https://stackoverflow.com/questions/6518811/interpolate-nan-values-in-a-numpy-array
    goal   : Helper to handle indices and logical indices of NaNs.

    Input  :
        - y, 1d numpy array with possible NaNs
    Output :  
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """
    return np.isnan(y), lambda z: z.nonzero()[0]
def tick_function(X):
    V = 1/(X)
    return ["%.f" % z for z in V]


######################################################################################

#%%
# processing of the selected hour
print('processing date: '+date)
print('* reading ship data')
if Hrz == 1:     
    dataset      = pd.read_csv(pathFolderTree+'/ship_meteor_data/'+date+'_DSHIP_all_1Hz.dat',\
                                        sep="\t",  encoding='windows-1252', header=0, skiprows=[1,2])#, \
    timeShipArr  = dataset['SYS.CALC.Timestamp'].values
    unitsShipArr = 'seconds since 1970-01-01 00:00:00'
    datetimeShip = nc4.num2date(timeShipArr, unitsShipArr, only_use_cftime_datetimes=False)
    heaveShip    = dataset['SEAPATH.PSXN.Heave'].values
    rollShip     = dataset['SEAPATH.PSXN.Roll'].values
    pitchShip    = dataset['SEAPATH.PSXN.Pitch'].values
    yawShip      = dataset['SEAPATH.PSXN.Heading'].values
else:
    dataset      = read_seapath(date)
    ShipDataset  = dataset.to_xarray()
    datetimeShip = pd.to_datetime(ShipDataset['datetime'].values)
    heaveShip    = ShipDataset['Heave [m]'].values
    rollShip     = ShipDataset['Roll [°]'].values
    pitchShip    = ShipDataset['Pitch [°]'].values
    yawShip      = ShipDataset['Heading [°]'].values
    
dims               = ['time']
coords             = {'time':datetimeShip}
RollVar            = xr.DataArray(dims=dims, coords=coords, data=rollShip)
PitchVar           = xr.DataArray(dims=dims, coords=coords, data=pitchShip)
YawVar             = xr.DataArray(dims=dims, coords=coords, data=yawShip)
HeaveVar           = xr.DataArray(dims=dims, coords=coords, data=heaveShip)


 # Put everything in a nice Dataset
variables = {'roll' : RollVar,
             'pitch': PitchVar, 
             'yaw'  : YawVar, 
             'heave': HeaveVar}
global_attributes = {'created_by':'Claudia Acquistapace',
                     'created_on':str(datetime.now()),
                     'comment':'ship data from RV. METEOR'}
ShipDataset = xr.Dataset(data_vars=variables,
                         coords=coords,
                         attrs=global_attributes)    

# slice one hour of data 
shipDataHour = ShipDataset.sel(time=slice(datetime(int(yy), int(mm),int(dd), int(hour),0,0),\
                                       datetime(int(yy), int(mm),int(dd), int(hour)+1,0,0)))

# shifting time stamp for ship data hour
shipDataHourCenter = f_shiftTimeDataset(shipDataHour)
    
# calculation of w_heave
heaveHour    = shipDataHourCenter['heave'].values
timeShipHour = pd.to_datetime(shipDataHourCenter['time_shifted'].values)
w_heave      = np.zeros(len(timeShipHour))
for i in range(1,len(timeShipHour)):
    w_heave[i] = (heaveHour[i] - heaveHour[i-1])/ \
        (timeShipHour[i] - timeShipHour[i-1]).total_seconds()

# calculating rotational terms 
rollHour  = shipDataHourCenter['roll'].values
pitchHour = shipDataHourCenter['pitch'].values
yawHour   = shipDataHourCenter['yaw'].values
NtimeShip = len(timeShipHour)
r_ship    = np.zeros((3,NtimeShip))

#calculate the position of the  radar on the ship r_ship:
R          = f_calcRMatrix(rollHour,pitchHour,yawHour,NtimeShip)
for i in range(NtimeShip):
    r_ship[:,i] = np.dot(R[:,:,i],r_FMCW)
        
# calculating vertical component of the velocity of the radar on the ship (v_rot)
w_rot = np.zeros(len(timeShipHour))
for i in range(1,len(timeShipHour)):
    w_rot[i] = (r_ship[2,i] - r_ship[2,i-1] ) / \
       (timeShipHour[i] - timeShipHour[i-1]).total_seconds()
       
# calculating total ship velocity
w_ship = w_rot + w_heave

#Take from the ship data only the valid times where roll and pitch are not -999 (or nan) - i assume gaps are short and rare
# select valid values of ship time series
i_valid = np.where(~np.isnan(rollHour) * \
                   ~np.isnan(pitchHour) * \
                   ~np.isnan(heaveHour) *\
                   ~np.isnan(w_ship))
    
w_ship_valid   = w_ship[i_valid]
timeShip_valid = timeShipHour[i_valid]
w_heave_valid  = w_heave[i_valid]


# plot time series of w_ship and w_heave for the plot interval
timeStartDay   = datetime(int(yy), int(mm),int(dd), 16,30,0)
timeEndDay     = datetime(int(yy), int(mm),int(dd), 16,32,0)
labelsizeaxes  = 12
fontSizeTitle  = 12
fontSizeX      = 12
fontSizeY      = 12
cbarAspect     = 10
fontSizeCbar   = 12

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
ax.xaxis_date()
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
major_ticks = np.arange(timeStartDay, timeEndDay, 50000000, dtype='datetime64')
minor_ticks = np.arange(timeStartDay, timeEndDay, 10000000, dtype='datetime64')
ax.tick_params(which = 'both', direction = 'out')
ax.set_xticks(major_ticks)
ax.set_xticks(minor_ticks, minor=True)
ax.grid(which='both')
ax.grid(which='minor', alpha=0.2)
ax.grid(which='major', alpha=0.5)
#ax.scatter(x, y, color = "m", marker = "o", s=1)
ax.plot(timeShip_valid, w_ship_valid, color='red', label='w_ship')
ax.plot(timeShip_valid, w_heave_valid, color='black', label='w_heave')

ax.legend(frameon=False)
ax.xaxis_date()
ax.set_ylim(-1.5,1.5)                                               # limits of the y-axesn  cmap=plt.cm.get_cmap("viridis", 256)
ax.set_xlim(timeStartDay, timeEndDay)                                 # limits of the x-axes
ax.set_title('time serie for the day : '+date+' - no time shift', fontsize=fontSizeTitle, loc='left')
ax.set_xlabel("time [hh:mm:ss]", fontsize=fontSizeX)
ax.set_ylabel('w [m s-1]', fontsize=fontSizeY)
fig.tight_layout()
fig.savefig(pathFig+date+'_'+hour+'_wship_heave_timeSerie.png', format='png')

#%%
# reading radar data 
radarData          = xr.open_dataset(pathFolderTree+'/meteor_data/D16/'+yy[2:4]+mm+dd+'_'+hour+'0000_P07_ZEN.LV1.nc')
datetimeRadar      = nc4.num2date(radarData['Time'].values, 'seconds since 2001-01-01 00:00:00', only_use_cftime_datetimes=False)
C1Range            = radarData['C1Range'].values
C2Range            = radarData['C2Range'].values
C3Range            = radarData['C3Range'].values
rangeRadar         = np.zeros(len(C1Range) + len(C2Range)+len(C3Range))
rangeRadar[0:len(C1Range)] = C1Range
rangeRadar[len(C1Range):len(C1Range)+len(C2Range)] = C2Range
rangeRadar[len(C1Range)+len(C2Range):len(C1Range)+len(C2Range)+len(C3Range)] = C3Range
C1mdv              = radarData['C1MeanVel'].values
C2mdv              = radarData['C2MeanVel'].values
C3mdv              = radarData['C3MeanVel'].values
mdv                = np.zeros((len(datetimeRadar), len(rangeRadar)))
mdv[:, 0:len(C1Range)] = C1mdv
mdv[:, len(C1Range):len(C1Range)+len(C2Range)] = C2mdv
mdv[:, len(C1Range)+len(C2Range):len(C1Range)+len(C2Range)+len(C3Range)] = C3mdv
mdv[mdv == -999.] = np.nan
range_offset       = [C1Range[0], C1Range[-1], C2Range[-1]]
chirpIntegrations  = radarData['SeqIntTime'].values
millisec           = radarData['Timems'].values/1000.
NtimeRadar         = len(datetimeRadar)
Nchirps            = len(chirpIntegrations)


# plot on mean doppler velocity time height 
plot_2Dmaps(datetimeRadar, \
            rangeRadar, \
            mdv, \
            'Mean Doppler velocity', \
            -6., \
            4., \
            100., \
            2200., \
            timeStartDay, \
            timeEndDay, \
            'seismic', \
            date, \
            'meanDopplerVel', \
            pathFig)

#%%
# calculating exact radar time stamps
timeChirp1, timeChirp2, timeChirp3 = f_calculateExactRadarTime(millisec, chirpIntegrations, datetimeRadar)

# prepare interpolation of the ship data
Cs                = CubicSpline(timeShip_valid, w_ship_valid)

# interpolating W_ship for each chirp on the time exact array of the chirp
wShip_exactChirp1 = Cs(pd.to_datetime(timeChirp1))
wShip_exactChirp2 = Cs(pd.to_datetime(timeChirp2))
wShip_exactChirp3 = Cs(pd.to_datetime(timeChirp3))

#%%
# calculation of the deltaT_shift array to use for getting good matching between t_radar and T-ship
# time shift array to be tested 
DeltaTmin      = -3.
DeltaTmax      = 3.
res            = 0.05
DimDeltaT      = (DeltaTmax- DeltaTmin)/res
DeltaTimeShift = np.arange(DeltaTmin, DeltaTmax, step=res)

# calculating time shift and correction for mean doppler velocity proceeding per chirp
timeShiftArray  = np.zeros((3))
timeShiftArray.fill(-999.)
timeExactFinal  = np.zeros((Nchirps, len(datetimeRadar)))
WshipExactFinal = np.zeros((Nchirps, len(datetimeRadar)))

chirpStringArr = []

# assigning lenght of the mean doppler velocity time serie for calculating time shift 
# with 3 sec time resolution, 200 corresponds to 10 min
NtimeStampsRun   = 200 
correctionMatrix = np.zeros((len(datetimeRadar),len(rangeRadar)))

for i_chirp in range(0,Nchirps):
    
    print('processing chirp '+str(i_chirp))
    
    #assigning string identifying the chirp that is processed
    chirp = 'chirp_'+str(i_chirp)
    chirpStringArr.append(chirp)
    
    # reading corresponding ship data and radar exact time array 
    if i_chirp == 0:
        wShip_exactChirp = wShip_exactChirp1
        timeSerieRadar   = timeChirp1
    elif i_chirp == 1:
        wShip_exactChirp = wShip_exactChirp2
        timeSerieRadar   = timeChirp2
    else:
        wShip_exactChirp = wShip_exactChirp3        
        timeSerieRadar   = timeChirp3

    # selecting index of min and max height of the chirp
    i_h_min = f_closest(rangeRadar, range_offset[i_chirp])
    if i_chirp+1 != 3:
        i_h_max = f_closest(rangeRadar,range_offset[i_chirp+1])
    else:
        i_h_max = -1
    
    print('processing between '+str(rangeRadar[i_h_min])+' and '+str(rangeRadar[i_h_max]))
    
    # reading mdv values for the selected chirp
    mvd_chirp  = mdv[:,i_h_min:i_h_max]
    dimHchirp  = np.shape(mvd_chirp)[1]
    rangeChirp = rangeRadar[i_h_min:i_h_max]
    
    # # search for at least 10 min of consecutive w obs in the chirp
    w_radar, timeRadarSel, w_radar_meanCol = f_findMdvTimeSerie(mvd_chirp, \
                                                                timeSerieRadar, \
                                                                rangeChirp, \
                                                                NtimeStampsRun, \
                                                                pathFig, \
                                                                chirp)
    
    # selecting wship values of the chirp over the same time interval
    Cs_chirp        = CubicSpline(timeSerieRadar, wShip_exactChirp)
    w_ship_chirpSel = Cs_chirp(timeRadarSel)

    # calculating time shift for the chirp
    if np.sum(np.where(~np.isnan(w_radar_meanCol))) != 0:
        timeShiftArray[i_chirp] = f_calcTimeShift(w_radar_meanCol, \
                                                  DeltaTimeShift, \
                                                  w_ship_chirpSel, \
                                                  timeRadarSel, \
                                                  pathFig, \
                                                  chirp, \
                                                  date, \
                                                  hour)
    else:
        timeShiftArray[i_chirp] = np.nan
    
    # recalculating exact time including time shift due to lag
    if ~np.isnan(timeShiftArray[i_chirp]):
        timeExact = pd.to_datetime(timeSerieRadar) - timedelta(seconds=timeShiftArray[i_chirp])
    else:
        timeExact = pd.to_datetime(timeSerieRadar) 

    timeExactFinal[i_chirp,:] = pd.to_datetime(timeExact)
    
    # interpolating ship data on the exact time again, with the right time found after the calculating
    # and the addition of the time shift
    Cs_final                   = CubicSpline(timeExact, wShip_exactChirp)
    W_ship1_exact              = Cs_final(pd.to_datetime(timeSerieRadar))
    WshipExactFinal[i_chirp,:] = W_ship1_exact
        
    #############################################################################
    # plot of the quantities as quicklooks
    timePlot  = pd.to_datetime(timeExact)
    timeStart = pd.to_datetime(timeRadarSel[0])#datetime(2020,1,20,6,0,0,0)#pd.to_datetime(timeSerieRadar)[0]
    timeEnd   = pd.to_datetime(timeRadarSel[40])#= datetime(2020,1,20,6,2,0,0)#pd.to_datetime(timeSerieRadar)[0]+timedelta(seconds=120)
    
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
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    #if Hrz == 1:
    #    major_ticks = np.arange(timeStart, timeEnd, 24, dtype='datetime64')
    #    minor_ticks = np.arange(timeStart, timeEnd, 120, dtype='datetime64')
    #else: 
    #    major_ticks = np.arange(timeStart, timeEnd, 24, dtype='datetime64')
    #    minor_ticks = np.arange(timeStart, timeEnd, 120, dtype='datetime64')

    ax.tick_params(which = 'both', direction = 'out')
    ax.set_xticks(major_ticks)
    ax.set_xticks(minor_ticks, minor=True)
    ax.grid(which='both')
    ax.grid(which='minor', alpha=0.2)
    ax.grid(which='major', alpha=0.5)
    ax.plot(timeRadarSel, w_radar_meanCol, color='red', label='mean w_radar over column')
    #ax.plot(timeRadarSel, w_radar, linewidth = 0.2, color='red', label='w_radar at one height') 
    ax.plot(timeRadarSel, w_ship_chirpSel, color='blue', linewidth=0.2, label='w_ship original')
    ax.plot(timePlot, wShip_exactChirp, color='blue', label='w_ship shifted of deltaT found')
    ax.scatter(pd.to_datetime(timeSerieRadar), W_ship1_exact, color='green', label='w_ship shifted interpolated on radar exact time')
    ax.set_ylim(-4.,2.)   
    ax.legend(frameon=False)
                                            # limits of the y-axesn  cmap=plt.cm.get_cmap("viridis", 256)
    ax.set_xlim(timeStart,timeEnd)                                 # limits of the x-axes
    ax.set_title('velocity for time delay calculations : '+date+' '+hour+':'+str(int(hour)+1)+' shift = '+str(timeShiftArray[i_chirp]), fontsize=fontSizeTitle, loc='left')
    ax.set_xlabel("time [hh:mm:ss]", fontsize=fontSizeX)
    ax.set_ylabel('w [m s-1]', fontsize=fontSizeY)
    fig.tight_layout()
    fig.savefig(pathFig+date+'_'+hour+'_'+str(i_chirp)+'_timeSerie_wship_wradar.png', format='png')

    # building correction term matrix
    correctionMatrix[:,i_h_min:i_h_max] = - np.repeat(W_ship1_exact, len(rangeChirp)).reshape(len(timeExact), len(rangeChirp))
    
#calculating corrected mean doppler velocity
mdv_corr = mdv + correctionMatrix    
#%%

# plot of the 2d map of mean doppler velocity corrected for the selected hour
plot_2Dmaps(datetimeRadar,rangeRadar,mdv_corr,'mdv corrected', -5., 4., 0., 2250., timeStartDay, timeEndDay, 'seismic', date, 'mdv_corr', pathFig)


# applying rolling average to the data
df        = pd.DataFrame(mdv_corr, index=datetimeRadar, columns=rangeRadar)
mdv_roll3 = df.rolling(window=3,center=True, axis=0).apply(lambda x: np.nanmean(x)) 

# plot of the 2d map of mean doppler velocity corrected for the selected hour with 3 steps running mean applied
plot_2Dmaps(datetimeRadar,rangeRadar,mdv_roll3.values,'mdv corrected rolling', -5., 4., 0., 2250., timeStartDay, timeEndDay, 'seismic', date, 'mdv_corr_rolling3', pathFig)

#%%

# calculation of the power spectra of the correction terms and of the original and corrected mean Doppler velocity time series at a given height

# interpolating ship correction terms of rotation and heave on the chirp time array
Cs_rot                = CubicSpline(timeShip_valid, w_rot)
Cs_heave              = CubicSpline(timeShip_valid, w_heave)
w_rot2                = Cs_rot(pd.to_datetime(timeChirp2))
w_heave2              = Cs_heave(pd.to_datetime(timeChirp2))

# plotting ffts of a selected height in the cloud (height selected in user parameter section)
iHeight               = f_closest(rangeRadar, selHeight)
timeExact             = pd.to_datetime(timeChirp2) - timedelta(seconds=timeShiftArray[1])
CS_interpfft          = CubicSpline(timeExact, wShip_exactChirp2)
CS_interp_rot         = CubicSpline(timeExact, w_rot2)
CS_interp_heave       = CubicSpline(timeExact, w_heave2)

# deriving values of wship rot and heave at the chirp times using the interpolation on the derived exact time
W_ship_interp         = CS_interpfft(timeChirp2)
w_rot_interp          = CS_interp_rot(timeChirp2)
w_heave_interp        = CS_interp_heave(timeChirp2)

# calculating corrected mdv with and without the time shift to the exact value
W_corr                = mdv[:,iHeight] - W_ship_interp
W_corr_no_shift       = mdv[:,iHeight] - wShip_exactChirp2

# interpolating over nans the two series ( in order to calculate fft tranformation)
nans, x               = nan_helper(W_corr)
W_corr[nans]          = np.interp(x(nans), x(~nans), W_corr[~nans])

nans, x               = nan_helper(W_corr_no_shift)
W_corr_no_shift[nans] = np.interp(x(nans), x(~nans), W_corr_no_shift[~nans])

w_radar_orig          = mdv[:,iHeight]
nans, x               = nan_helper(w_radar_orig)
w_radar_orig[nans]    = np.interp(x(nans), x(~nans), w_radar_orig[~nans])


# calculating power spectra of the selected corrected time series
pow_radarCorr, freq_radarCorr       = f_calcFftSpectra(W_corr,pd.to_datetime(timeChirp2))  
pow_radarCorr_NS, freq_radarCorr_NS = f_calcFftSpectra(W_corr_no_shift, pd.to_datetime(timeChirp2))  
pow_wShip, freq_Ship                = f_calcFftSpectra(W_ship_interp, pd.to_datetime(timeChirp2))  
pow_wrot, freq_rot                  = f_calcFftSpectra(w_rot_interp, pd.to_datetime(timeChirp2))  
pow_wheave, freq_heave              = f_calcFftSpectra(w_heave_interp, pd.to_datetime(timeChirp2))  
pow_radarOrig, freq_radarOrig       = f_calcFftSpectra(w_radar_orig, pd.to_datetime(timeChirp2))  
#%%%

# plot of the power spectra calculated
fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(12,10))
rcParams['font.sans-serif'] = ['Tahoma']
matplotlib.rcParams['savefig.dpi'] = 100
plt.gcf().subplots_adjust(bottom=0.15)
fig.tight_layout()
ax = plt.subplot(3,1,1)
ax.spines["top"].set_visible(False)  
ax.spines["right"].set_visible(False)  
ax.get_xaxis().tick_bottom()  
ax.get_yaxis().tick_left() 
#ax.set_yscale('log')
matplotlib.rc('xtick', labelsize=labelsizeaxes)  # sets dimension of ticks in the plots
matplotlib.rc('ytick', labelsize=labelsizeaxes)  # sets dimension of ticks in the plots
ax.loglog(freq_Ship, pow_wShip, label='ship', color='black', alpha=0.5)
ax.loglog(freq_rot, pow_wrot, label='w_rot', color='purple')
ax.loglog(freq_heave, pow_wheave, label='w_heave', color='orange')
ax.legend(frameon=False)
ax2 = ax.twiny()
new_tick_locations = np.array([0.2, \
                               0.1, \
                               0.06666667, \
                               0.05, \
                               0.04, \
                               0.02, \
                               0.01666667    ])   
ax2.set_xlabel('periods [s]')
ax2.set_xscale('log')
ax2.set_xlim(ax.get_xlim())
ax2.set_xticks(new_tick_locations)
ax2.set_xticklabels(tick_function(new_tick_locations))


axt = plt.subplot(3,1,2)
axt.spines["top"].set_visible(False)  
axt.spines["right"].set_visible(False)  
axt.get_xaxis().tick_bottom()  
axt.get_yaxis().tick_left() 
#ax.set_yscale('log')
matplotlib.rc('xtick', labelsize=labelsizeaxes)  # sets dimension of ticks in the plots
matplotlib.rc('ytick', labelsize=labelsizeaxes)  # sets dimension of ticks in the plots
axt.loglog(freq_radarOrig, pow_radarOrig, label='radar', color='black')
axt.loglog(freq_radarCorr, pow_radarCorr, label='corr with time shift', color='pink')

axt.legend(frameon=False)
axt2 = ax.twiny()
new_tick_locations = np.array([0.2, \
                               0.1, \
                               0.06666667, \
                               0.05, \
                               0.04, \
                               0.02, \
                               0.01666667    ])   
axt2.set_xlabel('periods [s]')
axt2.set_xscale('log')
axt2.set_xlim(ax.get_xlim())
axt2.set_xticks(new_tick_locations)
axt2.set_xticklabels(tick_function(new_tick_locations))


axtt = plt.subplot(3,1,3)
axtt.spines["top"].set_visible(False)  
axtt.spines["right"].set_visible(False)  
axtt.get_xaxis().tick_bottom()  
axtt.get_yaxis().tick_left() 
#ax.set_yscale('log')
matplotlib.rc('xtick', labelsize=labelsizeaxes)  # sets dimension of ticks in the plots
matplotlib.rc('ytick', labelsize=labelsizeaxes)  # sets dimension of ticks in the plots
axtt.loglog(freq_radarOrig, pow_radarOrig, label='radar', color='black')
axtt.loglog(freq_radarCorr_NS, pow_radarCorr_NS, label='corr without time shift', color='green')

axtt.legend(frameon=False)
axtt2 = ax.twiny()
new_tick_locations = np.array([0.2, \
                               0.1, \
                               0.06666667, \
                               0.05, \
                               0.04, \
                               0.02, \
                               0.01666667    ])   
axtt2.set_xlabel('periods [s]')
axtt2.set_xscale('log')
axtt2.set_xlim(ax.get_xlim())
axtt2.set_xticks(new_tick_locations)
axtt2.set_xticklabels(tick_function(new_tick_locations))
#ax.set_xlim(0.001, 0.5)
#ax.set_ylim(10**(-9.), 10)
fig.tight_layout()
fig.savefig(pathFig+date+'_'+hour+'_'+str(i_chirp)+'_fft_check.png', format='png')