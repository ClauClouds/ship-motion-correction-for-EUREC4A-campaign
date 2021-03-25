#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  26 2021 
@author: cacquist
@goal: produce plot for the paper showing the application of the ship motion corrections
"""

# importing necessary libraries
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


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 10:42:54 2020
 code where we attempt a new matching strategy for the time steps
@author: cacquist
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
from functions_essd import f_closest
from functions_essd import f_calculateMomentsCol
from functions_essd import f_readAndMergeRadarDataDay_DopplerCorrection
from functions_essd import f_readAndMergeRadarDataDay
from functions_essd import generate_preprocess
from functions_essd import generate_preprocess_zen
from scipy.interpolate import CubicSpline
from pathlib import Path
import os.path
from matplotlib import rcParams

################################## functions definitions ####################################
def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    return np.isnan(y), lambda z: z.nonzero()[0]
def f_fftAnalysisForCheck(w_radar_orig, timeRadar, wShip, timeShip, W_corr, W_corr_no_shift, PathFigHour, date, hour):
    '''
    author: Claudia Acquistapace
    date  : 17/12/2020
    goal  : given the velocity time series for the various variables, calculates 
    and plots the power spectra obtained via fft transformation
    input parameters:
    ----------
    wRadar : TYPE ndarray, units ms-1
        DESCRIPTION.radar mean Doppler velocity time serie before the correction
    wShip : TYPE ndarray, units ms-1
        DESCRIPTION. ship velocity (heave plus rotation)
    wCorr : TYPE ndarray units ms-1
        DESCRIPTION. corrected mean Doppler velocity obtained applying the time shift
    wCorrNoShift : TYPE ndarray ms-1
        DESCRIPTION. corrected mean Doppler velocity obtained without applying the time shift
    PathFigHour : type string
        DESCRIPTION: string for output directory where to find the plot
    date: type string
        DESCRIPTION: string for the date processed
    hour: type string
        DESCRIPTION: string for the hour processed
    Returns
    -------
    list
        DESCRIPTION.

    '''
    # interpolate wship on radar time grid
    CsShip    = CubicSpline(pd.to_datetime(timeShip),wShip)
    W_ship    = CsShip(pd.to_datetime(timeRadar))
    

    
    # interpolating over nans the two series
    nans, x               = nan_helper(W_ship[:])
    W_ship[nans]          = np.interp(x(nans), x(~nans), W_ship[~nans])
    
    nans, x               = nan_helper(W_corr)
    print(nans, x)
    W_corr[nans]          = np.interp(x(nans), x(~nans), W_corr[~nans])

    nans, x               = nan_helper(W_corr_no_shift)
    W_corr_no_shift[nans] = np.interp(x(nans), x(~nans), W_corr_no_shift[~nans])
    
    nans, x               = nan_helper(w_radar_orig)
    w_radar_orig[nans]    = np.interp(x(nans), x(~nans), w_radar_orig[~nans])
    
    
    
    # calculating power spectra of the selected corrected time series
    pow_radarCorr, freq_radarCorr       = f_calcFftSpectra(W_corr,pd.to_datetime(timeRadar))  
    pow_radarCorr_NS, freq_radarCorr_NS = f_calcFftSpectra(W_corr_no_shift, pd.to_datetime(timeRadar))  
    pow_wShip, freq_Ship                = f_calcFftSpectra(W_ship, pd.to_datetime(timeRadar))  
    pow_radarOrig, freq_radarOrig       = f_calcFftSpectra(w_radar_orig, pd.to_datetime(timeRadar))                  

    fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(12,10))
    labelsizeaxes   = 12
    fontSizeTitle = 12
    fontSizeX = 12
    fontSizeY = 12
    cbarAspect = 10
    fontSizeCbar = 12
    # positions of the ticks for the period that appears as second axis
    new_tick_locations = np.array([0.2, \
                           0.1, \
                           0.06666667, \
                           0.05, \
                           0.04, \
                           0.02, \
                           0.01666667    ])   
    def tick_function(X):
        ''' function returning periods V when provided with frequencies X'''
        V = 1/(X)
        return ["%.f" % z for z in V]
    rcParams['font.sans-serif']        = ['Tahoma']
    matplotlib.rcParams['savefig.dpi'] = 100
    matplotlib.rc('xtick', labelsize=labelsizeaxes)  # sets dimension of ticks in the plots
    matplotlib.rc('ytick', labelsize=labelsizeaxes)  # sets dimension of ticks in the plots
    plt.gcf().subplots_adjust(bottom=0.15)
    fig.tight_layout()
    ax = plt.subplot(3,1,1)
    ax.spines["top"].set_visible(False)  
    ax.spines["right"].set_visible(False)  
    ax.get_xaxis().tick_bottom()  
    ax.get_yaxis().tick_left() 
    ax.loglog(freq_Ship, pow_wShip, label='ship', color='black', alpha=0.5)
    ax.loglog(freq_radarOrig, pow_radarOrig, label='radar', color='orange')
    ax.legend(frameon=False)
    ax2 = ax.twiny()
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
    axt.loglog(freq_radarOrig, pow_radarOrig, label='w_radar as measured', color='black')
    axt.loglog(freq_radarCorr, pow_radarCorr, label='correction with time shift', color='pink')
    
    axt.legend(frameon=False)
    axt2 = ax.twiny() 
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
    axtt.loglog(freq_radarOrig, pow_radarOrig, label='w_radar as measured', color='black')
    axtt.loglog(freq_radarCorr_NS, pow_radarCorr_NS, label='correction without time shift', color='green')
    
    axtt.legend(frameon=False)
    axtt2 = ax.twiny()
    axtt2.set_xlabel('periods [s]')
    axtt2.set_xscale('log')
    axtt2.set_xlim(ax.get_xlim())
    axtt2.set_xticks(new_tick_locations)
    axtt2.set_xticklabels(tick_function(new_tick_locations))
    #ax.set_xlim(0.001, 0.5)
    #ax.set_ylim(10**(-9.), 10)
    fig.tight_layout()
    fig.savefig(PathFigHour+date+'_'+hour+'_fft_analysis.png', format='png')
    return()
def f_interpolateRadarChirpTime(timeSerieValues, timeArr, radarChirpTime):
        '''
        author: Claudia Acquistapace
        date  : 14/12/2020
        goal  : interpolate a given time serie over a radar chirp time array provided as 
        input. Nans are removed from the time serie and then cubic interpolation is build on the 
        remaining values if they are non zero. 
        then, the cubic interpolation is calculated on the radar chirp time

        Parameters
        ----------
        timeArr : TYPE datetime array
            DESCRIPTION. the time array associated with the time serie provided as input
        timeSerieValues : TYPE ndarray
            DESCRIPTION. time serie of values to be interpolated
        radarChirpTime : TYPE datetime
            DESCRIPTION. time array of the radar chirp on which to interpolate

        Returns
        -------
        ValuesChirp: TYPE ndarray
            DESCRIPTION. Time serie of the 

        '''
        from scipy.interpolate import CubicSpline

        # find out non nan elements
        i_valid           = np.where(~np.isnan(timeSerieValues))
        if np.sum(i_valid) != 0:
            valTimeSerieValid = timeSerieValues[i_valid]
        else:
            'all values of time series are nan! no interpolation possible'
        # check that the amount of non nan elements is >= 50% of the values
        Nelements         = len(timeSerieValues)
        Nvalid            = len(valTimeSerieValid)
        if Nvalid > Nelements/2:   
            TimeArrValid      = timeArr[i_valid]
            Cs                = CubicSpline(TimeArrValid, valTimeSerieValid)
            ValuesChirp       = Cs(pd.to_datetime(radarChirpTime))
        else:
            ValuesChirp       = np.repeat(np.nan, len(radarChirpTime))
            
       
        return(ValuesChirp)
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
        w_fft fft tranform of the velocity time serie
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
def f_findMdvTimeSerie_VMerge(values, datetime, rangeHeight, NtimeStampsRun, pathFig, chirp):
    '''
    author: Claudia Acquistapace
    date: 25 november 2020
    modified: 16 December 2020 to work for all conditions (table work and not working)
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
    valuesTimeSerie  : type(ndarray) - time serie of the lenght prescribed by NtimeStampsRun corresponding 
    to the minimum amount of nan values found in the matrix of values
    i_height_sel     : type scalar - index of the height for the selected serie of values 
    timeSerie        : type datetime - time array corresponding to the selected time serie of values
    valuesColumnMean : type ndarray - values of the variable averaged over height for the selected time interval 
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
    if len(datetime)-NtimeStampsRun > 0:
        nanAmountMatrix = np.zeros((len(datetime)-NtimeStampsRun, len(rangeHeight)))
        nanAmountMatrix.fill(np.nan)
        for indtime in range(len(datetime)-NtimeStampsRun):
            mdvChunk = values[indtime:indtime+NtimeStampsRun, :]
            df = pd.DataFrame(mdvChunk, index = datetime[indtime:indtime+NtimeStampsRun], columns=rangeHeight)
            
            # count number of nans in each height
            nanAmountMatrix[indtime,:] = df.isnull().sum(axis=0).values


        # find indeces where nanAmount is minimal
        ntuples          = np.where(nanAmountMatrix == np.nanmin(nanAmountMatrix))
        i_time_sel       = ntuples[0][0]
        i_height_sel     = ntuples[1][0]
            
        # extract corresponding time Serie of mean Doppler velocity values for the chirp
        valuesTimeSerie  = values[i_time_sel:i_time_sel+NtimeStampsRun, i_height_sel]
        timeSerie        = datetime[i_time_sel:i_time_sel+NtimeStampsRun]
        heightSerie      = np.repeat(rangeHeight[i_height_sel], NtimeStampsRun)
    
    
        ###### adding test for columns ########
        valuesColumn     = values[i_time_sel:i_time_sel+NtimeStampsRun, :]
        valuesColumnMean = np.nanmean(valuesColumn, axis=1)
    else:
        nanAmountMatrix = np.zeros((len(datetime), len(rangeHeight)))
        nanAmountMatrix[~np.isnan(values)] = 1
        # count number of nans in each height
        NnansCol = []
        for indh in range(len(rangeHeight)):
            NnansCol.append(np.sum(~np.isnan(values[:,indh])))
        print(NnansCol)
        i_height_sel = np.argmax(NnansCol)
        print('height found ', i_height_sel)
        i_time_sel = np.arange(0,len(datetime)-1)
        valuesTimeSerie = values[:, i_height_sel]
        timeSerie = datetime
        heightSerie = np.repeat(rangeHeight[i_height_sel], len(datetime))
        valuesColumnMean = np.nanmean(values, axis=1)
        
    # plotting quicklooks of the values map and the picked time serie interval
    labelsizeaxes    = 12
    fontSizeTitle    = 12
    fontSizeX        = 12
    fontSizeY        = 12
    cbarAspect       = 10
    fontSizeCbar     = 12
    fig, ax          = plt.subplots(nrows=1, ncols=1, figsize=(12,6))
    rcParams['font.sans-serif'] = ['Tahoma']
    matplotlib.rcParams['savefig.dpi'] = 100
    plt.gcf().subplots_adjust(bottom=0.15)
    fig.tight_layout()
    ax               = plt.subplot(2,1,1)
    ax.spines["top"].set_visible(False)  
    ax.spines["right"].set_visible(False)  
    ax.get_xaxis().tick_bottom()  
    ax.get_yaxis().tick_left() 
    matplotlib.rc('xtick', labelsize=labelsizeaxes)  # sets dimension of ticks in the plots
    matplotlib.rc('ytick', labelsize=labelsizeaxes)  # sets dimension of ticks in the plots
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.xaxis.set_minor_formatter(mdates.DateFormatter("%H:%M"))
    ax.xaxis_date()
    if len(datetime)-NtimeStampsRun > 0:
        cax              = ax.pcolormesh(datetime[:-NtimeStampsRun], rangeHeight, nanAmountMatrix.transpose(), vmin=0., vmax=200., cmap='viridis')
        ax.set_xlim(datetime[0], datetime[-200])                                 # limits of the x-axes

    else:
        cax              = ax.pcolormesh(datetime[:], rangeHeight, nanAmountMatrix.transpose(), vmin=0., vmax=200., cmap='viridis')
        ax.set_xlim(datetime[0], datetime[-1])                                 # limits of the x-axes

    #ax.scatter(timeSerie, heightSerie, s=nanAmountSerie, c='orange', marker='o')
    ax.plot(timeSerie, heightSerie, color='orange', linewidth=7.0)
    ax.set_ylim(rangeHeight[0],rangeHeight[-1]+200.)                                               # limits of the y-axesn  cmap=plt.cm.get_cmap("viridis", 256)
    ax.set_title('time-height plot for the day : '+date, fontsize=fontSizeTitle, loc='left')
    ax.set_xlabel("time [hh:mm]", fontsize=fontSizeX)
    ax.set_ylabel("height [m]", fontsize=fontSizeY)
    cbar             = fig.colorbar(cax, orientation='vertical', aspect=cbarAspect)
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
    fig.savefig(pathFig+date+'_chirp_'+chirp+'_mdvSelected4Timeshift.png', format='png')
 
    return(valuesTimeSerie, i_height_sel, timeSerie, valuesColumnMean)
def plot_2Dmaps(time,height,y,ystring,ymin, ymax, hmin, hmax, timeStartDay, timeEndDay, colormapName, date, yVarName, pathFig): 
    """
    function to plot time series of any variable
    input:
        time: time coordinate in datetime format
        height: height coordinate
        y: numpyarra 2d of the corresponding variable to be mapped
        ystring: string with name and units of the variable to be plotted
        ymin: min value to be plotted for the variable
        ymax: max value to be plotted for the variable
        timeStartDay: datetime start for the xaxis
        timeEndDay: datetime end for the xaxis
        colormapName: 
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
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    ax.xaxis.set_minor_formatter(mdates.DateFormatter("%H:%M:$S"))
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
def f_calcRMatrix(rollShipArr,pitchShipArr,yawShipArr,NtimeShip):
            '''
            author: Claudia Acquistapace
            date : 27/10/2020
            goal: function to calculate R matrix given roll, pitch, yaw
            input:
                roll array in degrees
                pitch array in degrees
                yaw array in degrees
                dimtime: dimension of time array for the definition of R_inv as [3,3,dimTime]
            output: 
                R[3,3,Dimtime]
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
    function to plot time series of any variable
    input:
        x: time coordinate in datetime format
        y: numpyarray of the corresponding variable
        ystrring: string with name and units of the variable to be plotted
        ymin: min value to be plotted for the variable
        ymax: max value to be plotted for the variable
        timeStartDay: datetime start for the xaxis
        timeEndDay: datetime end for the xaxis
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
def f_calcTimeShift(w_radar_meanCol, DeltaTimeShift, w_ship_chirp, timeSerieRadar, pathFig, chirp, date, hour):
    '''             w_radar_meanCol, DeltaTimeShift, w_ship_chirpSel, timeRadar_valid, pathFig, chirp, date, hour
    author: Claudia Acquistapace, Jan. H. Schween
    date; 25/11/2020
    goal: calculate and estimation of the time lag between the radar time stamps and the ship time stamp
    
    NOTE: adding or subtracting the obtained time shift depends on what you did
    during the calculation of the covarainces: if you added/subtracted time _shift 
    to t_radar you have to do the same for the 'exact time' 
    if you added/subtracted it to t_ship you have to subtract/add it to t_radar.
    Here is the time shift anaylysis as plot:
    <ww> is short for <w'_ship*w'_radar> i.e. covariance between vertical speeds from 
    ship movements and radar its maximum gives an estimate for optimal agreement in 
    vertical velocities of ship and radar
    <Delta w^2> is short for <(w[i]-w[i-1])^2> where w = w_rad - 2*w_ship - this 
    is wat i called yesterday a measure for the stripeness. Its minimum gives an 
    estimate how to get the smoothest w data

    Parameters
    ----------
    INPUT: 
    w_radar : ndarray
        DESCRIPTION: mdv time serie of NtimeStampsRun elements selected from the mdv matrix of the radar 
    DeltaTimeShift : array of time shifts to be used to calculate shifted series and derive covariances
        DESCRIPTION.
    Cs : TYPe: interpolation
        DESCRIPTION: cubic interpolation of ship velocity  
    w_ship_valid, 
    timeShip_valid,
    timeSerieRadar : datetime array
        DESCRIPTION.time array corresponding to the w_radar serie of values
    pathFig : TYPE string
        DESCRIPTION. output path for quicklooks
    i_chirp : TYPE string
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
    ax.set_title('covariance and sharpiness for chirp '+str(i_chirp)+': '+date+' '+hour+':'+str(int(hour)+1)+', time lag found : '+str(DeltaTimeShift[indMin]), fontsize=fontSizeTitle, loc='left')
    ax.set_xlabel("time Shift [seconds]", fontsize=fontSizeX)
    ax.set_ylabel('w [m s-1]', fontsize=fontSizeY)
    fig.tight_layout()
    fig.savefig(pathFig+date+'_'+hour+'_'+str(i_chirp)+'_timeShiftQuicklook.png', format='png')
    
    return(timeShift_chirp)       
def f_calculateExactRadarTime(millisec, chirpIntegrations, datetimeRadar):
    '''
    date:23/11/2020
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
def f_closestTime(timeArray,value):
    '''
    # closest function
    #---------------------------------------------------------------------------------
    # date :  16.10.2017
    # author: Claudia Acquistapace
    # goal: return the index of the element of the input time array that 
    in closest to the value provided to the function
        input: 
            timeArray: numpy ndarray or datetime time array
            value: datetime tiem value of the form datetime(yy,mm,dd,hh,min,sec) 
        output: index of timearray closest to the value 
    '''
    import pandas as pd
    import numpy as np
    # converting time format to datetime if input time array is ndarray
    if type(timeArray).__module__ == np.__name__:
        timeArray = pd.to_datetime(timeArray)
        
    idx = (np.abs(timeArray-value)).argmin()
    return idx  
def f_findExtremesInterval(timeSerie):
    """
    goal : function to derive start and ending time of the nan gaps.
    input: timeSerie : time array in datetime format of times where the variable has a nan value
    output: 
        timeStart array : time containing starting time of consecutive time gaps
        timeEnd array   : time containing ending time of consecutive time gaps

    """
    # Construct dummy dataframe
    df = pd.DataFrame(timeSerie, columns=['time'])
    deltas = df['time'].diff()[0:]
    #print(deltas)
    gaps = deltas[deltas > timedelta(seconds=1)]
    #print(gaps)
    # build arrays to store data
    timeStopArr       = np.zeros((len(gaps)), dtype='datetime64[s]')
    timeRestartArr    = np.zeros((len(gaps)), dtype='datetime64[s]')
    durationSingleGaps = [" " for i in range(len(gaps))] 
    #TotalDuration      = gaps.sum()

    # Print results
    #print(f'{len(gaps)} gaps with total gap duration: {gaps.sum()}')
    indArr = 0
    for i, g in gaps.iteritems():
        time_stop = df['time'][i - 1]
        time_restart = df['time'][i + 1]
        timeStopArr[indArr] = datetime.strftime(time_stop, "%Y-%m-%d %H:%M:%S")
        timeRestartArr[indArr] = datetime.strftime(time_restart, "%Y-%m-%d %H:%M:%S")
        durationSingleGaps[indArr] = str(g.to_pytimedelta())
        indArr = indArr + 1    
        
        
        #print(f'time stop: {datetime.strftime(time_stop, "%Y-%m-%d %H:%M:%S")} | '
        #      f'Duration gap: {str(g.to_pytimedelta())} | '
        #       f'time Restart: {datetime.strftime(time_restart, "%Y-%m-%d %H:%M:%S")}')
    return(timeStopArr, timeRestartArr)
def f_cubicInterp(time, var, time2interp, varName):
    '''
    goal : function to calculate interpolation of the input variable on a new grid
    author: Claudia Acquistapace
    date: 12/11/2020
    input:
        time : datetime Array of the time stamps for the given variable
        Var : variable time serie
        newRes: new resolution on which to interpolate the variable
        varName: string indicating the variable name
    output: 
        dataset containing old and new interpolated variables and coordinates
    '''
    
    from scipy.interpolate import CubicSpline
    #time2interp = np.arange(time[0], time[-1], timedelta(seconds=newRes), dtype='datetime64')
    
    # removing nans from variable timeserie and interpolating over non-nans
    i_valid     = np.where(~np.isnan(var))
    time_valid  = time[i_valid]
    var_valid   = var[i_valid]

    if np.sum(i_valid) != 0:
        Cs          = CubicSpline(time_valid, var_valid)
        varInterp   = Cs(time2interp)
        # setting nans to the corresponding times where the original time serie was nan
        timeStart, timeEnd = f_findExtremesInterval(time_valid)
        for iLoop in range(len(timeEnd)):
            iStart = f_closestTime(time2interp, timeStart[iLoop])
            iEnd   = f_closestTime(time2interp, timeEnd[iLoop])
            varInterp[iStart:iEnd] = np.nan
    else:
        varInterp = np.repeat(np.nan, len(time2interp))
    
    # defining variables as output for the ncdf file
    dimsVar            = ['timeOld']
    dimsVarInterp      = ['time']
    coordsVar          = {'timeOld':time}
    coords             = {'timeOld':time, 'time':time2interp}
    coordsVarInterp    = {"time":time2interp}
    Var                = xr.DataArray(dims=dimsVar, coords=coordsVar, data=var)
    VarInterp          = xr.DataArray(dims=dimsVarInterp, coords=coordsVarInterp, data=varInterp)
    
     # Put everything in a nice Dataset
    variables = {varName                   :Var,
                 varName+'_interp'         :VarInterp}
    global_attributes = {'created_by':'Claudia Acquistapace',
                         'created_on':str(datetime.now()),
                         'comment':''}
    dataset = xr.Dataset(data_vars=variables,
                         coords=coords,
                         attrs=global_attributes)
    return(dataset)
def plot_timeSeries(x,y, ystring,ymin, ymax, timeStartDay, timeEndDay, date, yVarName, pathFig): 
    """
    function to plot time series of any variable
    input:
        x: time coordinate in datetime format
        y: numpyarray of the corresponding variable
        ystrring: string with name and units of the variable to be plotted
        ymin: min value to be plotted for the variable
        ymax: max value to be plotted for the variable
        timeStartDay: datetime start for the xaxis
        timeEndDay: datetime end for the xaxis
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
def f_convertHourString2Int(radarFileName):

    '''
    author: Claudia Acquistapace
    date  : 17/12/2020
    goal  : extract from radar file name the string corresponding to the hour and 
    read the corresponding integer. The function conforms to the following radar file
    name convention: 'msm94_msm_200120_000001_P07_ZEN_v2.nc'
    input : radar file name (string)
    output: hourString, hourInt
    '''
    hourString = radarFileName[-20:-18]
    if hourString[0] == '0':
        hourInt = int(hourString[-1])
    else: 
        hourInt = int(hourString)
    return(hourString, hourInt)       
def f_plotCorrectioTerms(w_ship_shifted, mdv_corr, radarData, PathFigHour, date, hour, timeStart, timeEnd, indHeightSelected, timeExactFinal, i_chirp):
    '''
    author: Claudia Acquistapace
    date  : 23/12/2020
    goal  : plot correction terms for the height selected 
    input: 
        w_ship_shifted: (xarray dataset) dataset containing all correction terms
        mdv_corr: (xarray data array) data array containing the corrected mean Doppler velocity 
        radarData: (xarray dataset) dataset containing original radar data
        PathFigHour: (string) output path for the plot
        date: (string) date of the processed day
        hour: (string) hour string for the hour processed
        timeStart : (datetime) initial time for the plot
        timeEnd : (datetime) final time for the plot
        indHeightSelected: (integer) index of the range gate to be plotted
        timeExactFinal: (datetime array of size (3, dimtime)
        i_chirp: the index for the chirp that is plotted
    Returns plot in the folder 
    -------
    None.
    V_course_vec, V_trasl_vec, V_rot_vec, Ep_vec
    '''    
    from datetime import timedelta

    # reading variables to be plotted
    # radar original mdv
    radarSlice         = radarData.sel(time=slice(timeStart, timeEnd))
    mdv                = radarSlice['vm'].values[:,indHeightSelected]
    datetimeRadar      = radarSlice['time'].values[:]
    millisec           = radarSlice['sample_tms'].values/1000.
    MaxVel             = radarSlice['nqv'].values     
    seq_avg            = radarSlice['seq_avg'].values # number of averaged chirps in each chirp
    chirpRepFreq       = (4 * MaxVel * 94.*10)/ 3.  
    chirpDuration      = 1/chirpRepFreq
    chirpIntegrations  = chirpDuration*seq_avg
    
    # calculating exact radar time stamp for every chirp (assumption: radar time stamp is the final time stamp of the interval)
    datetimeChirps     = f_calculateExactRadarTime(millisec, chirpIntegrations, datetimeRadar) 
    
    # reading time array for the chirp selected
    datetimeChirpRadar = datetimeChirps[i_chirp,:]
    
    # correction terms
    WshipSlice         = w_ship_shifted.sel(time_shifted=slice(timeStart, timeEnd))
    Vwinds             = WshipSlice['V_wind_s'].values[:,:,indHeightSelected]
    Ep                 = WshipSlice['Ep'].values[:,:,indHeightSelected]
    V_rot              = WshipSlice['V_rot'].values
    V_trasl            = WshipSlice['V_trasl'].values
    V_course           = WshipSlice['V_course'].values
    V_radar            = V_rot + V_trasl + V_course
    timeShip           = WshipSlice['time'].values
    
    # calculating time series of ther 
    VwindsEp_x  = Vwinds[0,:]*Ep[0,:]
    VwindsEp_y  = Vwinds[1,:]*Ep[1,:]
    VwindsEp_z  = Vwinds[2,:]*Ep[2,:]
    VradarEp_x  = V_radar[0,:]*Ep[0,:]
    VradarEp_y  = V_radar[1,:]*Ep[1,:]    
    VradarEp_z  = V_radar[2,:]*Ep[2,:]
    VcourseEp_x = V_course[0,:]*Ep[0,:]
    VcourseEp_y = V_course[1,:]*Ep[1,:]    
    
    print(np.shape(timeExactFinal))
    
    # setting time stamps between time start and time end for exact time stamps (from input) and original radar time stamps 
    timeChirpExact = timeExactFinal[(timeExactFinal >= timeStart)*(timeExactFinal < timeEnd)]
    timeRadar      = datetimeChirpRadar[(pd.to_datetime(datetimeChirpRadar) >= timeStart)*(pd.to_datetime(datetimeChirpRadar) <= timeEnd)]
    

    if len(timeChirpExact) > len(timeRadar):
        minlen = len(timeRadar)
        timeChirpExact = timeChirpExact[0:minlen]
    elif len(timeChirpExact) < len(timeRadar):
        minlen = len(timeChirpExact)
        timeRadar = timeRadar[0:minlen]  
    else:
        print('time arrays have same lenght')
  
    # interpolating w_ship and correction denominator on radar time stamps
    w_ship         = V_radar[2, :]
    
    # check on number of values in the time serie: if less than 2, then skip the plot
    print()
    if (np.sum(~np.isnan(w_ship)) > 2): 
    
        i_valid        = np.where(~np.isnan(w_ship))
        w_ship_valid   = w_ship[i_valid]
        timeShip_valid = timeShip[i_valid]
        Cs_ship        = CubicSpline(timeShip_valid, w_ship_valid) 

        W_shipRadar    = Cs_ship(pd.to_datetime(timeRadar))      
        
        
        # interpolating w_ship shifted of the time gap on the radar time stamps
        CS_exact_w     = CubicSpline(timeChirpExact, W_shipRadar)
        W_ship_exact   = CS_exact_w(pd.to_datetime(datetimeRadar))
        
        labelsizeaxes   = 12
        fontSizeTitle   = 12
        fontSizeX       = 12
        fontSizeY       = 12
        fig, ax         = plt.subplots(nrows=4, ncols=1, figsize=(12,16))
        rcParams['font.sans-serif'] = ['Tahoma']
        matplotlib.rcParams['savefig.dpi'] = 100
        plt.gcf().subplots_adjust(bottom=0.15)
        
        # subplot of correction terms along z and mdv to check the time shift
        ax              = plt.subplot(4,1,1)
        ax.spines["top"].set_visible(False)  
        ax.spines["right"].set_visible(False)  
        ax.get_xaxis().tick_bottom()  
        ax.get_yaxis().tick_left() 
        matplotlib.rc('xtick', labelsize=labelsizeaxes)  # sets dimension of ticks in the plots
        matplotlib.rc('ytick', labelsize=labelsizeaxes)  # sets dimension of ticks in the plots
        ax.xaxis_date()
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
        major_ticks = np.arange(timeStart, timeEnd, timedelta(seconds=15), dtype='datetime64')
        minor_ticks = np.arange(timeStart, timeEnd, timedelta(seconds=1), dtype='datetime64')
        ax.tick_params(which = 'both', direction = 'out')
        ax.set_xticks(major_ticks)
        ax.set_xticks(minor_ticks, minor=True)
        ax.grid(which='both')
        ax.grid(which='minor', alpha=0.2)
        ax.grid(which='major', alpha=0.5)
        ax.plot(timeShip, -V_trasl[2,:], color='black', linewidth=0.5, label='-w_heave')
        ax.plot(pd.to_datetime(datetimeRadar), -W_ship_exact, color='blue', label='w_ship shifted interpolated on radar exact time')
        ax.plot(pd.to_datetime(timeRadar), -W_shipRadar, color='purple', label='w_ship interpolated on radar time steps ')
        ax.scatter(pd.to_datetime(timeRadar), -W_shipRadar, color='green', label='w_ship values used for correction')    
        ax.plot(datetimeRadar, mdv, color='green', linewidth=0.2, label='w_radar original')
        ax.plot(datetimeChirpRadar, mdv, color='green', linewidth=1, label='w_radar on radar exact time stamps')
        ax.set_ylim(-2.,2.)                                          # limits of the y-axes 
        ax.legend(frameon=False, loc='upper left')
        ax.set_xlim([pd.to_datetime(timeStart), pd.to_datetime(timeEnd)])                             # limits of the x-axes
        ax.set_title('time serie for the day : '+date+' - '+hour, fontsize=fontSizeTitle, loc='left')
        ax.set_xlabel("time [mm:ss]", fontsize=fontSizeX)
        ax.set_ylabel('w [m s-1]', fontsize=fontSizeY)
    
    
        # subplot of terms along x axis
        ax1              = plt.subplot(4,1,2)
        ax1.spines["top"].set_visible(False)  
        ax1.spines["right"].set_visible(False)  
        ax1.get_xaxis().tick_bottom()  
        ax1.get_yaxis().tick_left() 
        matplotlib.rc('xtick', labelsize=labelsizeaxes)  # sets dimension of ticks in the plots
        matplotlib.rc('ytick', labelsize=labelsizeaxes)  # sets dimension of ticks in the plots
        ax1.xaxis_date()
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
        ax1.tick_params(which = 'both', direction = 'out')
        ax1.set_xticks(major_ticks)
        ax1.set_xticks(minor_ticks, minor=True)
        ax1.grid(which='both')
        ax1.grid(which='minor', alpha=0.2)
        ax1.grid(which='major', alpha=0.5)
        
        ax1.plot(pd.to_datetime(timeShip), VwindsEp_x, color='black', label='(v_winds * Ep)_x')
        ax1.plot(pd.to_datetime(timeShip), VradarEp_x, color='purple', label='(V_radar * Ep)_x')  
        ax1.plot(pd.to_datetime(timeShip), VcourseEp_x, color='cyan', label='(V_course * Ep)_x')  
        ax1.legend(frameon=False, loc='upper left')    
        ax1.set_xlim([pd.to_datetime(timeStart), pd.to_datetime(timeEnd)])                          # limits of the x-axes
        ax1.set_title('time serie of x components of the correction term for the day : '+date+' - '+hour, fontsize=fontSizeTitle, loc='left')
        ax1.set_xlabel("time [mm:ss]", fontsize=fontSizeX)
        ax1.set_ylabel('components on x axis', fontsize=fontSizeY)
        ax1.set_ylim(-5., 5.)                                          # limits of the y-axes 
        ax2 = ax1.twinx()
        ax2.plot(pd.to_datetime(w_ship_shifted['time_shifted'].values),  Ep_vec[0,:], color='blue', label='Ep_x')
        ax2.set_ylabel('Epx []', fontsize=fontSizeY, color='blue')
        ax2.set_ylim(np.nanmin(Ep_vec[0,:]), \
                    np.nanmax(Ep_vec[0,:]))    
        ax2.xaxis.label.set_color('blue')
        ax2.tick_params(axis='y', colors='blue')        
      
        # subplot of terms along y axis
        ax3              = plt.subplot(4,1,3)
        ax3.spines["top"].set_visible(False)  
        ax3.spines["right"].set_visible(False)  
        ax3.get_xaxis().tick_bottom()  
        ax3.get_yaxis().tick_left() 
        matplotlib.rc('xtick', labelsize=labelsizeaxes)  # sets dimension of ticks in the plots
        matplotlib.rc('ytick', labelsize=labelsizeaxes)  # sets dimension of ticks in the plots
        ax3.xaxis_date()
        ax3.xaxis.set_major_formatter(mdates.DateFormatter("%M:%S"))
        ax3.tick_params(which = 'both', direction = 'out')
        ax3.set_xticks(major_ticks)
        ax3.set_xticks(minor_ticks, minor=True)
        ax3.grid(which='both')
        ax3.grid(which='minor', alpha=0.2)
        ax3.grid(which='major', alpha=0.5)
        ax3.plot(pd.to_datetime(timeShip), VwindsEp_y, color='black', label='(v_winds * Ep)_y')
        ax3.plot(pd.to_datetime(timeShip), VradarEp_y, color='purple', label='(V_radar * Ep)_y')  
        ax3.plot(pd.to_datetime(timeShip), VcourseEp_y, color='cyan', label='(V_course * Ep)_y')                                 
        ax3.legend(frameon=False, loc='upper left')
        ax3.set_ylim(-8., 8.)                                          # limits of the y-axes 
        ax3.set_xlim([pd.to_datetime(timeStart), pd.to_datetime(timeEnd)])                         # limits of the x-axes
        ax3.set_title('time serie of y components for the day : '+date+' - '+hour, fontsize=fontSizeTitle, loc='left')
        ax3.set_xlabel("time [mm:ss]", fontsize=fontSizeX)
        ax3.set_ylabel('components on y axis', fontsize=fontSizeY)
        ax4 = ax3.twinx()
        ax4.plot(pd.to_datetime(w_ship_shifted['time_shifted'].values),  Ep_vec[1,:], color='blue', label='Ep_y')
        ax4.set_ylabel('Epx []', fontsize=fontSizeY, color='blue')
        ax4.set_ylim(np.nanmin(Ep_vec[1,:]), \
                    np.nanmax(Ep_vec[1,:]))    
        ax4.xaxis.label.set_color('blue')
        ax4.tick_params(axis='y', colors='blue')                
                    
        # subplot of terms along z axis
        ax5              = plt.subplot(4,1,4)
        ax5.spines["top"].set_visible(False)  
        ax5.spines["right"].set_visible(False)  
        ax5.get_xaxis().tick_bottom()  
        ax5.get_yaxis().tick_left() 
        matplotlib.rc('xtick', labelsize=labelsizeaxes)  # sets dimension of ticks in the plots
        matplotlib.rc('ytick', labelsize=labelsizeaxes)  # sets dimension of ticks in the plots
        ax5.xaxis_date()
        ax5.xaxis.set_major_formatter(mdates.DateFormatter("%M:%S"))
    
        ax5.tick_params(which = 'both', direction = 'out')
        ax5.set_xticks(major_ticks)
        ax5.set_xticks(minor_ticks, minor=True)
        ax5.grid(which='both')
        ax5.grid(which='minor', alpha=0.2)
        ax5.grid(which='major', alpha=0.5)
        ax5.plot(pd.to_datetime(timeShip), VwindsEp_z, color='black', label='(v_winds * Ep)_z')
        ax5.plot(pd.to_datetime(timeShip), VradarEp_z, color='purple', label='(V_radar * Ep)_z')  
        ax5.legend(frameon=False, loc='upper left')
        ax5.set_ylim(-8., 8.)                                          # limits of the y-axes 
    
        ax5.set_xlim([pd.to_datetime(timeStart), pd.to_datetime(timeEnd)])                           # limits of the x-axes
        ax5.set_title('time serie of y components for the day : '+date+' - '+hour, fontsize=fontSizeTitle, loc='left')
        ax5.set_xlabel("time [mm:ss]", fontsize=fontSizeX)
        ax5.set_ylabel('components on z axis', fontsize=fontSizeY)
        ax6 = ax5.twinx()
        ax6.plot(pd.to_datetime(w_ship_shifted['time_shifted'].values),  Ep_vec[2,:], color='blue', label='Ep_z')
        ax6.set_ylabel('Epz []', fontsize=fontSizeY, color='blue')
        ax6.set_ylim(np.nanmin(Ep_vec[2,:]), \
                    np.nanmax(Ep_vec[2,:]))   
        ax6.xaxis.label.set_color('blue')
        ax6.tick_params(axis='y', colors='blue')
        fig.tight_layout()
        fig.savefig(PathFigHour+date+'_'+hour+'_chirp_'+str(i_chirp)+'_correctionTerm.png', format='png')
        
    else:
        print('skipping the plot - ')
    return(PathFigHour+date+'_'+hour+'_chirp_'+str(i_chirp)+'_correctionTerm.png - plot done')
                 

#%%
    

################ User data : insert here paths, filenames and parameters to set ########################

# paths to the different data files and output directories for plots
pathFolderTree  = '/Volumes/Extreme SSD/ship_motion_correction_merian/'
pathNcDataAnc   = pathFolderTree+'/ncdf_ancillary/'
shipFile        = pathFolderTree+'/ship_data/new/shipData_all2.nc'

# days without modelled data: [datetime(2020,1,22), datetime(2020,1,25), datetime(2020,2,2), datetime(2020,2,3)]


### instrument position coordinates [+5.15m; +5.40m;−15.60m]
r_FMCW          = [5.15 , 5.4, -15.6] # [m]


# generating array of days for the dataset
Eurec4aDays     = pd.date_range(datetime(2020,1,19),datetime(2020,2,19),freq='d')
NdaysEurec4a    = len(Eurec4aDays)
#%%

'''setting indDay == 1 to process the 20-01-2020 and setting indHour= 7 to process for test the hour 06:07'''
for indDay in range(NdaysEurec4a):
    indDay          = 1
    dayEu           = Eurec4aDays[indDay]
    yy              = str(dayEu)[0:4]
    mm              = str(dayEu)[5:7]
    dd              = str(dayEu)[8:10]
    date            = yy+mm+dd
    # setting dates strings
    date            = dd+mm+yy      #'04022020'
    dateRadar       = yy[0:2]+mm+dd #'200204'
    dateReverse     = yy+mm+dd      #'20200204'   
    
    
    # establishing paths for data input and output
    pathRadar       = pathFolderTree+'/w_band_radar_data/EUREC4Aprocessed_20201102/'+yy+'/'+mm+'/'+dd+'/'
    pathFig         = pathFolderTree+'plots/merged_algs/'+yy+'/'+mm+'/'+dd+'/'
    radarFileList   = np.sort(glob.glob(pathRadar+'msm94_msm_'+dateRadar+'*ZEN_v2.nc'))
    pathOutData     = pathFolderTree+'/corrected_data/'+yy+'/'+mm+'/'+dd+'/'
    
    print(radarFileList)
    # loop on hours for the selected day
    for indHour in range(len(radarFileList)):
        indHour = 6
        radarFileName = radarFileList[6]
        # read radar height array for interpolation of model data from one single radar file
        radarDatatest          = xr.open_dataset('/Volumes/Extreme SSD/ship_motion_correction_merian/w_band_radar_data/EUREC4Aprocessed_20201102/2020/01/19/msm94_msm_200119_000002_P07_ZEN_compact_v2.nc')
        heightRad = radarDatatest['range'].values

        # reading hour string and int quantity
        hour, hourInt =  f_convertHourString2Int(radarFileName)
        
        # check if file has already been processed or not
        #if os.path.isfile(pathOutData+date+'_'+hour+'msm94_msm_ZEN_corrected.nc'):
        #    print(date+'_'+hour+' - already processed')
        #else:
        checked = 0
        
        if checked == 0:

            # check if folder for plots and for outputs exists, otherwise creates it
            #PathFigHour     = pathFig+hour+'/'
            PathFigHour = 
            Path(PathFigHour).mkdir(parents=True, exist_ok=True)  
            Path(pathOutData).mkdir(parents=True, exist_ok=True)  
         
            # setting starting and ending time for the hour an
            timeStart    = datetime(int(yy),int(mm), int(dd),hourInt,0,0)
            if hour != '23':
                timeEnd      = datetime(int(yy),int(mm), int(dd),hourInt+1,0,0)
            else:
                timeEnd      = datetime(int(yy),int(mm), int(dd),hourInt,59,59)
            timeArr         = pd.to_datetime(np.arange(timeStart, timeEnd, timedelta(seconds=1), dtype='datetime64'))
    
            print('processing '+date+', hour :'+hour)
            print('**************************************')
            
            print('* reading correction terms for the day/hour :'+yy+'-'+mm+'-'+dd+', hour: '+hour)
            '''reading correction terms for the day and interpolating them on the 0.25s time resolution'''
            #                   ----------------------------
            print(' -- reading and resampling V_wind_s dataset')
            if (date == '02022020') or (date == '03022020'):
                
                           # saving variables of wind_s in a xarray dataset
                            dims2             = ['time','height']
                            coords2           = {"time":timeArr, "height":heightRad}
                            
                            v_wind_s_x        = xr.DataArray(dims=dims2, coords=coords2, data=np.full([len(timeArr), len(heightRad)], np.nan),
                                                     attrs={'long_name':'zonal wind speed profile in ship reference system from ICON-LEM 1.25 km',
                                                            'units':'m s-1'})
                            v_wind_s_y        = xr.DataArray(dims=dims2, coords=coords2, data=np.full([len(timeArr), len(heightRad)], np.nan),
                                                     attrs={'long_name':'meridional wind speed in ship reference system from ICON-LEM 1.25 km',
                                                            'units':'m s-1'})
                            variables         = {'vs_x':v_wind_s_x,
                                                 'vs_y':v_wind_s_y}
                            global_attributes = {'created_by':'Claudia Acquistapace',
                                                 'created_on':str(datetime.now()),
                                                 'comment':'wind direction and speed in the ship reference system for RV Merian'}
                            V_windS_hour      = xr.Dataset(data_vars = variables,
                                                               coords = coords2,
                                                               attrs = global_attributes)
            else:
                # reading v_wind_s for the selected day
                V_windS_dataset    = xr.open_dataset(pathNcDataAnc+yy+'-'+mm+'-'+dd+'_wind_s_dataset.nc')
                # interpolating time array of V_wind_s since it has 86400 elements instead of 86399 as all other datasets
                V_windS_hour       = V_windS_dataset.sel(time=slice(timeStart, timeEnd))
            #                   ----------------------------
            #V_windS_dataset    = xr.open_dataset(pathNcDataAnc+yy+'-'+mm+'-'+dd+'_wind_s_dataset.nc')
            # interpolating time array of V_wind_s since it has 86400 elements instead of 86399 as all other datasets
            #V_windS_hour       = V_windS_dataset.sel(time=slice(timeStart, timeEnd))
            # calculating 
            
            
            print(' -- reading and resampling Ep dataset')
            # reading file with Ep dataset and and extracting data for the selected day to process
            Ep_dataset         = xr.open_dataset(pathNcDataAnc+dateReverse+'_Ep_dataset.nc')
            Ep_hour            = Ep_dataset.sel(time=slice(timeStart, timeEnd))
            del Ep_dataset
            Ep                 = Ep_hour['Ep']
            timeEp             = pd.to_datetime(Ep_hour['time'].values)
            #%                   ----------------------------
            print(' -- reading and resampling V_rotational and V_traslational dataset')
            # reading file with V_rot/V_trasl terms and extracting data for the selected day to process
            V_rot_dataset      = xr.open_dataset(pathNcDataAnc+'V_rot_V_trasl_dataset.nc')
            Vrot_hour           = V_rot_dataset.sel(time=slice(timeStart, timeEnd))
            del V_rot_dataset
            #                   -------------------------
            print(' -- reading and resampling V_course dataset')
            # reading V_course term and extracting data for the selected day
            V_course_dataset   = xr.open_dataset(pathNcDataAnc+'V_course_dataset.nc')
            V_course_hour      = V_course_dataset.sel(time=slice(timeStart, timeEnd))
            del V_course_dataset
            
            # shifting time stamp to the center of the time intervals for the dataset
            #V_course_hour           = f_shiftTimeDataset(V_course_hour)
            V_course_x         = V_course_hour['v_course_x'].values
            V_course_y         = V_course_hour['v_course_y'].values
            timeCourse         = V_course_hour['time'].values
            
            print('---- end of reading dataset for correcting ship motions ----')
            
            # check on dimensions: if V_windS has dimensions different than others, interpolate on correct time grid
            if len(Vrot_hour['time'].values) != len(V_windS_hour['time'].values):
                timeInterp   = Vrot_hour['time'].values
                V_windS_hour = V_windS_hour.interp(time=timeInterp) 
                
            if (len(V_windS_hour['height'].values) != 550):
                V_windS_hour = V_windS_hour.interp(height=heightRad) 
                
                
                
            print('* calculation of w_ship correction term, depending on wether table works or not (Ep vector) ')
            
            time_correction    = pd.to_datetime(Vrot_hour['time'].values)
            dimTime            = len(time_correction)
            
            # putting correction terms in vectorial form for calculating vectorial expression of formula 29
            # Ep and Ez point vectors
            Ep_vec             = Ep_hour['Ep'].values
            Ez                 = np.repeat(np.array([0.,0.,1.]), len(Ep[0,:])).reshape(3,len(Ep[0,:]))
            den_scalar_product = np.nansum(Ez*Ep_vec, axis=0)
            
            # wind speed in ship reference system
            #if vs_x.shape[1] != 550:
                
            vs_x               = V_windS_hour['vs_x'].values
            vs_y               = V_windS_hour['vs_y'].values
            vs_z               = np.repeat(np.repeat(0., len(V_windS_hour['vs_y'].values)), 550).reshape(len(V_windS_hour['vs_y'].values), 550)
            V_wind_s           = np.array([vs_x,vs_y,vs_z])
        
            # V course, V rotation and V translation 
            V_course_vec       = np.array([V_course_x, V_course_y, np.repeat(0, len(V_course_y))])
            V_rot_vec          = np.array([Vrot_hour['Vrot_x2'].values, Vrot_hour['Vrot_y2'].values, Vrot_hour['Vrot_z2'].values])
            V_trasl_vec        = np.array([np.repeat(0., len(Vrot_hour['V_trasl'].values)), \
                                          np.repeat(0., len(Vrot_hour['V_trasl'].values)), \
                                          Vrot_hour['V_trasl'].values]).reshape(3,len(Ep[0,:]))
            
            # application of the formula 29 (or 7) from the theory attached
            V_radar_vec        = V_rot_vec + V_trasl_vec + V_course_vec
            num_termC          = V_wind_s - np.repeat(V_radar_vec, 550).reshape(3, dimTime, 550)
            EP_matrix          = np.repeat(Ep_vec, 550).reshape(3, dimTime, 550)
            num_correction     = np.nansum(num_termC*EP_matrix, axis=0) 
            den_correction     = np.repeat(den_scalar_product, 550).reshape(dimTime, 550)
            correction_term    = num_correction
            heightWind         = V_windS_hour['height'].values
            
            
            # save xarray dataset containing the correction terms for the hour
            dimt              = ['time']
            dims2             = ['time','height']
            dim3              = ['axis','time']
            dimAll            = ['axis', 'time', 'height']
            coords2           = {"time":time_correction, "height":heightWind}
            coords            = {"time":time_correction}
            coords3           = {"time":time_correction, "axis": np.arange(3)}
            coordsAll         = {"time":time_correction, "axis": np.arange(3),  "height":heightWind}
            Ep                = xr.DataArray(dims=dimAll, coords=coordsAll, data=EP_matrix,
                                     attrs={'long_name':'radar pointing vector',
                                            'units':''})
            V_wind_s          = xr.DataArray(dims=dimAll, coords=coordsAll, data=V_wind_s,
                                     attrs={'long_name':'Wind velocity in ship ref system',
                                            'units':'m s-1'})  
            V_course          = xr.DataArray(dims=dim3, coords=coords3, data=V_course_vec,
                                     attrs={'long_name':'Course velocity',
                                            'units':'m s-1'})     
            V_trasl           = xr.DataArray(
                dims=dim3, coords=coords3, data=V_trasl_vec,
                                     attrs={'long_name':'Traslational velocity',
                                            'units':'m s-1'})           
            V_rot             = xr.DataArray(dims=dim3, coords=coords3, data=V_rot_vec,
                                     attrs={'long_name':'Rotational velocity',
                                            'units':'m s-1'})        
            den_scalar_prod   = xr.DataArray(dims=dimt, coords=coords, data=den_scalar_product,
                                     attrs={'long_name':'denominator of the formula as in reference paper',
                                            'units':''})
            corr_term         = xr.DataArray(dims=dims2, coords=coords2, data=correction_term,
                                     attrs={'long_name':'correction term numerator as appears in the formula of the reference paper',
                                            'units':'m s-1'})
            variables         = {'den_scalar_product':den_scalar_prod,
                                 'correction_term':corr_term, 
                                 'V_wind_s': V_wind_s, 
                                 'V_course': V_course, 
                                 'V_trasl': V_trasl, 
                                 'V_rot': V_rot, 
                                 'Ep': Ep}
            global_attributes = {'CREATED_BY'       : 'Claudia Acquistapace',
                                'CREATED_ON'       :  str(datetime.now()),
                                'FILL_VALUE'       :  'NaN', 
                                'PI_NAME'          : 'Claudia Acquistapace',
                                'PI_AFFILIATION'   : 'University of Cologne (UNI), Germany', 
                                'PI_ADDRESS'       : 'Institute for geophysics and meteorology, Pohligstrasse 3, 50969 Koeln', 
                                'PI_MAIL'          : 'cacquist@meteo.uni-koeln.de',
                                'DO_NAME'          : 'University of Cologne - Germany',
                                'DO_AFFILIATION'   : 'University of Cologne - Germany',
                                'DO_address'       : 'Institute for geophysics and meteorology, Pohligstrasse 3, 50696 Koeln',
                                'DO_MAIL'          : 'cacquist@meteo.uni-koeln.de',
                                'DS_NAME'          : 'University of Cologne - Germany',
                                'DS_AFFILIATION'   : 'University of Cologne - Germany',
                                'DS_address'       : 'Institute for geophysics and meteorology, Pohligstrasse 3, 50696 Koeln',
                                'DS_MAIL'          : 'cacquist@meteo.uni-koeln.de',
                                'DATA_DESCRIPTION' : 'hourly cloud radar measurements on Maria S. Merian ship during EUREC4A campaign',
                                'DATA_DISCIPLINE'  : 'Atmospheric Physics - Remote Sensing Radar Profiler',
                                'DATA_GROUP'       : 'Experimental;Profile;Moving',
                                'DATA_LOCATION'    : 'Research vessel Maria S. Merian - Atlantic Ocean ',
                                'DATA_SOURCE'      : 'Radar.Standard.Moments.Ldr for ship data;run by uni_koeln https://github.com/igmk/w-radar/tree/master/scripts',
                                'DATA_PROCESSING'  : 'https://github.com/ClauClouds/ship-motion-correction-for-EUREC4A-campaign',
                                'INSTRUMENT_MODEL' : '94 GHz dual polatization FMCW band radar',
                                'MDF_PROGRAM_USED' :  'EUREC4A',
                                 'COMMENT'         : 'correction terms tensors based on algorithm described in ship_motion_correction.pdf' }
            w_ship_dataset    = xr.Dataset(data_vars = variables,
                                              coords = coords2,
                                               attrs = global_attributes)
            
            # assign the central time stamp to the the correction terms
            w_ship_shifted    = f_shiftTimeDataset(w_ship_dataset)
            
            # save ship motion correction terms in a separate file
            w_ship_shifted.to_netcdf(pathOutData+date+'_'+hour+'_w_ship.nc')
            
            
            print('* read radar data for the selected hour')
        
            if os.path.exists(radarFileList[indHour]):
                radarData          = xr.open_dataset(radarFileList[indHour])
                mdv                = radarData['vm'].values
                datetimeRadar      = radarData['time'].values
                rangeRadar         = radarData['range'].values
                mdv[mdv == -999.]  = np.nan
                range_offset       = rangeRadar[radarData['range_offsets'].values]
                indRangeChirps     = radarData['range_offsets'].values
                millisec           = radarData['sample_tms'].values/1000.
                NtimeRadar         = len(datetimeRadar)
                sampleDur          = radarData['SampDur'].values # full sample duration [s]
                MaxVel             = radarData['nqv'].values     
                seq_avg            = radarData['seq_avg'].values # number of averaged chirps in each chirp
                chirpRepFreq       = (4 * MaxVel * 94.*10)/ 3.  
                chirpDuration      = 1/chirpRepFreq
                chirpIntegrations  = chirpDuration*seq_avg
                Nchirps            = len(chirpIntegrations)
            
                # calculating exact radar time stamp for every chirp (assumption: radar time stamp is the final time stamp of the interval)
                datetimeChirps     = f_calculateExactRadarTime(millisec, chirpIntegrations, datetimeRadar)
                timeChirp1         = pd.to_datetime(datetimeChirps[0,:])
                timeChirp2         = pd.to_datetime(datetimeChirps[1,:])
                timeChirp3         = pd.to_datetime(datetimeChirps[2,:])
                 
                   
                # plot of 2d map of mean doppler velocity and time series of correction if not existing 
                if not (os.path.exists(PathFigHour+date+'_'+hour+'_mdv_2dmaps.png')) :
    
                    # plot on mean doppler velocity time height 
                    plot_2Dmaps(datetimeRadar, \
                                rangeRadar, \
                                mdv, \
                                'Mean Doppler velocity', \
                                -6., \
                                4., \
                                100., \
                                2200., \
                                timeStart, \
                                timeEnd, \
                                'seismic', \
                                date+'_'+hour, \
                                'mdv', \
                                PathFigHour)
             
                #### -------------------------------------------------------------------------- ####
                        
                # calculation of time lag between radar time serie and ship time serie
                print('* calculate time shift between ship and radar time for each chirp ')        
                
                # reading ship data
                denTimeSerie      = w_ship_shifted['den_scalar_product'].values
                timeShip          = w_ship_shifted['time_shifted'].values
                w_ship            = w_ship_shifted['correction_term'].values
            
                # assigning lenght of the mean doppler velocity time serie for calculating time shift 
                # with 3 sec time resolution, 200 corresponds to 10 min
                NtimeStampsRun   = 200
                
                
                # time shift array to be tested 
                DeltaTmin        = -3.
                DeltaTmax        = 3.
                res              = 0.05
                DimDeltaT        = (DeltaTmax- DeltaTmin)/res
                DeltaTimeShift   = np.arange(DeltaTmin, DeltaTmax, step=res)
                
                # calculating time shift and corrected time array for radar mean doppler velocity for each chirp
                timeShiftArray   = np.zeros((Nchirps))
                timeShiftArray.fill(-999.)
                heightSelArray   = np.zeros((Nchirps))
                heightSelArray.fill(-999.)
                timeBeginSamples = np.zeros((Nchirps), dtype='datetime64[ns]')
                timeEndSamples = np.zeros((Nchirps), dtype='datetime64[ns]')
                timeExactFinal   = np.zeros((Nchirps, len(datetimeRadar)), dtype='datetime64[ns]')
                
                
                for i_chirp in range(0, Nchirps):
                    print('- processing chirp '+str(i_chirp))
                
                    #assigning string identifying the chirp that is processed
                    chirp = 'chirp_'+str(i_chirp)
                    
                    # selecting index of min and max height of the chirp
                    i_h_min     = f_closest(rangeRadar, range_offset[i_chirp])
                    if i_chirp+1 != 3:
                        i_h_max = f_closest(rangeRadar, range_offset[i_chirp+1])
                    else:
                        i_h_max = -1
                        
                    # reading radar time array of the chirp
                    if i_chirp == 0:
                        timeSerieRadar   = timeChirp1
                    elif i_chirp == 1:
                        timeSerieRadar   = timeChirp2
                    else:
                        timeSerieRadar   = timeChirp3
                        
                    print('- finding best mdv time seried between '+str(rangeRadar[i_h_min])+' and '+str(rangeRadar[i_h_max]))
                        
                    # reading mdv values for the selected chirp
                    mvd_chirp  = mdv[:,i_h_min:i_h_max]
                    dimHchirp  = np.shape(mvd_chirp)[1]
                    rangeChirp = rangeRadar[i_h_min:i_h_max]
                    
                        
                    # search for at least 10 min of consecutive w obs in the chirp
                    w_radar, indHeightBest, timeRadarSel, w_radar_meanCol = f_findMdvTimeSerie_VMerge(mvd_chirp, \
                                                                            timeSerieRadar, \
                                                                            rangeChirp, \
                                                                            NtimeStampsRun, \
                                                                            PathFigHour, \
                                                                            chirp)
                    # adapting index for the chirp matrix to the entire matrix by adding i_h_min-1
                    indHeightBest = indHeightBest+i_h_min-1
                    
                    # reading correction time serie corresponding to the selected height in the selected time interval
                    i_time_ship_best  = (pd.to_datetime(timeShip) >= timeRadarSel[0]) * (pd.to_datetime(timeShip) <= timeRadarSel[-1])
                    w_ship_best       = w_ship[i_time_ship_best, indHeightBest]
                                
                    heightSelArray[i_chirp] = indHeightBest
                    timeBeginSamples[i_chirp] = pd.to_datetime(timeRadarSel[0])
                    if len(timeRadarSel) >= 40:
                        timeEndSamples[i_chirp] = pd.to_datetime(timeRadarSel[40])
                        TimeEndPlot     = timeRadarSel[40]

                    else:
                        timeEndSamples[i_chirp] = pd.to_datetime(timeRadarSel[-1])
                        TimeEndPlot     = timeRadarSel[-1]

                    print(' - calculating time shift for the chirp '+str(i_chirp))
                    # calculating time shift for the chirp
                    if np.sum(np.where(~np.isnan(w_radar))) != 0:
                        
                        i_valid    = np.where(~np.isnan(w_radar_meanCol))
                        w_valid    = w_radar_meanCol[i_valid]
                        time_valid = timeRadarSel[i_valid]
                        if (len(w_valid) > 10):
                            # interpolating mean radar mdv speed on the ship time resolution for time shift calculation
                            CS_rad                  = CubicSpline(time_valid, w_valid)
                            WChirpshipRes           = CS_rad(pd.to_datetime(timeShip[i_time_ship_best]))      
                            
                            # calculating time shift
                            timeShiftArray[i_chirp] = f_calcTimeShift(WChirpshipRes, \
                                                                      DeltaTimeShift, \
                                                                      w_ship_best, \
                                                                      pd.to_datetime(timeShip[i_time_ship_best]), \
                                                                      PathFigHour, \
                                                                      chirp, \
                                                                      date, \
                                                                      hour)
                        else:
                            timeShiftArray[i_chirp] = np.nan                            
                    else:
                        
                        timeShiftArray[i_chirp] = np.nan
            
                    print('- recalculating exact time for ship data ')
                    # recalculating exact time including time shift due to lag
                    if ~np.isnan(timeShiftArray[i_chirp]):
                        timeExact  = pd.to_datetime(timeSerieRadar) - timedelta(seconds=timeShiftArray[i_chirp])
                
                    else:
                        timeExact  = pd.to_datetime(timeSerieRadar) 
                
                    timeExactFinal[i_chirp, :] = pd.to_datetime(timeExact)
                    
                    # interpolating ship data on radar time 
                    Cs_ship        = CubicSpline(timeShip, w_ship[:, indHeightBest])
                    W_shipRadar    = Cs_ship(pd.to_datetime(timeSerieRadar))
                    
                    # interpolating ship data on the exact time again, with the right time found after the calculating
                    # and the addition of the time shift     
                    CS_exact        = CubicSpline(timeExact, W_shipRadar)
                    W_ship_exact    = CS_exact(pd.to_datetime(timeSerieRadar))
                    timePlot        = pd.to_datetime(timeExact)
                    TimeBeginPlot   = timeRadarSel[0]
                    
                    
                    labelsizeaxes   = 16
                    fontSizeTitle   = 16
                    fontSizeX       = 16
                    fontSizeY       = 16
                    cbarAspect      = 10
                    fontSizeCbar    = 16

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
                    major_ticks = np.arange(TimeBeginPlot, TimeEndPlot, 50000000, dtype='datetime64')
                    minor_ticks = np.arange(TimeBeginPlot, TimeEndPlot, 1000000, dtype='datetime64')
                    ax.tick_params(which = 'both', direction = 'out')
                    ax.set_xticks(major_ticks)
                    ax.set_xticks(minor_ticks, minor=True)
                    ax.grid(which='both')
                    ax.grid(which='minor', alpha=0.2)
                    ax.grid(which='major', alpha=0.5)
                    ax.plot(timeRadarSel, w_radar_meanCol, color='red', label='mean w_radar over column')
                    ax.plot(timeRadarSel, w_radar, linewidth = 0.2, color='red', label='w_radar at one height') 
                    ax.plot(timeSerieRadar, W_shipRadar, color='blue', linewidth=0.2, label='w_ship original')
                    ax.plot(timePlot, W_shipRadar, color='blue', label='w_ship shifted of deltaT found')
                    ax.scatter(timeSerieRadar, W_ship_exact, color='green', label='w_ship shifted interpolated on radar exact time')
                    ax.set_ylim(-4.,2.)   
                    ax.legend(frameon=False)
                    ax.set_xlim(TimeBeginPlot,TimeEndPlot)                                 # limits of the x-axes
                    ax.set_title(date+' '+hour+':'+'0'+str(int(hour)+1)+' $\Delta$T = '+str(round(timeShiftArray[i_chirp],2), fontsize=fontSizeTitle, loc='left')
                    ax.set_xlabel("time [hh:mm:ss]", fontsize=fontSizeX)
                    ax.set_ylabel('w [m s-1]', fontsize=fontSizeY)
                    fig.tight_layout()
                    fig.savefig(PathFigHour+date+'_'+hour+'_chirp_'+str(i_chirp)+'_quicklook_correction.png', format='png')
    
                    
                    
        
                print(' ### time shifts found for each chirp: ', timeShiftArray)  
                #### -------------------------------------------------------------------------- ####
                print('* calculating corrected mean Doppler velocity')
                
                # calculating now correction term based on the time shift found
                mdv_corrected = np.zeros((len(datetimeRadar), len(rangeRadar)))
                mdv_fake      = np.zeros((len(datetimeRadar), len(rangeRadar)))
                # loop on range gates: for each height, find the exact radar time
                for indHeight in range(len(rangeRadar)):
                    mdvTimeSerie = mdv[:, indHeight]
                    
                    # assigning radar time chirp based on the range selected in the loop
                    if indHeight <= indRangeChirps[1]:
                        timeExactShipChirp = timeExactFinal[0, :]
                        timeSerieRadar   = timeChirp1
                    if  (indHeight > indRangeChirps[1]) * (indHeight <= indRangeChirps[2]):
                        timeExactShipChirp = timeExactFinal[1, :]
                        timeSerieRadar   = timeChirp2
                    elif (indHeight > indRangeChirps[2]) * (indHeight <= 550):
                        timeExactShipChirp = timeExactFinal[2, :]
                        timeSerieRadar   = timeChirp3
                        
                    # interpolating w_ship and correction denominator on radar time stamps
                    Cs_ship        = CubicSpline(timeShip, w_ship[:, indHeight])
                    W_shipRadar    = Cs_ship(pd.to_datetime(timeSerieRadar))      
                    Cs_den         = CubicSpline(timeShip, denTimeSerie)
                    denShipRadar   = Cs_den(pd.to_datetime(timeSerieRadar)) 
                    
                    
                    # interpolating w_ship shifted of the time gap on the radar time stamps
                    CS_exact_w     = CubicSpline(timeExactShipChirp, W_shipRadar)
                    W_ship_exact   = CS_exact_w(pd.to_datetime(timeSerieRadar))
        
                    # interpolating den shifted of the time gap on the radar time stamps
                    CS_exact_den   = CubicSpline(timeExactShipChirp, denShipRadar)
                    den_exact      = CS_exact_den(pd.to_datetime(timeSerieRadar))
                    
                    # calculating corrected mdv for radar time stamps
                    mdv_corrected[:, indHeight] = - mdvTimeSerie/den_exact + W_ship_exact/den_exact
                    mdv_fake[:, indHeight]      = - mdvTimeSerie/denShipRadar + W_shipRadar/denShipRadar
              
      
                # plot of the correction terms for the selected hour at the chirp that is processed
                for indChirpPlot in range(Nchirps):
                    i_chirp = indChirpPlot 
                    if (heightSelArray[i_chirp] != -999.):
                    
                        f_plotCorrectioTerms(w_ship_shifted, \
                                                mdv_corrected, \
                                                    radarData, \
                                                        PathFigHour, \
                                                            date, \
                                                                hour, \
                                                                    pd.to_datetime(timeBeginSamples[i_chirp]).replace(microsecond=0), \
                                                                        pd.to_datetime(timeEndSamples[i_chirp]).replace(microsecond=0), \
                                                                            (int(heightSelArray[i_chirp])), \
                                                                                pd.to_datetime(timeExactFinal[i_chirp, :]), 
                                                                                i_chirp)
             
            else:
                print ('File for '+dateRadar+'_'+hour+' does not exist')
          
            
            # applying rolling average to the data
            print('* Applying running mean on 3 time steps for improving data quality')
            if np.sum(~np.isnan(mdv_corrected)) != 0:
                df        = pd.DataFrame(mdv_corrected, index=datetimeRadar, columns=rangeRadar)
                mdv_roll3 = df.rolling(window=3,center=True, axis=0).apply(lambda x: np.nanmean(x)) 
            else:
                mdv_r3    = np.zeros((len(datetimeRadar), len(rangeRadar)), dtype=float)
                mdv_r3[:] = np.nan
                mdv_roll3 = pd.DataFrame(mdv_r3, index=datetimeRadar, columns=rangeRadar)
                
            # saving corrrected mean doppler velocities matrices to be added to radar data
            # save xarray dataset containing the correction terms for the hour
            print('* saving data in ncdf format')
            dims2             = ['time','height']
            coords2           = {"time":datetimeRadar, "height":rangeRadar}
            mdv_corr          = xr.DataArray(dims=dims2, coords=coords2, data=mdv_corrected,
                                     attrs={'long_name':'Mean Doppler velocity corrected \
                                            using algorithm described in \
                                                https://github.com/ClauClouds/ship-motion-correction-for-EUREC4A-campaign',
                                            'units':'ms-1'})
            mdv_corr_smooth   = xr.DataArray(dims=dims2, coords=coords2, data=mdv_roll3.values,
                                     attrs={'long_name':'Mean Doppler velocity corrected \
                                            using algorithm described in \
                                                https://github.com/ClauClouds/ship-motion-correction-for-EUREC4A-campaign \
                                                and smoothed using 3 time steps running mean',
                                            'units':'ms-1'})
    
            global_attributes = {'CREATED_BY'     : 'Claudia Acquistapace',
                                 'CREATED_ON'       : str(datetime.now()),
                                 'FILL_VALUE'       : 'NaN', 
                                 'PI_NAME'          : 'Claudia Acquistapace',
                                 'PI_AFFILIATION'   : 'University of Cologne (UNI), Germany', 
                                 'PI_ADDRESS'       : 'Institute for geophysics and meteorology, Pohligstrasse 3, 50969 Koeln', 
                                 'PI_MAIL'          : 'cacquist@meteo.uni-koeln.de',
                                 'DO_NAME'          : 'University of Cologne - Germany',
                                 'DO_AFFILIATION'   : 'University of Cologne - Germany',
                                 'DO_address'       : 'Institute for geophysics and meteorology, Pohligstrasse 3, 50696 Koeln',
                                 'DO_MAIL'          : 'cacquist@meteo.uni-koeln.de',
                                 'DS_NAME'          : 'University of Cologne - Germany',
                                 'DS_AFFILIATION'   : 'University of Cologne - Germany',
                                 'DS_address'       : 'Institute for geophysics and meteorology, Pohligstrasse 3, 50696 Koeln',
                                 'DS_MAIL'          : 'cacquist@meteo.uni-koeln.de',
                                 'DATA_DESCRIPTION' : 'hourly cloud radar measurements on Maria S. Merian (msm) ship during EUREC4A campaign',
                                 'DATA_DISCIPLINE'  : 'Atmospheric Physics - Remote Sensing Radar Profiler',
                                 'DATA_GROUP'       : 'Experimental;Profile;Moving',
                                 'DATA_LOCATION'    : 'Research vessel Maria S. Merian - Atlantic Ocean ',
                                 'DATA_SOURCE'      : 'Radar.Standard.Moments processed using the standard code available at \
                                     uni_koeln https://github.com/igmk/w-radar/tree/master/scripts',
                                 'DATA_PROCESSING'  : 'additional data processing is performed for applying ship motion correction. \
                                     the code used is available at https://github.com/ClauClouds/ship-motion-correction-for-EUREC4A-campaign',
                                 'INSTRUMENT_MODEL' : '94 GHz dual polatization FMCW band radar',
                                 'MDF_PROGRAM_USED' : 'EUREC4A',
                                 'COMMENT'          : 'The file is the product of two processings. First data are \
                                     processed using the standard matlab script from UNI. Then , the ship motion \
                                         correction algorithm is applied to the mean Doppler velocity field. The code \
                                             and a description of the algorithm are available at (see DATA_PROCESSING. \
                                                 Two additional variables are provided: \
                                   * vm_corrected, corresponding to the mean Doppler velocity field \
                                  obtained by applying the ship motion correction algorithm,  \
                                  * vm_corrected_smoothed,   running mean over 3 time steps (approx 9 sec interval) \
                                      of the variable vm_corrected. Mean Doppler velocity is smoothed over 9 sec \
                                          time interval for quicklooks applications' }
                                                 
            # storing the new variables in the file
            radarData['vm_corrected']          = mdv_corr   
            radarData['vm_corrected_smoothed'] = mdv_corr_smooth          
            radarData.attrs                    = global_attributes
            radarData.to_netcdf(pathOutData+date+'_'+hour+'msm94_msm_ZEN_corrected.nc')
                                
                             
                                
            print('* plotting of the mean Doppler velocity fields')
            if np.sum(~np.isnan(mdv_corrected)) != 0:
                # plot of 2d map of mean doppler velocity and time series of correction if not existing 
                if not (os.path.exists(PathFigHour+date+'_'+hour+'_mdvCorr_2dmaps.png')) :
                       
                    # plot on mean doppler velocity time height 
                    plot_2Dmaps(datetimeRadar, \
                                    rangeRadar, \
                                        mdv_corrected, \
                                            'Mean Doppler velocity', \
                                                -6., \
                                                    4., \
                                                        100., \
                                                            2200., \
                                                                timeStart, \
                                                                    timeEnd, \
                                                                        'seismic', \
                                                                            date+'_'+hour, \
                                                                                'mdvCorr', \
                                                                                    PathFigHour)
                
            
                if not (os.path.exists(PathFigHour+date+'_'+hour+'_mdv_2dmaps.png')) :
                    # plot of 2d maps of mean Doppler velocity for the entire hour
                    plot_2Dmaps(datetimeRadar, \
                                    rangeRadar, \
                                        mdv, \
                                            'Mean Doppler velocity corrected', \
                                                -6., \
                                                    4., \
                                                        100., \
                                                            2200., \
                                                                timeStart, \
                                                                    timeEnd, \
                                                                        'seismic', \
                                                                            date+'_'+hour, \
                                                                                'mdv', \
                                                                                    PathFigHour)
            
                    
                if not (os.path.exists(PathFigHour+date+'_'+hour+'_mdv_corr_rolling2dmaps.png')) :
                    plot_2Dmaps(datetimeRadar,\
                                    rangeRadar,\
                                        mdv_roll3.values,\
                                            'mdv corrected rolling',\
                                                -5.,\
                                                    4.,\
                                                        100., \
                                                            2200.,\
                                                                timeStart, \
                                                                    timeEnd, \
                                                                        'seismic', \
                                                                            date, \
                                                                                'mdv_corr_rolling', \
                                                                                    PathFigHour)
                        
                        
                
               
                
                # plot ffts of the correction terms (w_ship and its constituents) for the height selected of the lowest chirp, 
                # and the ffts of the corrected mean Doppler velocity with and without the time shift found applied.
        
                # selecting time series for input
                if np.sum(~np.isnan(mdv_corrected[:, indHeight])) != 0:
                    indHeight = int(heightSelArray[0])
                    heightVal = rangeRadar[indHeight]
                    f_fftAnalysisForCheck(mdv[:,indHeight], \
                                              timeChirp1, \
                                                  w_ship_shifted['correction_term'].values[:,indHeight], \
                                                      w_ship_shifted['time_shifted'].values, \
                                                          mdv_corrected[:, indHeight], \
                                                              mdv_fake[:, indHeight], \
                                                                  PathFigHour, \
                                                                      date, \
                                                                          hour)
                else:
                    print('all nans in corrected mdv, skipping power spectra plot')

            else:
                print('all nans in the hour, skipping plots')
            print('end of processing for the '+date+', hour :'+hour)
            
            strasuka