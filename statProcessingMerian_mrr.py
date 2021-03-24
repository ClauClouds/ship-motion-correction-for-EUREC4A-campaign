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

################################## functions definitions ####################################

def estimate_noise_hs74(spectrum, navg, nnoise_min):
    """
    parameters: Navg = 5, nnoise_min=1 for 5 sec int time
    
    Estimate noise parameters of a Doppler spectrum.

    Use the method of estimating the noise level in Doppler spectra outlined
    by Hildebrand and Sehkon, 1974.
    For citing: f you use Py-ART in your work please cite it in your paper. While the developers appreciate mentions in the text and acknowledgements citing the paper helps more.
    For Py-ART cite our paper in the Journal of Open Research Software
    Helmus, J.J. & Collis, S.M., (2016). The Python ARM Radar Toolkit (Py-ART), a Library for Working with Weather Radar Data in the Python Programming Language. Journal of Open Research Software. 4(1), p.e25. DOI: http://doi.org/10.5334/jors.119
    For a general citation on Open Radar Software please cite Maik Heistermann in BAMS
    M. Heistermann, S. Collis, M. J. Dixon, S. Giangrande, J. J. Helmus, B. Kelley, J. Koistinen, D. B. Michelson, M. Peura, T. Pfaff, and D. B. Wolff, 2015: The Emergence of Open-Source Software for the Weather Radar Community. Bull. Amer. Meteor. Soc. 96, 117–128, doi: 10.1175/BAMS-D-13-00240.1.

    Parameters
    ----------
    spectrum : array like
        Doppler spectrum in linear units.
    navg : int, optional
        The number of spectral bins over which a moving average has been
        taken. Corresponds to the **p** variable from equation 9 of the
        article. The default value of 1 is appropriate when no moving
        average has been applied to the spectrum.
    nnoise_min : int, optional
        Minimum number of noise samples to consider the estimation valid.

    Returns
    -------
    mean : float-like
        Mean of points in the spectrum identified as noise.
    threshold : float-like
        Threshold separating noise from signal. The point in the spectrum with
        this value or below should be considered as noise, above this value
        signal. It is possible that all points in the spectrum are identified
        as noise. If a peak is required for moment calculation then the point
        with this value should be considered as signal.
    var : float-like
        Variance of the points in the spectrum identified as noise.
    nnoise : int
        Number of noise points in the spectrum.

    References
    ----------
    P. H. Hildebrand and R. S. Sekhon, Objective Determination of the Noise
    Level in Doppler Spectra. Journal of Applied Meteorology, 1974, 13,
    808-811.

    """
    sorted_spectrum = np.sort(spectrum)
    nnoise = len(spectrum)  # default to all points in the spectrum as noise

    rtest = 1+1/navg
    sum1 = 0.
    sum2 = 0.
    for i, pwr in enumerate(sorted_spectrum):
        npts = i+1
        sum1 += pwr
        sum2 += pwr*pwr

        if npts < nnoise_min:
            continue

        if npts*sum2 < sum1*sum1*rtest:
            nnoise = npts
        else:
            # partial spectrum no longer has characteristics of white noise.
            sum1 -= pwr
            sum2 -= pwr*pwr
            break

    mean = sum1/nnoise
    var = sum2/nnoise-mean*mean
    threshold = sorted_spectrum[nnoise-1]
    return mean, threshold, var, nnoise
def f_calcTimeShift_mrr(w_radar_meanCol, DeltaTimeShift, w_ship_chirp, timeSerieRadar, pathFig, date, hour):
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
    ax.set_title('covariance and sharpiness: '+date+' '+hour+':'+str(int(hour)+1)+', time lag found : '+str(DeltaTimeShift[indMin]), fontsize=fontSizeTitle, loc='left')
    ax.set_xlabel("time Shift [seconds]", fontsize=fontSizeX)
    ax.set_ylabel('w [m s-1]', fontsize=fontSizeY)
    fig.tight_layout()
    fig.savefig(pathFig+date+'_'+hour+'_timeShiftQuicklook.png', format='png')
    
    return(timeShift_chirp)       
def f_findMdvTimeSerie_VMerge(values, datetime, rangeHeight, NtimeStampsRun, pathFig):
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
    date = '20'+pd.to_datetime(datetime[0]).strftime(format="%y%m%d-%H")
    
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
    fig.savefig(pathFig+date+'_mdvSelected4Timeshift.png', format='png')
 
    return(valuesTimeSerie, i_height_sel, timeSerie, valuesColumnMean)    
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
def f_convertHourString2Int_mrr(radarFileName, lenPath):

    '''
    author: Claudia Acquistapace
    date  : 17/01/2021
    goal  : extract from radar file name the string corresponding to the hour and 
    read the corresponding integer. The function conforms to the following radar file
    name convention: 'msm94_msm_200120_000001_P07_ZEN_v2.nc'
    name for mrr = '/Volumes/Extreme SSD/ship_motion_correction_merian/mrr/input_albert/20200213_010000-FromManufacturer-processed.nc'
    input : radar file name (string)
    output: hourString, hourInt
    '''
    hourString = radarFileName[lenPath+9:lenPath+11]
    #hourString = radarFileName[-16:-14]
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
def group_consecutives(vals, step=1):
    """
    date: 29.01.2021
    author: Claudia Acquistapace
    goal: function to return list of consecutive lists of numbers from input array (number list).
    source: https://stackoverflow.com/questions/2154249/identify-groups-of-continuous-numbers-in-a-list
    input: 
        vals: ndarray
    output: 
        result: list of consecutive arrays
    """
    run = []
    result = [run]
    expect = None
    for v in vals:
        if (v == expect) or (expect is None):
            run.append(v)
        else:
            run = [v]
            result.append(run)
        expect = v + step
    return result
def integrate(x, y, axis=0, cumsum=False):
    r"""integrate y(x) over x
    \integral_x[0]^x[-1] y dx
    x[0] and x[-1] are the integral boundaries.
    Does not work well with `xarray.DataArray' objects. Use `.value' instead.
    Parameters
    ----------
    x : array_like, shape (N)
        x array
    y : array_like, shape (..., N, ...)
        y array
    axis : number, optional
        Array axis to integrate along (the default is 0)
    cumsum : bool, optional
        If True, return cumsum instead of integral. (the default is False)
    Returns
    -------
    {array_like, number}
        integral sum or cumulative integral
    """
    dx = np.empty_like(x)
    def a(s):
        """like [..., s, ...] with s on `axis` position"""
        a = [np.s_[:] for i in range(x.ndim)] # like colon (:) for all dimensions
        a[axis] = s
        return tuple(a) # numpy requires tuple for multidimensional indexing
    dx[a(0)] = (x[a(1)] - x[a(0)])/2
    dx[a(np.s_[1:-1])] = (x[a(np.s_[2:])] - x[a(np.s_[:-2])])/2.
    dx[a(-1)] = (x[a(-1)] - x[a(-2)])/2
    if cumsum:
        return np.cumsum(y*dx, axis=axis)
    else:
        return np.sum(y*dx, axis=axis)
def plot_2Dmaps_VDoppler(time,height,y,ystring,ymin, ymax, hmin, hmax, timeStartDay, timeEndDay, colormapName, date, yVarName, pathFig): 
    
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
    ax.set_title('time-vDoppler plot for the day : '+date, fontsize=fontSizeTitle, loc='left')
    ax.set_xlabel("time [hh:mm]", fontsize=fontSizeX)
    ax.set_ylabel("Height [m]", fontsize=fontSizeY)
    cbar = fig.colorbar(cax, orientation='vertical', aspect=cbarAspect)
    cbar.set_label(label=ystring, size=fontSizeCbar)
    cbar.ax.tick_params(labelsize=labelsizeaxes)
    fig.tight_layout()
    fig.savefig(pathFig+date+'_'+yVarName+'_2dmaps.png', format='png')        
def f_calcProminence(spec, time, navg, nnoise_min):
    '''
    date: 16 March 2021
    author: Claudia Acquistapace
    goal: calculate max, min prominence of each spectra line and their difference for 1 hour of MRR data
    input: spectra(time, height, Vdoppler)
           time, 
           navg, scalar, number of averages to be done to calculate HS spectra
           nnoise_min, scalar, n_noise_min to provide for calculating HS spectra
    output: deltaProminence, 
            prominenceMax, 
            prominenceMin
    '''
    # calculating  linear spectra
    specLin         = 10.**(spec/10.)

    # calculating HS noise for the spectra
    dimTime         = spec.shape[0]
    dimHeight       = spec.shape[1]
    dimVDoppler     = spec.shape[2]

    # defining matrices
    meanNoise       = np.zeros((dimTime, dimHeight, dimVDoppler))
    specNoNoise     = np.zeros((dimTime, dimHeight, dimVDoppler))
    prominenceMax   = np.zeros((dimTime, dimHeight))
    prominenceMin   = np.zeros((dimTime, dimHeight))


    # calculating HS noise for the spectra 
    for indTime in range(dimTime):
        for indHeight in range(dimHeight):
            # calculating HS mean and max noise level
            noise, thr, var, nn                 = estimate_noise_hs74(specLin[indTime, indHeight, :],\
                                                                      navg, \
                                                                      nnoise_min) 
            # storing mean noise level
            meanNoise[indTime, indHeight, :]    = np.repeat(noise, dimVDoppler)

            # calculating spectra minus noise
            specNoNoise[indTime, indHeight,:]   = np.repeat(noise, dimVDoppler)
            specNoNoise[indTime, indHeight, specLin[indTime, indHeight, :] > noise] = specLin[indTime, indHeight, specLin[indTime, indHeight, :] > noise]

            # calculate peaks and prominence on the spec-noise 
            peaks, _    = find_peaks(specNoNoise[indTime, indHeight, :]) 
            prominences = peak_prominences(specNoNoise[indTime, indHeight, :], peaks)[0]

            #storing max and min prominence
            if len(prominences) != 0:
                prominenceMax[indTime,indHeight]    = np.nanmax(prominences)
                prominenceMin[indTime,indHeight]    = np.nanmin(prominences)
            else: 
                prominenceMax[indTime,indHeight]    = 0.
                prominenceMin[indTime,indHeight]    = 0.           

    # calculate difference between max and min prominences
    deltaProminence = prominenceMax - prominenceMin
    
    return(deltaProminence, prominenceMax, prominenceMin)
def get_unique_numbers(numbers):
    list_of_unique_numbers = []
    unique_numbers = set(numbers)
    for number in unique_numbers:
        list_of_unique_numbers.append(number)
    return list_of_unique_numbers
def f_calcTimeResParam(DataList, dataPath):
    """
    author: Claudia Acquistapace
    date: 20/01/2021
    goal: function to calculate the integration time, and consequently 
    the parameters to derive HS noise needed to filter the noise
    input: DataList - list of hourly files from MRR
    output: 
        dataArray: array with all data-hour strings
        timeResArr: array containing all integration time of each hour
        nAvgArr: averaging time array used in each hour
        nnoise_minArr: parameter nnoise_min array to be used in each hour
        
    """
    # def list of time resolutions
    timeResArr = []
    # def list of date array
    datehhArr  = []
    # loop on all files
    for indFile in range(0,len(DataList)):
        
        data = xr.open_dataset(DataList[indFile])
        time = data['time'].values
        # Calculating difference list 
        diff_list = [] 
        for i in range(1, len(time)): 
            diff_list.append(time[i] - time[i-1]) 

        # find unique numbers in diff_list
        time_res_found = get_unique_numbers(diff_list)
        timeResArr.append(time_res_found)

        # saving list with file date and hour
        datehhArr.append(DataList[indFile][len(dataPath):len(dataPath)+11])
    
    # def averaging time array
    nAvgArr = np.zeros((len(datehhArr)))
    # def n_noise_min arr
    nnoise_minArr = np.zeros((len(datehhArr)))
    for i in range(len(datehhArr)):
        if (len(timeResArr[i]) == 0):
            print('file with zero dimension :'+DataList[indFile])
            nAvgArr[i]= np.nan
            nnoise_minArr[i]= np.nan
        else:
            nAvgArr[i] = timeResArr[i][0]
            nnoise_minArr[i]= 1      
    return(timeResArr, datehhArr, nAvgArr, nnoise_minArr)

################ User data : insert here paths, filenames and parameters to set ########################
# days without modelled data: [datetime(2020,1,22), datetime(2020,1,25), datetime(2020,2,2), datetime(2020,2,3)]
### instrument position coordinates [+5.15m; +5.40m;−15.60m]
r_MRR           = [7.18 , 4.92, -17.28] # [m]

# generating array of days for the dataset
Eurec4aDays     = pd.date_range(datetime(2020,1,19),datetime(2020,2,19),freq='d')
NdaysEurec4a    = len(Eurec4aDays)


# reading file list and its path
# establishing paths for data input and output
pathFolderTree  = '/Volumes/Extreme SSD/ship_motion_correction_merian'
pathNcDataAnc   = pathFolderTree+'/ncdf_ancillary/'
shipFile        = pathFolderTree+'/ship_data/new/shipData_all2.nc'
pathRadar       = pathFolderTree+'/mrr/input_albert/'
pathOutData     = pathFolderTree+'/mrr/output_clau/'

DataList        = np.sort(glob.glob(pathRadar+'*.nc'))

# call function to calculate time resolution and processing parameters for all dates 
timeRes, dates, navgArr, nnoise_arr = f_calcTimeResParam(DataList, pathRadar)
#%%

# loop on files from Albert
for indHour in range(len(DataList)):
    date_hour       = dates[indHour]
    yy              = date_hour[0:4]
    mm              = date_hour[4:6]
    dd              = date_hour[6:8]
    hh              = date_hour[9:11]
    
    # setting dates strings
    date            = dd+mm+yy      #'04022020'
    dateRadar       = yy[0:2]+mm+dd #'200204'
    dateReverse     = yy+mm+dd      #'20200204'   
    

    pathFig         = pathFolderTree+'plots/mrr/'+yy+'/'+mm+'/'+dd+'/'
    radarFileName   = DataList[indHour]
    print('file :', DataList[indHour])
    
    # reading corresponding metek file
    MetekFile       = pathFolderTree+'/mrr/'+yy+'/'+mm+'/'+dd+'/'+yy+mm+dd+'_'+hh+'0000.nc'
    
    # read radar height array for interpolation of model data from one single radar file
    #radarDatatest          = xr.open_dataset(radarFileName)
    #heightTest             = radarDatatest['Height'].values
    
    # reading hour string and int quantity
    hour, hourInt   =  f_convertHourString2Int_mrr(radarFileName, len(pathRadar))

    # reading integration time and related settings 
    timeReshour     = timeRes[indHour]
    navgArrhour     = navgArr[indHour]
    nnoise_hour     = nnoise_arr[indHour]
    
      
    
    print('file selected: '+date_hour)
    
    print('----------------------')
    
    if os.path.isfile(pathOutData+date+'_'+hour+'_preprocessedClau_4Albert.nc'):
        print(date+'_'+hour+' - already processed')
    else:    
        print('processing the file')
        print('1) removing interference pattern for the selected hourly file')
        # interference removal is based on a combination of filtering masks, based on variables that are different over noise and over signal
        
        # reading paramter threshold for applying the masks associated with integration time
        if timeReshour[0] == 1:
            thr_prominence  = 2.5
            peaks_threshold = 2.5
            Npeaksout       = 4
        elif timeReshour[0] == 5:
            peaks_threshold = 2.5
            thr_prominence  = 1.  
            Npeaksout       = 4
        elif timeReshour[0] == 10:
            peaks_threshold = 2.5
            thr_prominence  = 1.            
            Npeaksout       = 4
            
        # reading mdv and spectra from original metek data
        dataMetek      = xr.open_dataset(MetekFile)
        height         = dataMetek['range'].values
        mdv_mrr        = dataMetek['VEL'].values
        time           = dataMetek['time'].values
        spec           = dataMetek['spectrum_raw'].values
        
        
        # calculating doppler velocity array
        dimVDoppler     = spec.shape[2]
        deltaF          = 30.52 # Hz
        delta_ni        = 0.18890 # ms^-1
        vDoppler        = np.zeros(dimVDoppler)
        for ind in range(dimVDoppler):
                vDoppler[ind] = ind*delta_ni

        
        dimTime   = len(time)
        dimHeight = len(height)
        dimv      = len(vDoppler)
        units_time      = 'seconds since 1970-01-01 00:00:00' 
        datetimeM       = pd.to_datetime(time, unit ='s', origin='unix')        
        
        # generating empty matrices where to store data filtered
        VelFiltered  = np.zeros((dimTime, dimHeight))
        specFiltered = np.zeros((dimTime, dimHeight, dimv))
        
        # plotting spectrograms to check if data are nan or not
        indTime = 1900
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
        cax = ax.pcolormesh(vDoppler, height, spec[indTime, :, :], vmin=0., vmax=30., cmap='jet')
        ax.set_ylim(0.,1200.)                                               # limits of the y-axesn  cmap=plt.cm.get_cmap("viridis", 256)
        ax.set_xlim(0., 12.)                                 # limits of the x-axes
        #ax.set_title(' day : '+strTitleSignal, fontsize=fontSizeTitle, loc='left')
        ax.set_xlabel("VDoppler [$ms^{-1}$]", fontsize=fontSizeX)
        ax.set_ylabel("height [m]", fontsize=fontSizeY)
        cbar = fig.colorbar(cax, orientation='vertical', aspect=cbarAspect)
        cbar.set_label(label='Power [dB]', size=fontSizeCbar)
        cbar.ax.tick_params(labelsize=labelsizeaxes)
        fig.tight_layout()
        fig.savefig(pathFig+'fig_mrr_proc_height_spectrogram_signal.png', format='png')        
        
        
        
        print('calculating prominence difference for all spectra')
        # calculating max, min, and delta prominence from the spectra
        deltaP, maxP, minP = f_calcProminence(spec, time, navgArrhour, nnoise_hour)


        
        # testing conditions for prominence mask for time res = 1s
        maskProminence = np.zeros((dimTime, dimHeight))
        maskProminence[(deltaP > thr_prominence)] = 1   # selectimg as good data all deltaprominence >2.
        

        
        # calculating mask based on profiles of the difference of consecutive w values in the profile below 600 m
        indHeightTest = np.where(height <600)[0]
        mask_profiles = np.zeros((dimTime, dimHeight))
        for ind in range(dimTime):
    
            #calculate number of peaks above and below Threshold
            peaks_gt_thr = np.where(np.ediff1d(-mdv_mrr[ind,indHeightTest]) > peaks_threshold)[0]
            peaks_lt_thr = np.where(np.ediff1d(-mdv_mrr[ind,indHeightTest]) < -peaks_threshold)[0]
    
            if (len(peaks_gt_thr) > Npeaksout) * (len(peaks_lt_thr) > Npeaksout):
                mask_profiles[ind,:] = 0
            else:
                mask_profiles[ind,:] = 1
    
        
        # applying condition of mask prominence
        mask_total = np.zeros((dimTime, dimHeight))
        mask_total[maskProminence == 1. ] = 1.
        
        # applying condition of mask_profiles
        mask_total[(mask_profiles == 0.)] = 0.
        

        print('applying third spatial filter to the mask')
        #applying spatial filtering mask
        coordinatesPoints = np.where(mask_total == 1)
        NpointsSelected = len(coordinatesPoints[0])
        for indPoint in range(NpointsSelected):
            indTime = coordinatesPoints[0][indPoint]
            indHeight = coordinatesPoints[1][indPoint]
            NoiseSubMatrix = mask_total[indTime-1:indTime+1, indHeight-3:indHeight+3]
            if len(np.where(NoiseSubMatrix == 1)[0]) < 3:
                mask_total[indTime, indHeight] = 0   
        
        # filtering variables based on the mask
        for indTime in range(dimTime):
            for indHeight in range(dimHeight):
                if mask_total[indTime, indHeight] == 0.:
                    VelFiltered[indTime, indHeight] = np.nan
                    spec[indTime, indHeight, :] = np.nan
                else:
                    VelFiltered[indTime, indHeight] = mdv_mrr[indTime, indHeight]



        print('filtering interference noise - completed ')
        print('#########################################')
        
        
        # if integration time == 1s we now apply the ship motion corrections
        if timeReshour[0] == 1:
            
            print('time integration: 1s - application of ship motion corrections')
            # check if file for w-ship correction has already been processed or not
            if os.path.isfile(pathNcDataAnc+'ancillary_mrr/'+dd+mm+yy+'_'+hour+'_w_ship_mrr.nc'):
                
                print('ship motions already calculated')
                w_ship_shifted = xr.open_dataset(pathNcDataAnc+'ancillary_mrr/'+dd+mm+yy+'_'+hour+'_w_ship_mrr.nc')
        
            else:
        
                # check if folder for plots and for outputs exists, otherwise creates it
                PathFigHour     = pathFig+hour+'/'
                Path(PathFigHour).mkdir(parents=True, exist_ok=True)  
                Path(pathNcDataAnc+'ancillary_mrr/').mkdir(parents=True, exist_ok=True)  
             
                # setting starting and ending time for the hour an
                timeStart    = datetime(int(yy),int(mm), int(dd),hourInt,0,0)
                if hour != '23':
                    timeEnd      = datetime(int(yy),int(mm), int(dd),hourInt+1,0,0)
                else:
                    timeEnd      = datetime(int(yy),int(mm), int(dd),hourInt,59,59)
                timeArr         = pd.to_datetime(np.arange(timeStart, timeEnd, timedelta(seconds=1), dtype='datetime64'))
        
                print('processing '+date+', hour :'+hour)
                print('timestart' , timeStart)
                print('timeEnd', timeEnd)
                print('**************************************')
                print('* reading correction terms for the day/hour :'+yy+'-'+mm+'-'+dd+', hour: '+hour)
                '''reading correction terms for the day and interpolating them on the 0.25s time resolution'''
                #                   ----------------------------
                print(' -- reading and resampling V_wind_s dataset')
                if (date == '22012020') or \
                    (date == '25012020') or \
                        (date == '02022020') or \
                            (date == '03022020'):
                                # saving variables of wind_s in a xarray dataset
                                dims2             = ['time','height']
                                coords2           = {"time":timeArr, "height":height}
                                
                                v_wind_s_x        = xr.DataArray(dims=dims2, coords=coords2, data=np.full([len(timeArr), len(height)], np.nan),
                                                         attrs={'long_name':'zonal wind speed profile in ship reference system from ICON-LEM 1.25 km',
                                                                'units':'m s-1'})
                                v_wind_s_y        = xr.DataArray(dims=dims2, coords=coords2, data=np.full([len(timeArr), len(height)], np.nan),
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
                V_rot_dataset      = xr.open_dataset(pathNcDataAnc+'V_rot_V_trasl_dataset_mrr.nc')
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
                
                print('---- end of reading dataset for calculating ship motion correction terms  ----')
                
                # check on dimensions: if V_windS has dimensions different than others, interpolate on correct time grid
                if len(Vrot_hour['time'].values) != len(V_windS_hour['time'].values):
                    timeInterp   = Vrot_hour['time'].values
                    V_windS_hour = V_windS_hour.interp(time=timeInterp) 
                    
                if (len(V_windS_hour['height'].values) != 550):
                    V_windS_hour = V_windS_hour.interp(height=height) 
                    
                    
                    
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
                vs_z               = np.repeat(np.repeat(0., len(V_windS_hour['vs_y'].values)), 128).reshape(len(V_windS_hour['vs_y'].values), 128)
                V_wind_s           = np.array([vs_x,vs_y,vs_z])
            
                # V course, V rotation and V translation 
                V_course_vec       = np.array([V_course_x, V_course_y, np.repeat(0, len(V_course_y))])
                V_rot_vec          = np.array([Vrot_hour['Vrot_x2'].values, Vrot_hour['Vrot_y2'].values, Vrot_hour['Vrot_z2'].values])
                V_trasl_vec        = np.array([np.repeat(0., len(Vrot_hour['V_trasl'].values)), \
                                              np.repeat(0., len(Vrot_hour['V_trasl'].values)), \
                                              Vrot_hour['V_trasl'].values]).reshape(3,len(Ep[0,:]))
                
                # application of the formula 29 (or 7) from the theory attached
                V_radar_vec        = V_rot_vec + V_trasl_vec + V_course_vec
                num_termC          = V_wind_s - np.repeat(V_radar_vec, 128).reshape(3, dimTime, 128)
                EP_matrix          = np.repeat(Ep_vec, 128).reshape(3, dimTime, 128)
                num_correction     = np.nansum(num_termC*EP_matrix, axis=0) 
                den_correction     = np.repeat(den_scalar_product, 128).reshape(dimTime, 128)
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
                V_trasl           = xr.DataArray(dims=dim3, coords=coords3, data=V_trasl_vec,
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
                                    'DATA_DESCRIPTION' : 'hourly MRR correction terms on Maria S. Merian ship during EUREC4A campaign',
                                    'DATA_DISCIPLINE'  : 'Atmospheric Physics - Remote Sensing Radar Profiler',
                                    'DATA_GROUP'       : 'Experimental;Profile;Moving',
                                    'DATA_LOCATION'    : 'Research vessel Maria S. Merian - Atlantic Ocean ',
                                    'DATA_SOURCE'      : 'Radar.Standard.Moments.Ldr for ship data;run by uni_koeln https://github.com/igmk/w-radar/tree/master/scripts',
                                    'DATA_PROCESSING'  : 'https://github.com/ClauClouds/ship-motion-correction-for-EUREC4A-campaign',
                                    'INSTRUMENT_MODEL' : 'K band radar - MRR from Metek',
                                     'COMMENT'         : 'correction terms tensors based on algorithm described in ship_motion_correction.pdf' }
                w_ship_dataset    = xr.Dataset(data_vars = variables,
                                                  coords = coords2,
                                                   attrs = global_attributes)
                
                # assign the central time stamp to the the correction terms
                w_ship_shifted    = f_shiftTimeDataset(w_ship_dataset)
                
                # save ship motion correction terms in a separate file
                w_ship_shifted.to_netcdf(pathNcDataAnc+'ancillary_mrr/'+date+'_'+hour+'_w_ship_mrr.nc')
                
        
            
            # reading ship data 
            timeShip          = w_ship_shifted['time_shifted'].values
            w_ship            = w_ship_shifted['correction_term'].values
            denTimeSerie      = w_ship_shifted['den_scalar_product'].values
            
    
            # assigning lenght of the mean doppler velocity time serie for calculating time shift 
            # with 3 sec time resolution, 200 corresponds to 10 min
            NtimeStampsRun   = 120      
                
    
            # search for at least 2 min of consecutive w obs in the chirp using noise filtered data
            #dataFilter = xr.open_dataset('/Volumes/Extreme SSD/ship_motion_correction_merian/mrr/data_filtered/20200213_01_MRR_Metek_Raprom.nc')
            #velFilter = dataFilter['W'].values
            #velFilter = VelFiltered
            #datetimeFilter = pd.to_datetime(dataFilter['time'].values, unit ='s', origin='unix')                
            #heightFilter = dataFilter['Height'].values
            w_radar, indHeightBest, timeRadarSel, w_radar_meanCol = f_findMdvTimeSerie_VMerge(-VelFiltered, \
                                                                    datetimeM, \
                                                                    height, \
                                                                    NtimeStampsRun, \
                                                                    pathFig)
            
            
            # reading correction time serie corresponding to the selected height in the selected time interval
            i_time_ship_best  = (pd.to_datetime(timeShip) >= timeRadarSel[0]) * (pd.to_datetime(timeShip) <= timeRadarSel[-1])
            w_ship_best       = w_ship[i_time_ship_best, indHeightBest]
            
        
            # time shift array to be tested 
            DeltaTmin        = -3.
            DeltaTmax        = 3.
            res              = 0.05
            DimDeltaT        = (DeltaTmax- DeltaTmin)/res
            DeltaTimeShift   = np.arange(DeltaTmin, DeltaTmax, step=res)
            
            # calculating time shift for the chirp
            if np.sum(np.where(~np.isnan(w_radar))) != 0:
                
                # excluding possible nan values from the best selected time serie of w_mrr for doing the comparison
                i_valid    = np.where(~np.isnan(w_radar))
                w_valid    = w_radar[i_valid]
                time_valid = timeRadarSel[i_valid]
            
                if (len(w_valid) > 10):
                    
                    # interpolating mean radar mdv speed on the ship time resolution for time shift calculation
                    CS_rad                  = CubicSpline(time_valid, w_valid)
                    WChirpshipRes           = CS_rad(pd.to_datetime(timeShip[i_time_ship_best]))      
                    
                    
                    # calculating time shift
                    timeShiftArray          = f_calcTimeShift_mrr(WChirpshipRes, \
                                                              DeltaTimeShift, \
                                                              w_ship_best, \
                                                              pd.to_datetime(timeShip[i_time_ship_best]), \
                                                              pathFig, \
                                                              date, \
                                                              hour)
                else:
                    timeShiftArray          = np.nan                            
            else:
                
                timeShiftArray              = np.nan
                                
        
            print('- recalculating exact time for ship data ')
            # recalculating exact time including time shift due to lag
            if ~np.isnan(timeShiftArray):
                timeExact  = pd.to_datetime(datetimeM) - timedelta(seconds=timeShiftArray)
            
            else:
                timeExact  = pd.to_datetime(datetimeM) 
            
            timeExactFinal = pd.to_datetime(timeExact)
            
            # interpolating ship data on radar time 
            Cs_ship        = CubicSpline(timeShip, w_ship[:, indHeightBest])
            W_shipRadar    = Cs_ship(pd.to_datetime(datetimeM))
            
            # interpolating ship data on the exact time again, with the right time found after the calculating
            # and the addition of the time shift     
            CS_exact        = CubicSpline(timeExact, W_shipRadar)
            W_ship_exact    = CS_exact(pd.to_datetime(datetimeM))
            timePlot        = pd.to_datetime(timeExact)
            TimeBeginPlot   = timeRadarSel[0]
            TimeEndPlot     = timeRadarSel[-1]
            
            labelsizeaxes   = 12
            fontSizeTitle   = 12
            fontSizeX       = 12
            fontSizeY       = 12
            cbarAspect      = 10
            fontSizeCbar    = 12
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
            ax.plot(timeRadarSel, w_radar, linewidth = 1, color='red', label='w_radar at one height') 
            ax.plot(pd.to_datetime(datetimeM), W_shipRadar, color='purple', linewidth=1., linestyle=':', label='w_ship original')
            ax.plot(timePlot, W_shipRadar, color='blue', label='w_ship shifted of deltaT found')
            ax.scatter(pd.to_datetime(datetimeM), W_ship_exact, color='green', label='w_ship shifted interpolated on radar exact time')
            ax.set_ylim(-8.,2.)   
            ax.legend(frameon=False)
            ax.set_xlim(TimeBeginPlot,TimeEndPlot)                                 # limits of the x-axes
            ax.set_title('velocity for time delay calculations : '+date+' '+hour+':'+str(int(hour)+1)+' shift = '+str(timeShiftArray), fontsize=fontSizeTitle, loc='left')
            ax.set_xlabel("time [hh:mm:ss]", fontsize=fontSizeX)
            ax.set_ylabel('w [m s-1]', fontsize=fontSizeY)
            fig.tight_layout()
            fig.savefig(pathFig+date+'_'+hour+'_quicklook_correction.png', format='png')
            
            print(' ### time shifts found for the hourly file: ', timeShiftArray)  
    
                        
            print('* calculating corrected mean Doppler velocity')
            # calculating now correction term based on the time shift found
            mdv_corrected = np.zeros((len(time), len(height)))
            
            # loop on range gates: for each height, find the exact radar time
            for indHeight in range(len(height)):
                mdvTimeSerie = -VelFiltered[:, indHeight]
                
                # interpolating w_ship and correction denominator on radar time stamps
                Cs_ship        = CubicSpline(timeShip, w_ship[:, indHeight])
                W_shipRadar    = Cs_ship(pd.to_datetime(datetimeM))      
                Cs_den         = CubicSpline(timeShip, denTimeSerie)
                denShipRadar   = Cs_den(pd.to_datetime(datetimeM)) 
                
                
                # interpolating w_ship shifted of the time gap on the radar time stamps
                CS_exact_w     = CubicSpline(timeExactFinal, W_shipRadar)
                W_ship_exact   = CS_exact_w(pd.to_datetime(datetimeM))
    
                # interpolating den shifted of the time gap on the radar time stamps
                CS_exact_den   = CubicSpline(timeExactFinal, denShipRadar)
                den_exact      = CS_exact_den(pd.to_datetime(datetimeM))
                
                # calculating corrected mdv for radar time stamps
                mdv_corrected[:, indHeight] = - mdvTimeSerie/den_exact + W_ship_exact/den_exact
              
               
            # plot of the velocities 
            
            timeStart = pd.to_datetime(datetime(2020,2,13,1,31,0))#datetime(2020,1,20,6,19,0)
            timeEnd =pd.to_datetime(datetime(2020,2,13,1,35,0))#datetime(2020,1,20,6,23,0)
            plot_2Dmaps_VDoppler(datetimeM,height,-mdv_mrr,'Vd [$ms-1$]',0.,-8., 0., 1200., timeStart, timeEnd, 'jet', date,  'mdv_original', pathFig)       
            plot_2Dmaps_VDoppler(datetimeM,height,mdv_corrected,'Vd [$ms-1$]',0.,-8., 0., 1200., timeStart, timeEnd, 'jet', date,  'mdv_corrected', pathFig)       
            
    
            print('* Applying running mean on 3 time steps for improving data quality')
            if np.sum(~np.isnan(mdv_corrected)) != 0:
                df        = pd.DataFrame(mdv_corrected, index=datetimeM, columns=height)
                mdv_roll3 = df.rolling(window=3,center=True, axis=0, min_periods=1).apply(lambda x: np.nanmean(x)) 
            else:
                mdv_r3    = np.zeros((len(time), len(height)), dtype=float)
                mdv_r3[:] = np.nan
                mdv_roll3 = pd.DataFrame(mdv_r3, index=time, columns=height)
            
            
            # plotting rolling mean of the corrected ship motions
            plot_2Dmaps_VDoppler(datetimeM,height,mdv_roll3.values,'Vd [$ms-1$]',0.,-8., 0., 1200., timeStart, timeEnd, 'jet', date,  'mdv_corrected_smooth', pathFig)       

    

            # plot quicklook of filtered and corrected mdv for checking
            labelsizeaxes   = 14
            fontSizeTitle = 16
            fontSizeX = 16
            fontSizeY = 16
            cbarAspect = 10
            fontSizeCbar = 16
            rcParams['font.sans-serif'] = ['Tahoma']
            matplotlib.rcParams['savefig.dpi'] = 100
            
            grid = True
            fig, axs = plt.subplots(2, 2, figsize=(14,9), sharey=True, constrained_layout=True)
            
            # build colorbar
            mesh = axs[0,0].pcolormesh(datetimeM, height, -mdv_mrr.T, vmin=-10, vmax=2, cmap='jet', rasterized=True)
            axs[0,0].set_title('Original')
            axs[0,0].spines["top"].set_visible(False)  
            axs[0,0].spines["right"].set_visible(False)  
            axs[0,0].get_xaxis().tick_bottom()  
            axs[0,0].get_yaxis().tick_left() 
            axs[0,0].set_xlim(datetimeM[0], datetimeM[-1])

            mesh = axs[0,1].pcolormesh(datetimeM, height, -VelFiltered.T, vmin=-10, vmax=2,  cmap='jet', rasterized=True)
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
            mesh = axs[1,0].pcolormesh(datetimeM, height, -VelFiltered.T, vmin=-10, vmax=2, cmap='jet', rasterized=True)
            axs[1,0].set_title('Original')
            axs[1,0].spines["top"].set_visible(False)  
            axs[1,0].spines["right"].set_visible(False)  
            axs[1,0].get_xaxis().tick_bottom()  
            axs[1,0].get_yaxis().tick_left()     
            axs[1,0].set_xlim(timeStart, timeEnd)

            mesh = axs[1,1].pcolormesh(datetimeM, height, mdv_corrected.T, vmin=-10, vmax=2, cmap='jet', rasterized=True)
            axs[1,1].set_title('ship motions corrected')
            axs[1,1].spines["top"].set_visible(False)  
            axs[1,1].spines["right"].set_visible(False)  
            axs[1,1].get_xaxis().tick_bottom()  
            axs[1,1].get_yaxis().tick_left()     
            axs[1,1].set_xlim(timeStart, timeEnd)  
            [a.xaxis.set_major_formatter(mdates.DateFormatter('%M:%S')) for a in axs[1,:].flatten()]
            fig.colorbar(mesh, ax=axs[:,-1], label='Doppler velocity [$ms^{-1}$]', location='right', aspect=60, use_gridspec=grid)
            for ax, l in zip(axs.flatten(), ['(a)', '(b)', '(c)', '(d)']):
                ax.text(0, 1.02, l,  fontweight='black', transform=ax.transAxes)
            fig.savefig(pathFig+date+'_'+hour+'vel_orig_filtered_corr.png')
            fig.savefig(pathFig+date+'_'+hour+'vel_orig_filtered_corr.pdf')
        
    
            
            print('applying the Doppler shift found to the Doppler spectra postprocessed by Albert')
            # shifting now doppler spectra of the Doppler spectra from Albert's files
            # reading spec data from files preprocessed with Albert's script
            dataAlbert     = xr.open_dataset(radarFileName)
            vDoppler       = dataAlbert['3Range'].values
            specAl         = dataAlbert['spectral reflectivity'].values            
            
            # selecting doppler spectra of the bins of the corrected mean doppler velocity that are non nan
            specShifted =  np.zeros((len(time), len(height), len(vDoppler)))
            specShifted.fill(np.nan)    
            dv = vDoppler[4] - vDoppler[3] # spectral resolution
        
            for indTime in range(len(time)):
                for indHeight in range(len(height)):
                    # calculating exact doppler shift 
                    DopplerShift = mdv_corrected[indTime, indHeight]- (-VelFiltered[indTime, indHeight])
                    
                    if ~np.isnan(DopplerShift):
                        # calculating the number of bins to shift in the spectrum corresponding to the Doppler shift found
                        DS = int(round(DopplerShift/dv))      
                        
                        # shift in the spectra of the found number of bins and saving in spec_shifted
                        specShifted[indTime, indHeight, :] = np.roll(specAl[indTime,indHeight,:], -DS)
                    else:
                        specShifted[indTime, indHeight, :] = specAl[indTime,indHeight,:]
            
         
            
            #plot quicklooks of the spectra at given heights where the signal is shifted and original, with printed values of corresponding mean Doppler velocities
            HeightSel = [70., 300., 600., 800.]
            colorSel = ['red','blue','green','purple']
            labelsizeaxes   = 16
            fontSizeTitle   = 16
            fontSizeX       = 16
            fontSizeY       = 16
            cbarAspect      = 10
            fontSizeCbar    = 16
            # selecting time to be plotted :
            i_time_sel = 1900 #f_closest(datetimeM, timeRadarSel[0])
            
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
            for indH in range(len(HeightSel)):
                ind_closest= f_closest(height, HeightSel[indH])
                mdv_orig_str = str(round(-VelFiltered[i_time_sel, ind_closest],2))
                mdv_corr_str = str(round(mdv_roll3.values[i_time_sel, ind_closest]))
                label_orig_Str = str(round(HeightSel[indH],3))+' m, mdv orig = '+mdv_orig_str
                label_corr_Str = str(round(HeightSel[indH],3))+' m, mdv corr = '+mdv_corr_str
                ax.plot(-vDoppler, np.log10(specAl[i_time_sel, ind_closest, :]), label=label_orig_Str, linestyle=':', color=colorSel[indH])
                ax.plot(-vDoppler, np.log10(specShifted[i_time_sel, ind_closest, :]), label=label_corr_Str, color=colorSel[indH])
            
            ax.legend(frameon=False, prop={'size': 12})
            #ax.set_title(' day : '+strTitleSignal, fontsize=fontSizeTitle, loc='left')
            ax.set_xlabel("VDoppler [$ms^{-1}$]", fontsize=fontSizeX)
            ax.set_ylabel("Power [dB]", fontsize=fontSizeY)
            ax.set_xlim(-12., 0.)
            #ax.set_ylim(10.**(-9), 10.**(-6))
            ax.set_ylim(-8.0)
            fig.tight_layout()
            fig.savefig(pathFig+date_hour+'_spectra_selectedHeights_sample_shift.png', format='png')    
          
            # saving data in ncdf file
            dimt              = ['height']
            dims2             = ['time','height']
            dimsAll           = ['time','height', 'VDoppler']
            coordt            = {"height":height}
            coords2           = {"time":dataAlbert['time'].values, "height":height}
            coordsAll         = {"time":dataAlbert['time'].values, "height":height, "VDoppler":vDoppler}
            
            mdv_orig          = xr.DataArray(dims=dims2, coords=coords2, data=-mdv_mrr,
                                 attrs={'long_name':'Original Mean Doppler velocity with sign flipped with respect to Metek conventions. Now downward velocities are negative',
                                        'units':'ms-1'})
            #            mdv_filt          = xr.DataArray(dims=dims2, coords=coords2, data=-VelFiltered ,
            #                                     attrs={'long_name':'Original Mean Doppler velocity with sign flipped with respect to Metek conventions. Now downward velocities are negative',
            #                                            'units':'ms-1'})
            mdv_corr          = xr.DataArray(dims=dims2, coords=coords2, data=mdv_corrected,
                                 attrs={'long_name': 'Mean Doppler velocity corrected for ship motions with sign flipped with respect to Metek conventions. Now downward velocities are negative',
                                        'units':'ms-1'})
            mdv_corr_smooth   = xr.DataArray(dims=dims2, coords=coords2, data=mdv_roll3.values,
                                 attrs={'long_name': 'Mean Doppler velocity corrected for ship motions and smoothed with sign flipped with respect to Metek conventions. Now downward velocities are negative',
                                        'units':'ms-1'})           
            mask              = xr.DataArray(dims=dims2, coords=coords2, data=mask_total,
                                 attrs={'long_name': 'Mask for filtering noise from interference used to calculate ship motion correction',
                                        'units':''}) 
            spec_shifted      = xr.DataArray(dims=dimsAll, coords=coordsAll, data=specShifted,
                                 attrs={'long_name':'spectral reflectivity shifted after ship motion corrections',
                                        'units':'m-1'})    
            variables         = {'mdv_orig':mdv_orig,
                             'mdv_corr':mdv_corr, 
                             'mask':mask,
                             'mdv_corr_smooth':mdv_corr_smooth,
                             'spec_shifted':spec_shifted
                             }
            
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
                             'DATA_DESCRIPTION' : 'hourly MRR measurements on Maria S. Merian (msm) ship during EUREC4A campaign',
                             'DATA_DISCIPLINE'  : 'Atmospheric Physics - Remote Sensing Radar Profiler',
                             'DATA_GROUP'       : 'Experimental;Profile;Moving',
                             'DATA_LOCATION'    : 'Research vessel Maria S. Merian - Atlantic Ocean ',
                             'DATA_SOURCE'      : 'MRR data postprocessed by Metek software',
                             'DATA_PROCESSING'  : 'additional data processing is performed for applying ship motion correction and to filter interference patterns\
                                 the code used is available at https://github.com/ClauClouds/ship-motion-correction-for-EUREC4A-campaign',
                             'INSTRUMENT_MODEL' : '24.23 GHz (K-band) radar',
                             'COMMENT'          : '' }
            
            MRRData      = xr.Dataset(data_vars   = variables,
                                       coords = coordsAll,
                                       attrs  = global_attributes)                                     
            # storing the new variables in the file
            #RRData['calibration_constant']  = data['calibration_constant']   
            MRRData.attrs                    = global_attributes
            MRRData['time'].attrs          = {'units':'seconds since 1970-01-01 00:00:00'}
            MRRData.to_netcdf(pathOutData+date+'_'+hour+'_preprocessedClau_4Albert.nc')
            
 
            
            